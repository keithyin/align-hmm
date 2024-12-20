use core::str;
use std::{
    collections::HashMap,
    fmt::Debug,
    sync::{Arc, Mutex},
    thread,
};

use crossbeam::channel;
use gskits::{
    fastx_reader::{fasta_reader::FastaFileReader, read_fastx},
    pbar::{get_spin_pb, DEFAULT_INTERVAL},
};
use ndarray::Array1;
use rust_htslib::bam::{self, Read};

use crate::{
    cli::TrainingParams,
    common::TransState,
    dataset::{align_record_read_worker, read_refs, train_instance_worker},
    em_training::model::{decode_2_bases, decode_emit_base, encode_2_bases},
    hmm_model::HmmModel,
};

use crate::common::TrainInstance;

fn train_model(
    aligned_bam: &str,
    ref_fasta: &str,
    arrow_hmm: &mut HmmModel,
    dw_boundaries: &Vec<u8>,
) -> anyhow::Result<()> {
    let fasta_rader = FastaFileReader::new(ref_fasta.to_string());
    let refname2refseq = read_fastx(fasta_rader)
        .into_iter()
        .map(|read_info| (read_info.name, read_info.seq))
        .collect::<HashMap<_, _>>();

    let mut aligned_bam_reader = bam::Reader::from_path(aligned_bam).unwrap();
    aligned_bam_reader.set_threads(10).unwrap();
    let aligned_bam_header = bam::Header::from_template(aligned_bam_reader.header());
    let aligned_bam_header_view = bam::HeaderView::from_header(&aligned_bam_header);

    let mut tid2refname = HashMap::new();

    let pbar = get_spin_pb(
        format!("training... using {}", aligned_bam),
        DEFAULT_INTERVAL,
    );

    for align_record in aligned_bam_reader.records() {
        pbar.inc(1);
        let align_record = align_record.unwrap();
        let tid = align_record.tid();
        if !tid2refname.contains_key(&tid) {
            tid2refname.insert(
                tid,
                String::from_utf8(aligned_bam_header_view.tid2name(tid as u32).to_vec()).unwrap(),
            );
        }
        let refname = tid2refname.get(&tid).unwrap();
        let refseq = refname2refseq.get(refname).unwrap();

        let train_instance = TrainInstance::from_aligned_record_and_ref_seq_and_pin_start_end(
            &align_record,
            refseq,
            Some(dw_boundaries),
        );
        if let Some(train_instance) = &train_instance {
            arrow_hmm.update(&build_train_events(train_instance));
        }
    }
    pbar.finish();

    Ok(())
}

#[allow(unused)]
pub fn train_model_entrance(params: &TrainingParams) -> Option<()> {
    let aligned_bams = &params.aligned_bams;
    let ref_fastas = &params.ref_fas;
    let dw_boundaries = &params.dw_boundaries;

    let dw_buckets = dw_boundaries
        .split(",")
        .map(|v| v.trim())
        .map(|v| v.parse::<u8>().unwrap())
        .collect::<Vec<u8>>();
    assert!(aligned_bams.len() == ref_fastas.len());
    let mut arrow_hmm = HmmModel::new();

    for idx in 0..aligned_bams.len() {
        train_model(
            &aligned_bams[idx],
            &ref_fastas[idx],
            &mut arrow_hmm,
            &dw_buckets,
        )
        .unwrap();
    }
    arrow_hmm.finish();
    arrow_hmm.print_params();
    arrow_hmm.dump_to_file("arrow_hg002.params");

    Some(())
}

pub fn train_model_entrance_parallel(params: &TrainingParams) -> HmmModel {
    let aligned_bams = &params.aligned_bams;
    let ref_fastas = &params.ref_fas;
    let dw_boundaries = &params.parse_dw_boundaries();
    assert!(aligned_bams.len() == ref_fastas.len());

    let bam2refs = read_refs(aligned_bams, ref_fastas);


    let pbar = get_spin_pb(format!("training..."), DEFAULT_INTERVAL);

    let pbar = Arc::new(Mutex::new(pbar));

    let final_hmm_model = thread::scope(|s| {
        let bam2refs = &bam2refs;
        let (record_sender, record_receiver) = channel::bounded(1000);
        
        aligned_bams
            .iter()
            .for_each(|aligned_bam| {
                let record_sender_ = record_sender.clone();
                let pbar_ = pbar.clone();
                s.spawn(move || {
                    align_record_read_worker(aligned_bam, bam2refs.get(aligned_bam).unwrap(), pbar_, record_sender_);
                });
            });
        drop(record_sender);
        drop(pbar);

        let (train_ins_sender, train_ins_receiver) = channel::bounded(1000);
        for _ in 0..(num_cpus::get() / 2) {
            let record_receiver_ = record_receiver.clone();
            let train_ins_sender_ = train_ins_sender.clone();
            s.spawn(move || {
                train_instance_worker(Some(dw_boundaries), record_receiver_, train_ins_sender_)
            });
        }
        drop(record_receiver);
        drop(train_ins_sender);
        let mut hmm_model_parts = vec![];
        for _ in 0..(num_cpus::get() / 2) {
            let train_ins_recv_ = train_ins_receiver.clone();
            hmm_model_parts.push(s.spawn(move || update_hmm_model_worker(train_ins_recv_)));
        }

        let hmm_model_parts = hmm_model_parts
            .into_iter()
            .map(|hmm_model| hmm_model.join().unwrap())
            .collect::<Vec<_>>();

        let mut final_hmm_model = HmmModel::new();
        hmm_model_parts
            .into_iter()
            .for_each(|part| final_hmm_model.cnt_merge(&part));
        final_hmm_model.finish();
        final_hmm_model.print_params();
        final_hmm_model.dump_to_file("arrow_hg002.params");
        final_hmm_model
    });

    final_hmm_model
}

fn update_hmm_model_worker(receiver: channel::Receiver<TrainInstance>) -> HmmModel {
    let mut hmm = HmmModel::new();
    for train_instance in receiver {
        hmm.update(&build_train_events(&train_instance));
    }

    hmm
}

#[derive(Clone, Copy, PartialEq)]
pub struct Emit {
    pub ctx: u8,
    pub state: TransState,
    pub emit_base_enc: u8,
}

impl Emit {
    pub fn new(
        ref_base1: u8,
        ref_base2: u8,
        state: TransState,
        read_base: u8,
        dw_feat: u8,
    ) -> Self {
        assert!(dw_feat < 3, "dw_feat:{}", dw_feat);
        let ctx = encode_2_bases(ref_base1, ref_base2);
        let emit_base_enc = (dw_feat << 2) + gskits::dna::SEQ_NT4_TABLE[read_base as usize];
        assert!(
            emit_base_enc < 12,
            "dw_feat:{}, read_base:{}, emit_base_enc:{}",
            dw_feat,
            read_base as char,
            emit_base_enc
        );
        Self {
            ctx,
            state,
            emit_base_enc,
        }
    }
}

impl Debug for Emit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Emit")
            .field("ctx", &format!("{}", decode_2_bases(self.ctx)))
            .field("state", &(self.state as usize).to_string())
            .field("emit", &decode_emit_base(self.emit_base_enc))
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct CtxState {
    pub ctx: u8,
    pub state: TransState,
}

impl CtxState {
    pub fn new(ref_base1: u8, ref_base2: u8, state: TransState) -> Self {
        let ctx = encode_2_bases(ref_base1, ref_base2);

        Self { ctx, state }
    }
}

impl Debug for CtxState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CtxState")
            .field("ctx", &format!("{}", decode_2_bases(self.ctx)))
            .field("state", &(self.state as usize).to_string())
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrainEvent {
    EmitEvent(Emit),
    CtxStateEvent(CtxState),
}

/// ref: A' A C G T - - G C C A
/// rea:    A G G T G G G - C A
/// A'A AC CG GT TG TG TG GC CC CA
///   A  G  G  T  G  G  G  -  C  A

pub fn build_train_events(train_instance: &TrainInstance) -> Vec<TrainEvent> {
    let ref_seq_bytes = train_instance.ref_aligned_seq().as_bytes();
    let read_seq_bytes = train_instance.read_aligned_seq().as_bytes();
    let dw_features = train_instance.dw();

    let refpos2refpos = train_instance.ref_cur_pos2next_ref();

    let mut train_events = vec![];

    let mut pre_ref_base = 'A' as u8;
    for idx in 0..ref_seq_bytes.len() {
        let cur_ref_base = ref_seq_bytes[idx];
        let cur_read_base = read_seq_bytes[idx];
        if idx == 0 {
            assert!(cur_ref_base != '-' as u8);
            train_events.push(TrainEvent::EmitEvent(Emit::new(
                pre_ref_base,
                cur_ref_base,
                TransState::Match,
                cur_read_base,
                dw_features[idx].unwrap(),
            )));
        } else {
            match (cur_ref_base as char, cur_read_base as char) {
                ('-', read_base) => {
                    // insertion
                    let read_base = read_base as u8;
                    let ref_baseidx = refpos2refpos.get(&idx).copied().unwrap();
                    let ref_base = ref_seq_bytes[ref_baseidx];
                    assert_ne!(ref_base, '-' as u8, "qname:{}", train_instance.name);

                    let state = if ref_base == read_base {
                        TransState::Branch
                    } else {
                        TransState::Stick
                    };
                    train_events.push(TrainEvent::EmitEvent(Emit::new(
                        pre_ref_base,
                        ref_base,
                        state,
                        read_base,
                        dw_features[idx].unwrap(),
                    )));
                    train_events.push(TrainEvent::CtxStateEvent(CtxState::new(
                        pre_ref_base,
                        ref_base,
                        state,
                    )));
                }

                (ref_base, '-') => {
                    // deletion
                    let ref_base = ref_base as u8;
                    train_events.push(TrainEvent::CtxStateEvent(CtxState::new(
                        pre_ref_base,
                        ref_base,
                        TransState::Dark,
                    )));
                }
                (ref_base, read_base) => {
                    // match
                    let ref_base = ref_base as u8;
                    let read_base = read_base as u8;
                    train_events.push(TrainEvent::EmitEvent(Emit::new(
                        pre_ref_base,
                        ref_base,
                        TransState::Match,
                        read_base,
                        dw_features[idx].expect(&format!(
                            "no dw feature in idx:{}, ref_base:{}, read_base:{}",
                            idx, ref_base as char, read_base as char
                        )),
                    )));

                    if (idx + 1) != ref_seq_bytes.len() {
                        train_events.push(TrainEvent::CtxStateEvent(CtxState::new(
                            pre_ref_base,
                            cur_ref_base,
                            TransState::Match,
                        )));
                    }
                }
            }
        }
        if cur_ref_base != '-' as u8 {
            pre_ref_base = cur_ref_base;
        }
    }

    train_events
}

pub fn build_train_events_for_stat(train_instance: &TrainInstance) -> HashMap<String, Array1<usize>> {
    let ref_seq_bytes = train_instance.ref_aligned_seq().as_bytes();
    let read_seq_bytes = train_instance.read_aligned_seq().as_bytes();
    let dw_features = train_instance.dw();

    let refpos2refpos = train_instance.ref_cur_pos2next_ref();

    // ref-prebases, pref-curbase, state, emitbase. dw-cnt
    let mut counter = HashMap::new();

    let mut pre_ref_base = 'A' as u8;
    for idx in 0..ref_seq_bytes.len() {
        let cur_ref_base = ref_seq_bytes[idx];
        let cur_read_base = read_seq_bytes[idx];
        if idx == 0 {
            assert!(cur_ref_base != '-' as u8);

            let key = build_key(pre_ref_base, cur_ref_base, TransState::Match, cur_read_base);
            counter.entry(key).or_insert(Array1::<usize>::from_elem((256,), 0))[dw_features[idx].unwrap() as usize] +=
                1;
        } else {
            match (cur_ref_base as char, cur_read_base as char) {
                ('-', read_base) => {
                    // insertion
                    let read_base = read_base as u8;
                    let ref_baseidx = refpos2refpos.get(&idx).copied().unwrap();
                    let ref_base = ref_seq_bytes[ref_baseidx];
                    assert_ne!(ref_base, '-' as u8, "qname:{}", train_instance.name);

                    let state = if ref_base == read_base {
                        TransState::Branch
                    } else {
                        TransState::Stick
                    };

                    let key = build_key(pre_ref_base, ref_base, state, read_base);
                    counter.entry(key).or_insert(Array1::<usize>::from_elem((256,), 0))
                        [dw_features[idx].unwrap() as usize] += 1;
                }

                (ref_base, '-') => {
                    // deletion
                    let ref_base = ref_base as u8;
                }
                (ref_base, read_base) => {
                    // match
                    let ref_base = ref_base as u8;
                    let read_base = read_base as u8;

                    let key = build_key(pre_ref_base, ref_base, TransState::Match, read_base);
                    counter.entry(key).or_insert(Array1::<usize>::from_elem((256,), 0))
                        [dw_features[idx].unwrap() as usize] += 1;
                }
            }
        }
        if cur_ref_base != '-' as u8 {
            pre_ref_base = cur_ref_base;
        }
    }

    counter
}

fn build_key(ref_pre: u8, ref_cur: u8, state: TransState, emit: u8) -> String {
    format!(
        "{}{}:{}:{}",
        ref_pre as char, ref_cur as char, state, emit as char
    )
}

#[cfg(test)]
mod test {
    use crate::{
        common::TrainInstance,
        supervised_training::{build_train_events, TrainEvent},
    };

    #[test]
    fn test_build_train_events() {
        let ref_aligned_seq = "AA--CCCG-TC".to_string();
        let read_aligned_seq = "AAAACC-GGTC".to_string();
        let dw_buckets = vec![
            Some(1),
            Some(2),
            Some(0),
            Some(1),
            Some(1),
            Some(1),
            None,
            Some(1),
            Some(1),
            Some(1),
            Some(1),
        ];
        let train_ins = TrainInstance::new(
            ref_aligned_seq,
            read_aligned_seq,
            dw_buckets,
            "he".to_string(),
        );
        let events = build_train_events(&train_ins);

        println!("");
        events.iter().for_each(|event| match *event {
            TrainEvent::EmitEvent(emit) => print!("{:?}", emit),
            _ => (),
        });
        println!("");
        println!("");

        events.iter().for_each(|event| match *event {
            TrainEvent::CtxStateEvent(trans) => print!("{:?}", trans),
            _ => (),
        });
        println!("");
    }
}
