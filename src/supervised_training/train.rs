use core::str;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    thread,
};

use crossbeam::channel;
use gskits::{
    fastx_reader::{fasta_reader::FastaFileReader, read_fastx},
    pbar::{get_spin_pb, DEFAULT_INTERVAL},
};
use rust_htslib::bam::{self, Read};

use crate::{
    cli::TrainingParams,
    hmm_model::HmmModel,
    train_instance::{issue_align_record, train_instance_worker},
};

use crate::common::{build_train_events, TrainInstance};

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
            dw_boundaries,
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

pub fn train_model_entrance_parallel(params: &TrainingParams) -> HmmModel{
    let aligned_bams = &params.aligned_bams;
    let ref_fastas = &params.ref_fas;
    let dw_boundaries = &params.parse_dw_boundaries();
    assert!(aligned_bams.len() == ref_fastas.len());

    let pbar = get_spin_pb(format!("training..."), DEFAULT_INTERVAL);

    let pbar = Arc::new(Mutex::new(pbar));

    let final_hmm_model = thread::scope(|s| {
        let (record_sender, record_receiver) = channel::bounded(1000);

        aligned_bams
            .iter()
            .zip(ref_fastas.iter())
            .for_each(|(aligned_bam, ref_fasta)| {
                let record_sender_ = record_sender.clone();
                let pbar_ = pbar.clone();
                s.spawn(move || {
                    issue_align_record(aligned_bam, ref_fasta, pbar_, record_sender_);
                });
            });
        drop(record_sender);
        drop(pbar);

        let (train_ins_sender, train_ins_receiver) = channel::bounded(1000);
        for _ in 0..(num_cpus::get() / 2) {
            let record_receiver_ = record_receiver.clone();
            let train_ins_sender_ = train_ins_sender.clone();
            s.spawn(move || {
                train_instance_worker(dw_boundaries, record_receiver_, train_ins_sender_)
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
