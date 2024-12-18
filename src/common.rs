use std::collections::HashMap;

use bio::alignment::pairwise::Aligner;
use gskits::{
    dna::reverse_complement,
    ds::ReadInfo,
    gsbam::{
        bam_record_ext::{BamRecord, BamRecordExt},
        cigar_ext::parse_cigar_string,
    },
};
use lazy_static::lazy_static;
use mm2::{
    align_single_query_to_targets, build_aligner,
    params::{AlignParams, OupParams},
};
use rust_htslib::bam::{self, ext::BamRecordExtensions, Record};

use crate::em_training::model::encode_2_bases;

lazy_static! {
    pub static ref BASE_MAP: HashMap<u8, usize> = {
        let mut map = HashMap::new();
        map.insert('A' as u8, 0);
        map.insert('C' as u8, 1);
        map.insert('G' as u8, 2);
        map.insert('T' as u8, 3);
        map
    };
    pub static ref IDX_BASE_MAP: HashMap<u8, u8> = {
        let mut map = HashMap::new();
        map.insert(0, 'A' as u8);
        map.insert(1, 'C' as u8);
        map.insert(2, 'G' as u8);
        map.insert(3, 'T' as u8);
        map
    };
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransState {
    Match = 0,
    Branch = 1, // homo insertion
    Stick = 2,  // non-homo insertion
    Dark = 3,   // deletion
}

pub struct TrainInstance {
    name: String,
    ref_aligned_seq: String,
    read_aligned_seq: String,
    dw_buckets: Vec<Option<u8>>, // base 的 dw 在哪个桶里
}

impl TrainInstance {
    pub fn new(
        ref_aligned_seq: String,
        read_aligned_seq: String,
        dw_buckets: Vec<Option<u8>>,
        name: String,
    ) -> Self {
        Self {
            name,
            ref_aligned_seq,
            read_aligned_seq,
            dw_buckets,
        }
    }
    pub fn from_aligned_record_and_ref_seq_and_pin_start_end(
        align_record: &bam::Record,
        ref_seq: &str,
        dw_boundaries: &Vec<u8>,
    ) -> Option<Self> {
        return if !align_record.is_reverse() {
            Some(
                TrainInstance::from_fwd_aligned_record_and_ref_seq(
                    align_record,
                    ref_seq,
                    dw_boundaries,
                    None,
                )
                .pin_start_end(),
            )
        } else {
            let (records, new_ref_seq, dw) = change_direction_v2(align_record, ref_seq);
            let align_record_ext = BamRecordExt::new(align_record);

            if records.len() == 0 {
                tracing::warn!("change direction failed. {}", align_record_ext.get_qname());
                None
            } else {
                if records[0].is_reverse() {
                    tracing::warn!(
                        "change direction failed. {}. still reversed",
                        align_record_ext.get_qname()
                    );
                    None
                } else {
                    Some(
                        TrainInstance::from_fwd_aligned_record_and_ref_seq(
                            &records[0],
                            &new_ref_seq,
                            dw_boundaries,
                            Some(dw),
                        )
                        .pin_start_end(),
                    )
                }
            }
        };
    }

    pub fn ref_aligned_seq(&self) -> &str {
        &self.ref_aligned_seq
    }

    pub fn read_aligned_seq(&self) -> &str {
        &self.read_aligned_seq
    }

    pub fn dw(&self) -> &Vec<Option<u8>> {
        &self.dw_buckets
    }

    pub fn ref_cur_pos2next_ref(&self) -> HashMap<usize, usize> {
        let mut pre_base_pos = 0;
        let ref_seq_bytes = self.ref_aligned_seq.as_bytes();
        let mut curpos2nextpos = HashMap::new();
        for pos in 1..ref_seq_bytes.len() {
            if ref_seq_bytes[pos] != '-' as u8 {
                for idx in pre_base_pos..pos {
                    curpos2nextpos.insert(idx, pos);
                }

                pre_base_pos = pos;
            }
        }

        curpos2nextpos
    }

    fn from_fwd_aligned_record_and_ref_seq(
        align_record: &bam::Record,
        ref_seq: &str,
        dw_boundaries: &Vec<u8>,
        dw: Option<Vec<u32>>,
    ) -> Self {
        assert!(!align_record.is_reverse());

        let mut ref_aligned_seq = String::new();
        let mut read_aligned_seq = String::new();
        let mut dw_features = vec![];

        let record_ext = gskits::gsbam::bam_record_ext::BamRecordExt::new(align_record);
        let ref_start = record_ext.reference_start();
        let ref_end = record_ext.reference_end();
        let query_start = record_ext.query_alignment_start();
        let query_end = record_ext.query_alignment_end();
        // let dw = dw.unwrap_or(record_ext.get_dw().expect(&format!("no dw in bam file")));
        let dw = if dw.is_some() {
            dw.unwrap()
        } else {
            record_ext.get_dw().expect(&format!("no dw in bam file"))
        };
        // let dw = dw.unwrap_or_else(|| record_ext.get_dw().expect(&format!("no dw in bam file")));

        let mut qpos_cursor = None;
        let mut rpos_cursor = None;

        let query_seq = record_ext.get_seq();

        for [qpos, rpos] in align_record.aligned_pairs_full() {
            if qpos.is_some() {
                qpos_cursor = qpos;
            }
            if rpos.is_some() {
                rpos_cursor = rpos;
            }

            if rpos_cursor.is_none() || qpos_cursor.is_none() {
                continue;
            }

            if (rpos_cursor.unwrap() as usize) < ref_start
                || (qpos_cursor.unwrap() as usize) < query_start
            {
                continue;
            }

            if (rpos_cursor.unwrap() as usize >= ref_end)
                || (qpos_cursor.unwrap() as usize) >= query_end
            {
                break;
            }

            if let Some(rpos) = rpos {
                let rpos = rpos as usize;
                ref_aligned_seq.push_str(&ref_seq[rpos..rpos + 1]);
            } else {
                ref_aligned_seq.push_str("-");
            }

            if let Some(qpos) = qpos {
                let qpos = qpos as usize;
                read_aligned_seq.push_str(&query_seq[qpos..qpos + 1]);
                let cur_dw = dw[qpos];
                let bucket = match dw_boundaries.binary_search(&(cur_dw as u8)) {
                    Ok(n) | Err(n) => n as u8,
                };
                dw_features.push(Some(bucket));
            } else {
                read_aligned_seq.push_str("-");
                dw_features.push(None);
            }
        }

        assert_eq!(ref_aligned_seq.len(), read_aligned_seq.len());
        assert_eq!(ref_aligned_seq.len(), dw_features.len());

        // println!(
        //     "{}\n{}\n{}",
        //     ref_aligned_seq,
        //     read_aligned_seq,
        //     dw_features
        //         .iter()
        //         .map(|v| v.map(|v| v.to_string()).unwrap_or("-".to_string()))
        //         .collect::<Vec<String>>()
        //         .join("")
        // );

        Self {
            name: record_ext.get_qname(),
            ref_aligned_seq: ref_aligned_seq,
            read_aligned_seq: read_aligned_seq,
            dw_buckets: dw_features,
        }
    }

    fn pin_start_end(self) -> Self {
        let align_span_len = self.dw_buckets.len();

        let read_seq_bytes = self.read_aligned_seq.as_bytes();
        let ref_seq_bytes = self.ref_aligned_seq.as_bytes();
        let mut start_idx = None;
        let mut end_idx = None;
        for idx in 0..align_span_len {
            if read_seq_bytes[idx] != '-' as u8 && ref_seq_bytes[idx] != '-' as u8 {
                if start_idx.is_none() {
                    start_idx = Some(idx);
                }
                end_idx = Some(idx);
            }
        }

        let start_idx = start_idx.unwrap();
        let end_idx = end_idx.unwrap() + 1;

        Self {
            name: self.name,
            ref_aligned_seq: self.ref_aligned_seq[start_idx..end_idx].to_string(),
            read_aligned_seq: self.read_aligned_seq[start_idx..end_idx].to_string(),
            dw_buckets: self.dw_buckets[start_idx..end_idx].to_vec(),
        }
    }
}

#[allow(unused)]
pub fn change_direction(
    align_record: &bam::Record,
    ref_seq: &str,
) -> (Vec<Record>, String, Vec<u32>) {
    assert!(align_record.is_reverse());
    let record_ext = gskits::gsbam::bam_record_ext::BamRecordExt::new(align_record);
    let ref_start = record_ext.reference_start();
    let ref_end = record_ext.reference_end();
    let dw = record_ext.get_dw().expect(&format!("no dw in bam file"));

    let ref_start = if ref_start > 10 { ref_start - 10 } else { 0 };
    let ref_end = if ref_end < (ref_seq.len() - 10) {
        ref_end + 10
    } else {
        ref_seq.len()
    };

    let ref_sub_seq = gskits::dna::reverse_complement(&ref_seq[ref_start..ref_end]);
    let query_seq = reverse_complement(&record_ext.get_seq());
    let dw = dw.into_iter().rev().collect::<Vec<_>>();
    let ref_sub_seq_len = ref_sub_seq.len();
    let target_read = vec![ReadInfo::new_fa_record(
        "ref".to_string(),
        ref_sub_seq.clone(),
    )];
    let aligner = build_aligner(
        "map-ont",
        &mm2::params::IndexParams::default(),
        &mm2::params::MapParams::default(),
        &AlignParams::default(),
        &OupParams::default(),
        &target_read,
    );

    let mut target_idx = HashMap::new();
    target_idx.insert("ref".to_string(), (0, ref_sub_seq_len));

    let query_record = ReadInfo::new_fa_record(record_ext.get_qname(), query_seq);
    let records = align_single_query_to_targets(&query_record, &aligner, &target_idx);
    (records, ref_sub_seq, dw)
}

pub fn change_direction_v2(
    align_record: &bam::Record,
    ref_seq: &str,
) -> (Vec<Record>, String, Vec<u32>) {
    assert!(align_record.is_reverse());
    let record_ext = gskits::gsbam::bam_record_ext::BamRecordExt::new(align_record);
    let ref_start = record_ext.reference_start();
    let ref_end = record_ext.reference_end();
    let dw = record_ext.get_dw().expect(&format!("no dw in bam file"));

    let ref_start = if ref_start > 10 { ref_start - 10 } else { 0 };
    let ref_end = if ref_end < (ref_seq.len() - 10) {
        ref_end + 10
    } else {
        ref_seq.len()
    };

    let ref_sub_seq = gskits::dna::reverse_complement(&ref_seq[ref_start..ref_end]);
    let query_seq = reverse_complement(&record_ext.get_seq());
    let dw = dw.into_iter().rev().collect::<Vec<_>>();
    let qname = record_ext.get_qname();

    let record = local_alignment_to_record(&ref_sub_seq, &query_seq, Some(qname.as_str()));
    let records = vec![record];
    (records, ref_sub_seq, dw)
}

pub fn local_alignment_to_record(target: &str, query: &str, qname: Option<&str>) -> BamRecord {
    let mut aligner = Aligner::with_capacity(target.len(), query.len(), -4, -2, |a: u8, b: u8| {
        if a == b {
            2
        } else {
            -4
        }
    });
    let alignment = aligner.local(target.as_bytes(), query.as_bytes());
    let rstart = alignment.xstart;
    let qstart = alignment.ystart;
    let qend = alignment.yend;

    let mut pre_op = *alignment.operations.first().unwrap();
    let mut pre_op_cnt = 1;
    let mut cigar_str = String::new();
    alignment.operations.iter().skip(1).for_each(|&op| {
        if op == pre_op {
            pre_op_cnt += 1;
        } else {
            let cur = match pre_op {
                bio::alignment::AlignmentOperation::Match => format!("{}=", pre_op_cnt),
                bio::alignment::AlignmentOperation::Subst => format!("{}X", pre_op_cnt),
                bio::alignment::AlignmentOperation::Ins => format!("{}D", pre_op_cnt),
                bio::alignment::AlignmentOperation::Del => format!("{}I", pre_op_cnt),
                what => panic!("not a valid op:{:?}", what),
            };

            cigar_str.push_str(&cur);

            pre_op = op;
            pre_op_cnt = 1;
        }
    });

    let cur = match pre_op {
        bio::alignment::AlignmentOperation::Match => format!("{}=", pre_op_cnt),
        bio::alignment::AlignmentOperation::Subst => format!("{}X", pre_op_cnt),
        bio::alignment::AlignmentOperation::Ins => format!("{}D", pre_op_cnt),
        bio::alignment::AlignmentOperation::Del => format!("{}I", pre_op_cnt),
        what => panic!("not a valid op:{:?}", what),
    };

    cigar_str.push_str(&cur);

    let head_cigar = if qstart > 0 {
        format!("{}S", qstart)
    } else {
        "".to_string()
    };
    let tail_cigar = if query.len() > qend {
        format!("{}S", query.len() - qend)
    } else {
        "".to_string()
    };
    let cigar_str = format!("{}{}{}", head_cigar, cigar_str, tail_cigar);

    let mut record = BamRecord::new();
    let qname = qname.unwrap_or("default");
    record.set(
        qname.as_bytes(),
        Some(&parse_cigar_string(&cigar_str).unwrap()),
        query.as_bytes(),
        &vec![255; query.len()],
    );
    record.set_pos(rstart as i64);
    record.unset_reverse();
    record.unset_secondary();
    record.unset_supplementary();
    record.unset_duplicate();
    record.unset_unmapped();

    record
}

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone, Copy)]
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

    // println!(
    //         "{}\n{}",
    //         train_instance.ref_aligned_seq(),
    //         dw_features
    //             .iter()
    //             .map(|v| v.map(|v| v.to_string()).unwrap_or("-".to_string()))
    //             .collect::<Vec<String>>()
    //             .join("")
    //     );

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

#[cfg(test)]
mod test {
    use gskits::{
        fastx_reader::{fasta_reader::FastaFileReader, read_fastx},
        gsbam::bam_record_ext::BamRecordExt,
    };
    use rust_htslib::bam::{self, ext::BamRecordExtensions, Read};
    use std::collections::HashMap;

    use super::{build_train_events, local_alignment_to_record, TrainInstance};

    #[test]
    fn test_train_instance() {
        let ref_fasta =
            "/data/ccs_data/HG002/GCA_000001405.15_GRCh38_no_alt_analysis_set.chr1-chr22.fasta";
        let fasta_rader = FastaFileReader::new(ref_fasta.to_string());
        let refname2refseq = read_fastx(fasta_rader)
            .into_iter()
            .map(|read_info| (read_info.name, read_info.seq))
            .collect::<HashMap<_, _>>();

        let aligned_bam = "/data/ccs_data/HG002/20240402_Sync_Y0003_02_H01_Run0002_called_subreads.align4arrow.arrow_hmm.bam";
        let mut aligned_bam_reader = bam::Reader::from_path(aligned_bam).unwrap();
        aligned_bam_reader.set_threads(10).unwrap();
        let aligned_bam_header = bam::Header::from_template(aligned_bam_reader.header());
        let aligned_bam_header_view = bam::HeaderView::from_header(&aligned_bam_header);

        let mut tid2refname = HashMap::new();

        let dw_boundaries = &vec![18, 46];

        for align_record in aligned_bam_reader.records() {
            let align_record = align_record.unwrap();
            let align_record_ext = BamRecordExt::new(&align_record);
            if align_record_ext
                .get_qname()
                .eq("read_438028/438028/subread/2")
            {
                let tid = align_record.tid();
                if !tid2refname.contains_key(&tid) {
                    tid2refname.insert(
                        tid,
                        String::from_utf8(aligned_bam_header_view.tid2name(tid as u32).to_vec())
                            .unwrap(),
                    );
                }

                let refname = tid2refname.get(&tid).unwrap();
                let refseq = refname2refseq.get(refname).unwrap();

                let train_instance =
                    TrainInstance::from_aligned_record_and_ref_seq_and_pin_start_end(
                        &align_record,
                        refseq,
                        dw_boundaries,
                    )
                    .unwrap();
                println!("{}", train_instance.ref_aligned_seq());
                // println!("{:?}", train_instance.ref_cur_pos2next_ref());

                let ref_aligned_seq = train_instance.ref_aligned_seq().as_bytes();
                train_instance
                    .ref_cur_pos2next_ref()
                    .iter()
                    .for_each(|(&cur_pos, &next_pos)| {
                        if ref_aligned_seq[next_pos] == '-' as u8 {
                            println!("cur_pos:{}, next_pos:{}", cur_pos, next_pos);
                        }
                    });

                break;
            }
        }
    }

    #[test]
    fn test_local_alignment_to_record() {
        let target = "CCAAAGGGGGTGGACC";
        let query = "AAGGGGGGTGG";

        let record = local_alignment_to_record(target, query, None);
        let mut ref_aligned = String::new();
        let mut query_aligned = String::new();
        let query_seq = BamRecordExt::new(&record).get_seq();
        for [qpos, rpos] in record.aligned_pairs_full() {
            if let Some(rpos) = rpos {
                ref_aligned.push(target.as_bytes()[rpos as usize] as char);
            } else {
                ref_aligned.push('-');
            }

            if let Some(qpos) = qpos {
                query_aligned.push(query_seq.as_bytes()[qpos as usize] as char);
            } else {
                query_aligned.push('-');
            }
        }
        println!("{:?}", record);
        println!("ref:{}\nqry:{}", ref_aligned, query_aligned);
    }

    #[test]
    fn test_build_train_events() {
        let ref_aligned_seq = "AAA--CCCG-TC".to_string();
        let read_aligned_seq = "AAAAACC-GGTC".to_string();
        let dw_buckets = vec![Some(0), Some(1), Some(2), Some(0), Some(1), Some(1), Some(1), None, Some(1), Some(1), Some(1), Some(1)];
        let train_ins = TrainInstance::new(ref_aligned_seq, read_aligned_seq, dw_buckets, "he".to_string());
        let events = build_train_events(&train_ins);
        
    }
}
