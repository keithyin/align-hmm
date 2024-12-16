use std::collections::HashMap;

use crate::{bam_ext::BamRecordExt, utils};
use bio::alignment::AlignmentOperation;
use gskits::{dna::reverse_complement, ds::ReadInfo};
use mm2::{
    align_single_query_to_targets, build_aligner,
    params::{AlignParams, OupParams},
};
use rust_htslib::bam::{self, ext::BamRecordExtensions, Record};

#[derive(Clone, Copy, PartialEq)]
pub enum ArrowState {
    Match = 0,
    Branch = 1, // homo insertion
    Stick = 2,  // non-homo insertion
    Dark = 3,   // deletion
}

pub fn encode_ctx(pre_base: u8, cur_base: u8) -> usize {
    ((*utils::dna::BASE_MAP.get(&pre_base).unwrap()) << 2)
        + *(utils::dna::BASE_MAP.get(&cur_base).unwrap())
}

pub fn pin_start_end(aligned_record: &bam::Record) -> Vec<[Option<i64>; 2]> {
    let read_ref_pairs = aligned_record
        .aligned_pairs_full()
        .collect::<Vec<[Option<i64>; 2]>>();

    let ref_start = aligned_record.reference_start();
    let ref_end = aligned_record.reference_end();
    let mut start_pos = 0;
    let mut end_pos = 0;

    for pos in 0..read_ref_pairs.len() {
        let [read_pos, ref_pos] = &read_ref_pairs[pos];
        if ref_pos.is_some() && ref_pos.unwrap() >= ref_start && read_pos.is_some() {
            start_pos = pos;
            break;
        }
    }

    for pos in (0..read_ref_pairs.len()).rev() {
        let [read_pos, ref_pos] = &read_ref_pairs[pos];
        if ref_pos.is_some() && ref_pos.unwrap() < ref_end && read_pos.is_some() {
            end_pos = pos + 1;
            break;
        }
    }
    read_ref_pairs[start_pos..end_pos].to_vec()
}

pub struct SubreadRecord {
    pub qname: Option<String>,
    dw: Vec<u8>,
    cr: Vec<u8>,
    seq: String,
}

impl SubreadRecord {
    pub fn new(record: &bam::Record) -> Self {
        let facade = BamRecordExt::new(record);
        let qname = Some(facade.get_qname());
        let dw = facade
            .get_dw()
            .unwrap()
            .into_iter()
            .map(|v| v as u8)
            .collect::<Vec<u8>>();
        let cr = facade
            .get_cr()
            .unwrap()
            .into_iter()
            .map(|v| v as u8)
            .collect::<Vec<u8>>();

        assert_eq!(dw.len(), cr.len());
        let seq = facade.get_seq();

        Self { qname, dw, cr, seq }
    }
    #[allow(unused)]
    pub fn get_pos(&self, mut pos: usize, is_rev: bool) -> usize {
        if is_rev {
            pos = self.cr.len() - pos - 1;
        }
        pos
    }

    #[allow(unused)]
    pub fn get_cr(&self, mut pos: usize, is_rev: bool) -> u8 {
        pos = self.get_pos(pos, is_rev);

        self.cr[pos]
    }

    #[allow(unused)]
    pub fn get_dw(&self, mut pos: usize, is_rev: bool) -> u8 {
        pos = self.get_pos(pos, is_rev);
        self.dw[pos]
    }

    pub fn get_dw_range(&self, start: usize, end: usize) -> &[u8] {
        &self.dw[start..end]
    }

    pub fn get_seq_range(&self, start: usize, end: usize) -> &[u8] {
        &self.seq.as_bytes()[start..end]
    }
}

pub fn left_align_indel_v1(
    query_seq: &[u8],
    ref_seq: &[u8],
    align_ops: &mut Vec<AlignmentOperation>,
) {
    for _ in 0..align_ops.len() {
        let mut ref_cursor: i32 = -1;
        let mut query_cursor: i32 = -1;

        let mut moved_this_round = false;
        for align_idx in 0..(align_ops.len() - 1) {
            let op = align_ops[align_idx];
            match op {
                AlignmentOperation::Match | AlignmentOperation::Subst => {
                    ref_cursor += 1;
                    query_cursor += 1;
                }

                AlignmentOperation::Del => ref_cursor += 1,
                AlignmentOperation::Ins => query_cursor += 1,
                _ => panic!(""),
            };

            if op == AlignmentOperation::Match || op == AlignmentOperation::Subst {
                continue;
            }
            let next_op = align_ops[align_idx + 1];
            if next_op == AlignmentOperation::Ins || next_op == AlignmentOperation::Del {
                continue;
            }

            if op == AlignmentOperation::Ins {
                if (query_cursor as usize + 2) >= query_seq.len() {
                    break;
                }
                if query_seq[query_cursor as usize] == query_seq[query_cursor as usize + 1]
                    && query_seq[query_cursor as usize] == query_seq[query_cursor as usize + 2]
                {
                    align_ops[align_idx as usize] = align_ops[align_idx as usize + 1];
                    align_ops[align_idx as usize + 1] = op;
                    moved_this_round = true;
                    ref_cursor += 1;
                }
                continue;
            }

            if op == AlignmentOperation::Del {
                if (ref_cursor as usize + 1) >= ref_seq.len() {
                    break;
                }
                if ref_seq[ref_cursor as usize] == ref_seq[ref_cursor as usize + 1] {
                    align_ops[align_idx as usize] = align_ops[align_idx as usize + 1];
                    align_ops[align_idx as usize + 1] = op;
                    moved_this_round = true;
                    query_cursor += 1;
                }
                continue;
            }
        }

        if !moved_this_round {
            break;
        }
    }
}

pub fn left_align_indel_v2(
    query_seq: &[u8],
    ref_seq: &[u8],
    align_ops: &mut Vec<AlignmentOperation>,
) {
    for _ in 0..align_ops.len() {
        let mut ref_cursor: i32 = -1;
        let mut query_cursor: i32 = -1;

        let mut moved_this_round = false;
        for align_idx in 0..(align_ops.len() - 2) {
            let op = align_ops[align_idx];
            match op {
                AlignmentOperation::Match | AlignmentOperation::Subst => {
                    ref_cursor += 1;
                    query_cursor += 1;
                }

                AlignmentOperation::Del => ref_cursor += 1,
                AlignmentOperation::Ins => query_cursor += 1,
                _ => panic!(""),
            };

            if op == AlignmentOperation::Match || op == AlignmentOperation::Subst {
                continue;
            }
            let next_op = align_ops[align_idx + 1];
            if next_op == AlignmentOperation::Ins || next_op == AlignmentOperation::Del {
                continue;
            }

            if op == AlignmentOperation::Ins {
                if (query_cursor as usize + 2) >= query_seq.len() {
                    break;
                }
                if query_seq[query_cursor as usize] == query_seq[query_cursor as usize + 1] {
                    align_ops[align_idx as usize] = align_ops[align_idx as usize + 1];
                    align_ops[align_idx as usize + 1] = op;
                    moved_this_round = true;
                    ref_cursor += 1;
                }
                continue;
            }

            if op == AlignmentOperation::Del {
                if (ref_cursor as usize + 1) >= ref_seq.len() {
                    break;
                }
                if ref_seq[ref_cursor as usize] == ref_seq[ref_cursor as usize + 1] {
                    align_ops[align_idx as usize] = align_ops[align_idx as usize + 1];
                    align_ops[align_idx as usize + 1] = op;
                    moved_this_round = true;
                    query_cursor += 1;
                }
                continue;
            }
        }

        if !moved_this_round {
            break;
        }
    }
}

pub struct TrainInstance {
    ref_aligned_seq: String,
    read_aligned_seq: String,
    dw: Vec<Option<u8>>,
}

impl TrainInstance {
    pub fn from_aligned_record_and_ref_seq_and_pin_start_end(
        align_record: &bam::Record,
        ref_seq: &str,
        dw_boundaries: &Vec<u8>,
    ) -> Self {
        return if !align_record.is_reverse() {
            TrainInstance::from_fwd_aligned_record_and_ref_seq(
                align_record,
                ref_seq,
                dw_boundaries,
                None,
            )
            .pin_start_end()
        } else {
            let (records, new_ref_seq, dw) = change_direction(align_record, ref_seq);
            TrainInstance::from_fwd_aligned_record_and_ref_seq(
                &records[0],
                &new_ref_seq,
                dw_boundaries,
                Some(dw),
            )
            .pin_start_end()
        };
    }

    pub fn ref_aligned_seq(&self) -> &str {
        &self.ref_aligned_seq
    }

    pub fn read_aligned_seq(&self) -> &str {
        &self.ref_aligned_seq
    }

    pub fn dw(&self) -> &Vec<Option<u8>> {
        &self.dw
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

        if !align_record.is_reverse() {
            let record_ext = gskits::gsbam::bam_record_ext::BamRecordExt::new(align_record);
            let ref_start = record_ext.reference_start();
            let ref_end = record_ext.reference_end();
            let query_start = record_ext.query_alignment_start();
            let query_end = record_ext.query_alignment_end();
            let dw = dw.unwrap_or(record_ext.get_dw().expect(&format!("no dw in bam file")));

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
        }

        Self {
            ref_aligned_seq: ref_aligned_seq,
            read_aligned_seq: read_aligned_seq,
            dw: dw_features,
        }
    }

    fn pin_start_end(self) -> Self {
        let align_span_len = self.dw.len();

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
            ref_aligned_seq: self.ref_aligned_seq[start_idx..end_idx].to_string(),
            read_aligned_seq: self.read_aligned_seq[start_idx..end_idx].to_string(),
            dw: self.dw[start_idx..end_idx].to_vec(),
        }
    }
}

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
    assert!(records.len() > 0);

    (records, ref_sub_seq, dw)
}

pub struct Emit {
    ctx: u8,
    state: ArrowState,
    emit_base_enc: u8,
}

impl Emit {
    pub fn new(
        ref_base1: u8,
        ref_base2: u8,
        state: ArrowState,
        read_base: u8,
        dw_feat: u8,
    ) -> Self {
        let ctx = ref_base1 << 2 + ref_base2;
        let emit_base_enc = dw_feat << 2 + read_base;
        Self {
            ctx,
            state,
            emit_base_enc,
        }
    }
}

pub struct CtxState {
    ctx: u8,
    state: ArrowState,
}

impl CtxState {
    pub fn new(ref_base1: u8, ref_base2: u8, state: ArrowState) -> Self {
        let ctx = ref_base1 << 2 + ref_base2;
        Self { ctx, state }
    }
}

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
            train_events.push(TrainEvent::EmitEvent(Emit::new(
                pre_ref_base,
                cur_ref_base,
                ArrowState::Match,
                cur_read_base,
                dw_features[idx].unwrap(),
            )));
        } else {
            match (cur_ref_base as char, cur_read_base as char) {
                ('-', read_base) => {
                    // insertion
                    let read_base = read_base as u8;
                    let ref_baseidx = refpos2refpos.get(&idx).copied().unwrap();
                    let ref_base = read_seq_bytes[ref_baseidx];
                    assert_ne!(ref_base, '-' as u8);

                    let state = if ref_base == read_base {
                        ArrowState::Branch
                    } else {
                        ArrowState::Stick
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
                        cur_ref_base,
                        state,
                    )));
                }

                (ref_base, '-') => {
                    // deletion
                    let ref_base = ref_base as u8;
                    train_events.push(TrainEvent::CtxStateEvent(CtxState::new(
                        pre_ref_base,
                        ref_base,
                        ArrowState::Dark,
                    )));
                }
                (ref_base, read_base) => {
                    // match
                    let ref_base = ref_base as u8;
                    let read_base = read_base as u8;
                    train_events.push(TrainEvent::EmitEvent(Emit::new(
                        pre_ref_base,
                        ref_base,
                        ArrowState::Match,
                        read_base,
                        dw_features[idx].unwrap(),
                    )));

                    if (idx + 1) != ref_seq_bytes.len() {
                        train_events.push(TrainEvent::CtxStateEvent(CtxState::new(
                            pre_ref_base,
                            cur_ref_base,
                            ArrowState::Match,
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