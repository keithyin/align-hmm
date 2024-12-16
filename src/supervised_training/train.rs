use core::str;
use std::{collections::HashMap, io::{BufWriter, Write}};

use clap::ArgMatches;
use rust_htslib::{bam::{self, ext::BamRecordExtensions, Read}, faidx};

use crate::{bam_ext::BamRecordExt, cli::SupervisedTrainingParams, pb_tools::{self, get_spin_pb, DEFAULT_INTERVAL}, samtools_facade::samtools_fai};
use ndarray::{Array2, Array3, Axis, s};

lazy_static! {
    static ref BASE_MAP: HashMap<u8, usize> = {
        let mut map = HashMap::new();
        map.insert('A' as u8, 0);
        map.insert('C' as u8, 1);
        map.insert('G' as u8, 2);
        map.insert('T' as u8, 3);
        map
    };

    static ref IDX_BASE_MAP: HashMap<u8, u8> = {
        let mut map = HashMap::new();
        map.insert(0, 'A' as u8);
        map.insert(1, 'C' as u8);
        map.insert(2, 'G' as u8);
        map.insert(3, 'T' as u8);
        map
    };
}

#[derive(Clone, Copy, PartialEq)]
enum State {
    Match = 0,
    Branch = 1, // homo insertion
    Stick = 2, // non-homo insertion
    Dark = 3, // deletion
}

fn encode_ctx(pre_base: u8, cur_base: u8) -> usize{
    ((*BASE_MAP.get(&pre_base).unwrap()) << 2) + *(BASE_MAP.get(&cur_base).unwrap())
}


fn state_identification(ref_positions: &Vec<usize>, 
    ref_pos_cursor: i32, 
    read_pos: &Option<i64>, ref_pos: &Option<i64>, 
    read_seq: &[u8], ref_seq: &[u8]) -> State {
    
    assert!(read_pos.is_some() || ref_pos.is_some());

    return if let Some(read_pos_) = read_pos {
        if ref_pos.is_some() {
            State::Match
        } else {
            let ref_pos = ref_positions[ref_pos_cursor as usize];
            let ref_next_pos = ref_positions[(ref_pos_cursor + 1) as usize];
            if read_seq[*read_pos_ as usize] == ref_seq[ref_pos] || read_seq[*read_pos_ as usize] == ref_seq[ref_next_pos + 1] {
                State::Branch
            } else {
                State::Stick

            }

        }
    } else {
        // dark
        State::Dark
    };

}

fn pin_start_end(aligned_record: &bam::Record) -> Vec<[Option<i64>; 2]>{

    let read_ref_pairs = aligned_record.aligned_pairs_full().collect::<Vec<[Option<i64>; 2]>>();

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


struct ArrowHmm {
    num_state: usize,
    num_emit_state: usize,
    num_ctx: usize,
    num_emit: usize,

    ctx_trans_cnt: Array2<usize>,
    emission_cnt: Array3<usize>,

    ctx_trans_prob: Array2<f32>,
    emission_prob: Array3<f32>,

    dw_buckets: Vec<u8>
}

impl ArrowHmm {
    fn new(dw_buckets: Vec<u8>) -> Self{
        let num_state = 4_usize;
        let num_emit_state = 3_usize;
        let num_ctx = 16_usize;
        let num_emit = 4 * 4 * 4;
        ArrowHmm { 
            num_state,
            num_emit_state,
            num_ctx,
            num_emit,
            ctx_trans_cnt: Array2::<usize>::zeros((num_ctx, num_state)),  // zeros: no laplace smooth, ones: laplace smooth
            emission_cnt: Array3::<usize>::ones((num_emit_state, num_ctx, num_emit)) ,
            ctx_trans_prob: Array2::<f32>::zeros((num_ctx, num_state)), 
            emission_prob: Array3::<f32>::zeros((num_emit_state, num_ctx, num_emit)),
            dw_buckets
        }

    }

    fn update(&mut self, aligned_record: &bam::Record, ref_str: &str) {

        let ref_seq = ref_str.as_bytes();
        let is_rev = aligned_record.is_reverse();
        let align_pairs = pin_start_end(aligned_record);
        let ref_positions = align_pairs.iter()
            .filter(|[_, ref_pos]| ref_pos.is_some())
            .map(|[_, ref_pos]| ref_pos.unwrap() as usize)
            .collect::<Vec<usize>>();

        let mut ref_pos_cursor = -1;

        let read_seq = String::from_utf8(aligned_record.seq().as_bytes()).unwrap();
        let read_seq = read_seq.as_bytes();

        for [read_pos, ref_pos] in align_pairs {

            if ref_pos.is_some() {
                ref_pos_cursor += 1;
            }

            let cur_state = state_identification(&ref_positions, ref_pos_cursor, &read_pos, &ref_pos, read_seq, ref_seq);

            match cur_state {
                State::Match => {
                    if ref_pos_cursor == 0 {
                        continue;
                    }
                    let pre_base = if ref_pos_cursor == 0 {
                        'A' as u8
                    } else {
                        ref_seq[ref_positions[(ref_pos_cursor-1) as usize]]
                    };
                    let cur_base = ref_seq[ref_positions[ref_pos_cursor as usize]];
                    let ctx = encode_ctx(pre_base, cur_base);
                    let emit = self.encode_emit_feat(subread_record.get_dw(read_pos.unwrap() as usize, is_rev), read_seq[read_pos.unwrap() as usize]);
                    
                    self.ctx_trans_cnt[[ctx, cur_state as usize]] += 1;
                    self.emission_cnt[[cur_state as usize, ctx, emit]] += 1;


                },
                State::Dark => {
                    if ref_pos_cursor == 0 {
                        continue;
                    }
                    
                    let pre_base = if ref_pos_cursor == 0 {
                        'A' as u8
                    } else {
                        ref_seq[ref_positions[(ref_pos_cursor-1) as usize]]
                    };
                    let cur_base = ref_seq[ref_positions[ref_pos_cursor as usize]];
                    let ctx = encode_ctx(pre_base, cur_base);
                    
                    self.ctx_trans_cnt[[ctx, cur_state as usize]] += 1;

                },
                State::Stick | State::Branch => {
                    let pre_base = ref_seq[ref_positions[ref_pos_cursor as usize]];
                    let cur_base = ref_seq[ref_positions[(ref_pos_cursor + 1) as usize]];
                    let ctx = encode_ctx(pre_base, cur_base);
                    let emit = self.encode_emit_feat(subread_record.get_dw(read_pos.unwrap() as usize, is_rev), read_seq[read_pos.unwrap() as usize]);
                    
                    self.ctx_trans_cnt[[ctx, cur_state as usize]] += 1;
                    self.emission_cnt[[cur_state as usize, ctx, emit]] += 1;
                }
            };

        }


    }

    fn finish(&mut self) {
        let ctx_aggr = self.ctx_trans_cnt.sum_axis(Axis(1));
        let emission_aggr = self.emission_cnt.sum_axis(Axis(2));

        for i in 0..self.num_ctx {
            if ctx_aggr[[i]] == 0 {
                continue;
            }

            for j in 0..self.num_state {
                self.ctx_trans_prob[[i, j]] = (self.ctx_trans_cnt[[i, j]] as f32) / (ctx_aggr[[i]] as f32);
            }
        }

        for i in 0..self.num_emit_state {
            for j in 0..self.num_ctx {
                if emission_aggr[[i, j]] == 0 {
                    continue;
                }
                for k in 0..self.num_emit {
                    self.emission_prob[[i, j, k]] = (self.emission_cnt[[i, j, k]] as f32) / (emission_aggr[[i, j]] as f32);
                }
            }
        }

    }

    fn encode_dw_feat(&self, dw: u8) -> u8 {
        match self.dw_buckets.binary_search(&dw) {
            Ok(n) => n as u8,
            Err(n) => n as u8
        }
    }

    fn encode_emit_feat(&self, mut dw: u8, base: u8) -> usize {
        dw = self.encode_dw_feat(dw);
        ((dw as usize) << 2) + (*BASE_MAP.get(&base).unwrap())
    }

    fn ctx_trans_prob_to_string(&self) -> String {

        let mut param_strs = vec![];

        for i in 0..self.num_ctx {
            let second = IDX_BASE_MAP.get(& ((i & 0b0011) as u8)).unwrap();
            let first = IDX_BASE_MAP.get(& (((i >> 2) & 0b0011) as u8)).unwrap();
            let ctx = String::from_iter(vec![(*first) as char, (*second) as char]);
            let prob_str = format!("{{{}}}, // {}",
                self.ctx_trans_prob.slice(s![i, ..]).iter()
                    .map(|v| v.to_string()).collect::<Vec<String>>()
                    .join(","),
                    ctx
                );
            param_strs.push(prob_str);
        }

        param_strs.join("\n")
    }

    fn emit_prob_to_string(&self) -> String {
        let mut param_strs = vec![];

        for state_idx in 0..self.num_emit_state {
            let mut state_string = vec![];
            for ctx_idx in 0..self.num_ctx {
                let emit = self.emission_prob.slice(s![state_idx, ctx_idx, ..]);
                let emit = emit.iter().map(|v| v.to_string()).collect::<Vec<String>>().join(",");
                
                state_string.push(format!("{{{}}}", emit));

            }
            param_strs.push(format!("{{\n {} \n}}", state_string.join(",\n")));
        }

        param_strs.join(",\n")
    }

    pub fn print_params(&self) {
        println!("{}", self.ctx_trans_prob_to_string());
        println!("{}", self.emit_prob_to_string());
    }

    pub fn dump_to_file(&self, filename: &str) {
        let file = std::fs::File::create(filename).unwrap();
        let mut writer = BufWriter::new(file);
        writeln!(&mut writer, "{}", self.ctx_trans_prob_to_string()).unwrap();
        writeln!(&mut writer, "\n\n\n\n").unwrap();
        writeln!(&mut writer, "{}", self.emit_prob_to_string()).unwrap();
    }


    
}


struct SubreadRecord {
    qname: Option<String>,
    dw: Vec<u8>,
    cr: Vec<u8>,

    #[allow(unused)]
    seq: String
}

impl SubreadRecord {
    
    pub fn new(record: &bam::Record) -> Self {
        let facade = BamRecordExt::new(record);
        let qname = Some(facade.get_qname());
        let dw = facade.get_dw().unwrap().into_iter().map(|v| v as u8).collect::<Vec<u8>>();
        let cr = facade.get_cr().unwrap().into_iter().map(|v| v as u8).collect::<Vec<u8>>();

        assert_eq!(dw.len(), cr.len());
        let seq = facade.get_seq();
        
        Self { 
            qname, 
            dw, 
            cr,
            seq

        }
    }

    pub fn get_pos(&self, mut pos: usize, is_rev: bool) -> usize{
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

    pub fn get_dw(&self, mut pos: usize, is_rev: bool) -> u8 {
        pos = self.get_pos(pos, is_rev);

        self.dw[pos]
    }
}



fn train_model(aligned_bam: &str, ref_fasta: &str, arrow_hmm: &mut ArrowHmm) -> anyhow::Result<()>{
    samtools_fai(ref_fasta, false)?;
    let ref_fasta = faidx::Reader::from_path(ref_fasta).unwrap();

    let mut refname2refseq = HashMap::new();
    
    let mut aligned_bam_reader = bam::Reader::from_path(aligned_bam).unwrap();
    aligned_bam_reader.set_threads(10).unwrap();
    let aligned_bam_header  = bam::Header::from_template(aligned_bam_reader.header());
    let aligned_bam_header_view  = bam::HeaderView::from_header(&aligned_bam_header);

    let mut tid2refname = HashMap::new();

    let pbar = get_spin_pb(format!("training... using {}", aligned_bam), DEFAULT_INTERVAL);

    for align_record in aligned_bam_reader.records() {
        pbar.inc(1);
        let align_record = align_record.unwrap();
        let tid = align_record.tid();
        if !tid2refname.contains_key(&tid) {
            tid2refname.insert(tid, String::from_utf8(aligned_bam_header_view.tid2name(tid as u32).to_vec()).unwrap());
        }
        let refname = tid2refname.get(&tid).unwrap();
        let qname = String::from_utf8(align_record.qname().to_vec()).unwrap();
        
        if !refname2refseq.contains_key(refname) {
            let seq = ref_fasta.fetch_seq_string(refname, 0, 3_000_000_000_000).unwrap();
            refname2refseq.insert(refname.to_string(), seq);
        }

        let refseq = refname2refseq.get(refname).unwrap();

        arrow_hmm.update(&align_record, refseq);
        

    }
    pbar.finish();

    Ok(())

}

fn read_subreads_bam(subreads_bam: &str) -> HashMap<String, SubreadRecord> {
    let mut res = HashMap::new();
    let mut subreads_bam_reader = bam::Reader::from_path(subreads_bam).unwrap();
    subreads_bam_reader.set_threads(10).unwrap();

    let pbar = pb_tools::get_spin_pb(format!("reading {subreads_bam}"), DEFAULT_INTERVAL);
    for record in subreads_bam_reader.records() {
        pbar.inc(1);
        let record = record.unwrap();
        
        let mut subread_record = SubreadRecord::new(&record);
        let qname = subread_record.qname.take().unwrap();
        res.insert(qname, subread_record);
    }
    pbar.finish();
    res
}


pub fn train_model_entrance(params: &SupervisedTrainingParams) -> Option<()> {
    let aligned_bams = &params.aligned_bams;
    let ref_fastas = &params.ref_fas;
    let dw_boundaries = &params.dw_boundaries;

    let dw_buckets = dw_boundaries.split(",")
        .map(|v| v.trim())
        .map(|v| v.parse::<u8>().unwrap())
        .collect::<Vec<u8>>();
    assert!(aligned_bams.len() == ref_fastas.len());
    let mut arrow_hmm = ArrowHmm::new(dw_buckets);

    for idx in 0..aligned_bams.len() {
        train_model(&aligned_bams[idx], &ref_fastas[idx], &mut arrow_hmm).unwrap();
    }
    arrow_hmm.finish();
    arrow_hmm.print_params();
    arrow_hmm.dump_to_file("arrow_hg002.params");

    Some(())
}