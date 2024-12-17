use core::str;
use std::{
    collections::HashMap,
    io::{BufWriter, Write},
};

use gskits::{
    fastx_reader::{fasta_reader::FastaFileReader, read_fastx},
    pbar::{get_spin_pb, DEFAULT_INTERVAL},
};
use rust_htslib::bam::{self, Read};

use ndarray::{s, Array2, Array3, Axis};

use crate::cli::TrainingParams;

use super::common::{build_train_events, TrainEvent, TrainInstance, IDX_BASE_MAP};

struct ArrowHmm {
    num_state: usize,
    num_emit_state: usize,
    num_ctx: usize,
    num_emit: usize,

    ctx_trans_cnt: Array2<usize>,
    emission_cnt: Array3<usize>,

    ctx_trans_prob: Array2<f32>,
    emission_prob: Array3<f32>,
}

impl ArrowHmm {
    fn new() -> Self {
        let num_state = 4_usize;
        let num_emit_state = 3_usize;
        let num_ctx = 16_usize;
        let num_emit = 4 * 4 * 4;
        ArrowHmm {
            num_state,
            num_emit_state,
            num_ctx,
            num_emit,
            ctx_trans_cnt: Array2::<usize>::zeros((num_ctx, num_state)), // zeros: no laplace smooth, ones: laplace smooth
            emission_cnt: Array3::<usize>::ones((num_emit_state, num_ctx, num_emit)),
            ctx_trans_prob: Array2::<f32>::zeros((num_ctx, num_state)),
            emission_prob: Array3::<f32>::zeros((num_emit_state, num_ctx, num_emit)),
        }
    }

    fn update(&mut self, events: &Vec<TrainEvent>) {
        events.iter().for_each(|&event| match event {
            TrainEvent::EmitEvent(emit) => {
                self.emission_cnt[[
                    emit.state as usize,
                    emit.ctx as usize,
                    emit.emit_base_enc as usize,
                ]] += 1;
            }
            TrainEvent::CtxStateEvent(ctx_state) => {
                self.ctx_trans_cnt[[ctx_state.ctx as usize, ctx_state.state as usize]] += 1;
            }
        });
    }

    fn finish(&mut self) {
        let ctx_aggr = self.ctx_trans_cnt.sum_axis(Axis(1));
        let emission_aggr = self.emission_cnt.sum_axis(Axis(2));

        for i in 0..self.num_ctx {
            if ctx_aggr[[i]] == 0 {
                continue;
            }

            for j in 0..self.num_state {
                self.ctx_trans_prob[[i, j]] =
                    (self.ctx_trans_cnt[[i, j]] as f32) / (ctx_aggr[[i]] as f32);
            }
        }

        for i in 0..self.num_emit_state {
            for j in 0..self.num_ctx {
                if emission_aggr[[i, j]] == 0 {
                    continue;
                }
                for k in 0..self.num_emit {
                    self.emission_prob[[i, j, k]] =
                        (self.emission_cnt[[i, j, k]] as f32) / (emission_aggr[[i, j]] as f32);
                }
            }
        }
    }

    fn ctx_trans_prob_to_string(&self) -> String {
        let mut param_strs = vec![];

        for i in 0..self.num_ctx {
            let second = IDX_BASE_MAP.get(&((i & 0b0011) as u8)).unwrap();
            let first = IDX_BASE_MAP.get(&(((i >> 2) & 0b0011) as u8)).unwrap();
            let ctx = String::from_iter(vec![(*first) as char, (*second) as char]);
            let prob_str = format!(
                "{{{}}}, // {}",
                self.ctx_trans_prob
                    .slice(s![i, ..])
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
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
                let emit = emit
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
                    .join(",");

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

fn train_model(
    aligned_bam: &str,
    ref_fasta: &str,
    arrow_hmm: &mut ArrowHmm,
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
        arrow_hmm.update(&build_train_events(&train_instance));
    }
    pbar.finish();

    Ok(())
}

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
    let mut arrow_hmm = ArrowHmm::new();

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
