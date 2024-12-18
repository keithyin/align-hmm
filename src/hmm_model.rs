use core::str;
use ndarray::{s, Array2, Array3, Axis};
use std::io::{BufWriter, Write};

use crate::common::{TransState, IDX_BASE_MAP};

use super::supervised_training::TrainEvent;

const NUM_STATE: usize = 4;
const NUM_EMIT_STATE: usize = 3;
const NUM_CTX: usize = 16;
const NUM_EMIT: usize = 4 * 4 * 4;

pub struct HmmModel {
    ctx_trans_cnt: Array2<usize>,
    emission_cnt: Array3<usize>,

    ctx_trans_prob: Array2<f32>,
    emission_prob: Array3<f32>,
}

impl HmmModel {
    pub fn new() -> Self {
        HmmModel {
            ctx_trans_cnt: Array2::<usize>::zeros((NUM_CTX, NUM_STATE)), // zeros: no laplace smooth, ones: laplace smooth
            emission_cnt: Array3::<usize>::ones((NUM_EMIT_STATE, NUM_CTX, NUM_EMIT)),
            ctx_trans_prob: Array2::<f32>::zeros((NUM_CTX, NUM_STATE)),
            emission_prob: Array3::<f32>::zeros((NUM_EMIT_STATE, NUM_CTX, NUM_EMIT)),
        }
    }

    pub fn update(&mut self, events: &Vec<TrainEvent>) {
        events.iter().for_each(|&event| match event {
            TrainEvent::EmitEvent(emit) => {
                assert!((emit.state as usize) < NUM_EMIT_STATE);
                assert!((emit.ctx as usize) < NUM_CTX, "ctx:{}", emit.ctx);
                assert!(
                    (emit.emit_base_enc as usize) < NUM_EMIT,
                    "emit_base_enc:{}",
                    emit.emit_base_enc
                );

                self.emission_cnt[[
                    emit.state as usize,
                    emit.ctx as usize,
                    emit.emit_base_enc as usize,
                ]] += 1;
            }
            TrainEvent::CtxStateEvent(ctx_state) => {
                assert!((ctx_state.state as usize) < NUM_STATE);
                assert!((ctx_state.ctx as usize) < NUM_CTX, "ctx:{}", ctx_state.ctx);

                self.ctx_trans_cnt[[ctx_state.ctx as usize, ctx_state.state as usize]] += 1;
            }
        });
    }

    pub fn set_emission_prob(&mut self, emission_prob: Array3<f32>) {
        assert_eq!(self.emission_prob.shape(), emission_prob.shape());

        self.emission_prob = emission_prob;
    }

    pub fn set_ctx_trans_prob(&mut self, ctx_trans_prob: Array2<f32>) {
        assert_eq!(self.ctx_trans_prob.shape(), ctx_trans_prob.shape());
        self.ctx_trans_prob = ctx_trans_prob;
    }

    pub fn finish(&mut self) {
        let ctx_aggr = self.ctx_trans_cnt.sum_axis(Axis(1));
        let emission_aggr = self.emission_cnt.sum_axis(Axis(2));

        for i in 0..NUM_CTX {
            if ctx_aggr[[i]] == 0 {
                continue;
            }

            for j in 0..NUM_STATE {
                self.ctx_trans_prob[[i, j]] =
                    (self.ctx_trans_cnt[[i, j]] as f32) / (ctx_aggr[[i]] as f32);
            }
        }

        for i in 0..NUM_EMIT_STATE {
            for j in 0..NUM_CTX {
                if emission_aggr[[i, j]] == 0 {
                    continue;
                }
                for k in 0..NUM_EMIT {
                    self.emission_prob[[i, j, k]] =
                        (self.emission_cnt[[i, j, k]] as f32) / (emission_aggr[[i, j]] as f32);
                }
            }
        }
    }

    pub fn emit_prob(&self, movement: TransState, ctx: u8, emit: u8) -> f32 {
        assert!((movement as usize) < 3);
        self.emission_prob[[movement as usize, ctx as usize, emit as usize]]
    }

    pub fn ctx_move(&self, ctx: u8, movement: TransState) -> f32 {
        self.ctx_trans_prob[[ctx as usize, movement as usize]]
    }

    pub fn delta(&self, other: &HmmModel) -> f32 {
        let emit_prob_delta = (self.emission_prob.clone() - other.emission_prob.clone())
            .abs()
            .mean()
            .unwrap();
        let trans_delta = (self.ctx_trans_prob.clone() - other.ctx_trans_prob.clone())
            .abs()
            .mean()
            .unwrap();
        emit_prob_delta + trans_delta
    }

    pub fn cnt_merge(&mut self, other: &HmmModel) {
        for state_idx in 0..NUM_EMIT_STATE {
            for ctx_idx in 0..NUM_CTX {
                for emit in 0..NUM_EMIT {
                    self.emission_cnt[[state_idx, ctx_idx, emit]] +=
                        other.emission_cnt[[state_idx, ctx_idx, emit]];
                }
            }
        }

        for ctx_idx in 0..NUM_CTX {
            for state_idx in 0..NUM_STATE {
                self.ctx_trans_cnt[[ctx_idx, state_idx]] +=
                    other.ctx_trans_cnt[[ctx_idx, state_idx]];
            }
        }
    }

    fn ctx_trans_prob_to_string(&self) -> String {
        let mut param_strs = vec![];

        for i in 0..NUM_CTX {
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

        for state_idx in 0..NUM_EMIT_STATE {
            let mut state_string = vec![];
            for ctx_idx in 0..NUM_CTX {
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

impl From<&HmmBuilder> for HmmModel {
    fn from(value: &HmmBuilder) -> Self {
        let ctx_prob = value.ctx_move_prob_numerator.sum_axis(Axis(1));
        let ctx_move_prob = Array2::from_shape_fn((NUM_CTX, NUM_STATE), |(ctx, state)| {
            value.ctx_move_prob_numerator[[ctx, state]] / ctx_prob[ctx]
        });

        let move_ctx_prob = value.move_ctx_emit_prob_numerator.sum_axis(Axis(2));
        let move_ctx_emit_prob = Array3::from_shape_fn((NUM_EMIT_STATE, NUM_CTX, NUM_EMIT), |(state, ctx, emit)| {
            value.move_ctx_emit_prob_numerator[[state, ctx, emit]] / move_ctx_prob[[state, ctx]]
        });

        let mut default_model = Self::new();
        default_model.set_emission_prob(move_ctx_emit_prob);
        default_model.set_ctx_trans_prob(ctx_move_prob);
        default_model
    }
}

#[derive(Debug)]
pub struct HmmBuilder {
    ctx_move_prob_numerator: Array2<f32>,      // 16 * 4
    move_ctx_emit_prob_numerator: Array3<f32>, // 3 * 16 * 64 ?
}

impl HmmBuilder {
    pub fn new() -> Self {
        Self {
            ctx_move_prob_numerator: Array2::zeros((NUM_CTX, NUM_STATE)),
            move_ctx_emit_prob_numerator: Array3::zeros((NUM_EMIT_STATE, NUM_CTX, NUM_EMIT)),
        }
    }

    pub fn add_to_ctx_move_prob_numerator(&mut self, ctx: u8, movement: TransState, prob: f32) {
        self.ctx_move_prob_numerator[[ctx as usize, movement as usize]] += prob;
    }

    pub fn add_to_move_ctx_emit_prob_numerator(
        &mut self,
        ctx: u8,
        movement: TransState,
        emit: u8,
        prob: f32,
    ) {
        assert!((movement as usize) < 3);
        self.move_ctx_emit_prob_numerator[[movement as usize, ctx as usize, emit as usize]] += prob;
    }

    pub fn merge(&mut self, other: &HmmBuilder) {
        for ctx in 0..NUM_CTX {
            for state in 0..NUM_STATE {
                self.ctx_move_prob_numerator[[ctx, state]] +=
                    other.ctx_move_prob_numerator[[ctx, state]];
            }
        }

        // 3 state for emit
        for state in 0..NUM_EMIT_STATE {
            for ctx in 0..NUM_CTX {
                for emit in 0..NUM_EMIT {
                    self.move_ctx_emit_prob_numerator[[state, ctx, emit]] +=
                        other.move_ctx_emit_prob_numerator[[state, ctx, emit]];
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::HmmBuilder;


    #[test]
    fn test_builder_merge() {

        let mut b1 = HmmBuilder::new();
        let b2 = HmmBuilder::new();
        b1.merge(&b2);

    }
}
