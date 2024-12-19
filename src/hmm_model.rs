use core::{f64, str};
use ndarray::{s, stack, Array, Array1, Array2, Array3, Axis, Dimension};
use std::io::{BufWriter, Write};

use crate::common::{TransState, IDX_BASE_MAP};

use super::supervised_training::TrainEvent;

pub const NUM_STATE: usize = 4;
pub const NUM_EMIT_STATE: usize = 3;
pub const NUM_CTX: usize = 16;
pub const NUM_EMIT: usize = 4 * 4 * 4;

#[derive(Debug)]
pub struct HmmModel {
    ctx_trans_cnt: Array2<usize>,
    emission_cnt: Array3<usize>,

    ctx_trans_prob: Array2<f64>,
    emission_prob: Array3<f64>,
}

impl HmmModel {
    pub fn new() -> Self {
        HmmModel {
            ctx_trans_cnt: Array2::<usize>::zeros((NUM_CTX, NUM_STATE)), // zeros: no laplace smooth, ones: laplace smooth
            emission_cnt: Array3::<usize>::ones((NUM_EMIT_STATE, NUM_CTX, NUM_EMIT)),
            ctx_trans_prob: Array2::<f64>::zeros((NUM_CTX, NUM_STATE)),
            emission_prob: Array3::<f64>::zeros((NUM_EMIT_STATE, NUM_CTX, NUM_EMIT)),
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

    pub fn set_emission_prob(&mut self, emission_prob: Array3<f64>) {
        assert_eq!(self.emission_prob.shape(), emission_prob.shape());

        self.emission_prob = emission_prob;
    }

    pub fn set_ctx_trans_prob(&mut self, ctx_trans_prob: Array2<f64>) {
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
                    (self.ctx_trans_cnt[[i, j]] as f64) / (ctx_aggr[[i]] as f64);
            }
        }

        for i in 0..NUM_EMIT_STATE {
            for j in 0..NUM_CTX {
                if emission_aggr[[i, j]] == 0 {
                    continue;
                }
                for k in 0..NUM_EMIT {
                    self.emission_prob[[i, j, k]] =
                        (self.emission_cnt[[i, j, k]] as f64) / (emission_aggr[[i, j]] as f64);
                }
            }
        }
    }

    pub fn emit_prob(&self, movement: TransState, ctx: u8, emit: u8) -> f64 {
        assert!((movement as usize) < 3);
        let prob = self.emission_prob[[movement as usize, ctx as usize, emit as usize]];
        if prob < 1e-10 {
            1e-10
        } else {
            prob
        }
    }

    pub fn ctx_state(&self, ctx: u8, movement: TransState) -> f64 {
        let prob = self.ctx_trans_prob[[ctx as usize, movement as usize]];
        if prob < 1e-10 {
            1e-10
        } else {
            prob
        }
    }

    pub fn delta(&self, other: &HmmModel) -> f64 {
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
        let move_ctx_emit_prob =
            Array3::from_shape_fn((NUM_EMIT_STATE, NUM_CTX, NUM_EMIT), |(state, ctx, emit)| {
                value.move_ctx_emit_prob_numerator[[state, ctx, emit]] / move_ctx_prob[[state, ctx]]
            });

        let mut default_model = Self::new();
        default_model.set_emission_prob(move_ctx_emit_prob);
        default_model.set_ctx_trans_prob(ctx_move_prob);
        default_model
    }
}

impl From<&HmmBuilderV2> for HmmModel {
    fn from(value: &HmmBuilderV2) -> Self {
        let ctx_prob_max = max_axis_2d(&value.ctx_move_prob_numerator, 1);
        assert_eq!(ctx_prob_max.shape(), &[NUM_CTX]);
        // println!("{:?}", value.ctx_move_prob_numerator);
        // println!("{:?}", ctx_prob_max);
        let ctx_move_prob = Array2::from_shape_fn((NUM_CTX, NUM_STATE), |(ctx, state)| {
            (value.ctx_move_prob_numerator[[ctx, state]] - ctx_prob_max[ctx]).exp() + 1e-10
        });
        // println!("{:?}", ctx_move_prob);
        let ctx_prob = ctx_move_prob.sum_axis(Axis(1));
        let ctx_move_prob = Array2::from_shape_fn((NUM_CTX, NUM_STATE), |(ctx, state)| {
            ctx_move_prob[[ctx, state]] / ctx_prob[ctx]
        });
        // println!("{:?}", ctx_prob);
        // println!("{:?}", ctx_move_prob);

        let move_ctx_emit_max = value
            .move_ctx_emit_prob_numerator
            .axis_iter(Axis(0))
            .map(|arr2d| max_axis_2d(&arr2d.to_owned(), 1))
            .collect::<Vec<_>>();

        assert_eq!(move_ctx_emit_max.len(), NUM_EMIT_STATE);
        assert_eq!(move_ctx_emit_max[0].shape(), &[NUM_CTX]);

        let move_ctx_emit_max = Array2::from_shape_fn((NUM_EMIT_STATE, NUM_CTX), |(state, ctx)| {
            move_ctx_emit_max[state][ctx]
        });

        // println!("{:?}", value.move_ctx_emit_prob_numerator);
        // println!("{:?}", move_ctx_emit_max);

        let move_ctx_emit_prob =
            Array3::from_shape_fn((NUM_EMIT_STATE, NUM_CTX, NUM_EMIT), |(state, ctx, emit)| {
                (value.move_ctx_emit_prob_numerator[[state, ctx, emit]]
                    - move_ctx_emit_max[[state, ctx]]).exp() + 1e-10
            });
        let move_ctx_prob = move_ctx_emit_prob.sum_axis(Axis(2));
        let move_ctx_emit_prob =
            Array3::from_shape_fn((NUM_EMIT_STATE, NUM_CTX, NUM_EMIT), |(state, ctx, emit)| {
                move_ctx_emit_prob[[state, ctx, emit]] / move_ctx_prob[[state, ctx]]
            });

        let mut default_model = Self::new();
        default_model.set_emission_prob(move_ctx_emit_prob);
        default_model.set_ctx_trans_prob(ctx_move_prob);
        default_model
    }
}

#[derive(Debug)]
pub struct HmmBuilder {
    ctx_move_prob_numerator: Array2<f64>,      // 16 * 4
    move_ctx_emit_prob_numerator: Array3<f64>, // 3 * 16 * 64 ?
}

impl HmmBuilder {
    pub fn new() -> Self {
        Self {
            ctx_move_prob_numerator: Array2::zeros((NUM_CTX, NUM_STATE)),
            move_ctx_emit_prob_numerator: Array3::zeros((NUM_EMIT_STATE, NUM_CTX, NUM_EMIT)),
        }
    }

    pub fn add_to_ctx_move_prob_numerator(&mut self, ctx: u8, movement: TransState, prob: f64) {
        self.ctx_move_prob_numerator[[ctx as usize, movement as usize]] += prob;
    }

    pub fn add_to_move_ctx_emit_prob_numerator(
        &mut self,
        ctx: u8,
        movement: TransState,
        emit: u8,
        prob: f64,
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

#[derive(Debug)]
pub struct HmmBuilderV2 {
    ctx_move_prob_numerator: Array2<f64>,      // 16 * 4
    move_ctx_emit_prob_numerator: Array3<f64>, // 3 * 16 * 64 ?
    ctx_move_flag: Array2<bool>,
    move_ctx_emit_flag: Array3<bool>,
    log_likelihoods: Option<f64>
}

impl HmmBuilderV2 {
    pub fn new() -> Self {
        Self {
            ctx_move_prob_numerator: Array2::from_elem((NUM_CTX, NUM_STATE), f64::MIN),
            move_ctx_emit_prob_numerator: Array3::from_elem((NUM_EMIT_STATE, NUM_CTX, NUM_EMIT), f64::MIN),
            ctx_move_flag: Array2::from_shape_fn((NUM_CTX, NUM_STATE), |_| false),
            move_ctx_emit_flag: Array3::from_shape_fn((NUM_EMIT_STATE, NUM_CTX, NUM_EMIT), |_| {
                false
            }),
            log_likelihoods: None
        }
    }

    pub fn add_to_ctx_move_prob_numerator(&mut self, ctx: u8, movement: TransState, log_prob: f64) {
        if !self.ctx_move_flag[[ctx as usize, movement as usize]] {
            self.ctx_move_prob_numerator[[ctx as usize, movement as usize]] = log_prob;
            self.ctx_move_flag[[ctx as usize, movement as usize]] = true;
        } else {
            self.ctx_move_prob_numerator[[ctx as usize, movement as usize]] = stream_acc_log_prob(
                self.ctx_move_prob_numerator[[ctx as usize, movement as usize]],
                log_prob,
            );
        }
    }

    pub fn add_to_move_ctx_emit_prob_numerator(
        &mut self,
        ctx: u8,
        movement: TransState,
        emit: u8,
        log_prob: f64,
    ) {
        assert!((movement as usize) < 3);

        if !self.move_ctx_emit_flag[[movement as usize, ctx as usize, emit as usize]] {
            self.move_ctx_emit_prob_numerator[[movement as usize, ctx as usize, emit as usize]] =
                log_prob;
            self.move_ctx_emit_flag[[movement as usize, ctx as usize, emit as usize]] = true;
        } else {
            self.move_ctx_emit_prob_numerator[[movement as usize, ctx as usize, emit as usize]] =
                stream_acc_log_prob(
                    self.move_ctx_emit_prob_numerator
                        [[movement as usize, ctx as usize, emit as usize]],
                    log_prob,
                );
        }
    }

    pub fn add_log_likehood(&mut self, new_log_prob: f64) {

        if let Some(old_log_prob) = self.log_likelihoods {
            self.log_likelihoods = Some(stream_acc_log_prob(old_log_prob, new_log_prob));
        } else {
            self.log_likelihoods = Some(new_log_prob);

        }
    }

    pub fn merge(&mut self, other: &HmmBuilderV2) {

        if let Some(new_log_prob) = other.log_likelihoods {
            self.add_log_likehood(new_log_prob);
        }

        for ctx in 0..NUM_CTX {
            for state in 0..NUM_STATE {
                if other.ctx_move_flag[[ctx, state]] {
                    self.add_to_ctx_move_prob_numerator(
                        ctx as u8,
                        state.into(),
                        other.ctx_move_prob_numerator[[ctx, state]],
                    );
                }
            }
        }

        // 3 state for emit
        for state in 0..NUM_EMIT_STATE {
            for ctx in 0..NUM_CTX {
                for emit in 0..NUM_EMIT {
                    if other.move_ctx_emit_flag[[state, ctx, emit]] {
                        self.add_to_move_ctx_emit_prob_numerator(
                            ctx as u8,
                            state.into(),
                            emit as u8,
                            other.move_ctx_emit_prob_numerator[[state, ctx, emit]],
                        );
                    }
                }
            }
        }
    }

    pub fn get_log_likelihood(&self) -> Option<f64> {
        self.log_likelihoods
    }
}

/// log sum exp trick. for 2 log probs
pub fn stream_acc_log_prob(acc: f64, new_log_prob: f64) -> f64 {
    let max_log = if acc > new_log_prob {
        acc
    } else {
        new_log_prob
    };
    max_log + (1. + (-(acc - new_log_prob).abs()).exp()).ln()
}

/// log sum exp trick. for list of log probs
pub fn log_sum_exp(scores: &[f64]) -> f64 {
    let max_score = scores.iter().cloned().fold(f64::MIN, f64::max);
    if max_score == f64::MIN {
        f64::MIN
    } else {
        max_score + scores.iter().map(|&x| (x - max_score).exp()).sum::<f64>().ln()
    }
}

pub fn max_axis_2d(arr: &Array2<f64>, mut axis: usize) -> Array1<f64> {
    if axis == 0 {
        axis = 1;
    } else {
        axis = 0;
    };
    arr.axis_iter(Axis(axis))
        .map(|view| view.iter().cloned().fold(f64::MIN, f64::max))
        .collect()
}

#[cfg(test)]
mod test {
    use ndarray::Array2;

    use crate::hmm_model::max_axis_2d;

    use super::{stream_acc_log_prob, HmmBuilder, HmmBuilderV2, HmmModel};

    #[test]
    fn test_builder_merge() {
        let mut b1 = HmmBuilder::new();
        let b2 = HmmBuilder::new();
        b1.merge(&b2);
    }

    #[test]
    fn test_builder_v2() {
        let mut builder = HmmBuilderV2::new();
        builder.add_to_ctx_move_prob_numerator(0, crate::common::TransState::Match, -2.0);

        let hmm_model: HmmModel = (&builder).into();
        println!("{:?}", hmm_model);
    }

    #[test]
    fn test_max_axis_2d() {
        let values = vec![1.0_f64, 2., 3., 4., 5., 6.];
        let values: Array2<f64> = Array2::from_shape_vec((2, 3), values).unwrap();
        println!("{:?}", max_axis_2d(&values, 1));
    }

    #[test]
    fn test_stream_acc_log_prob() {
        let v = stream_acc_log_prob(-1.0, -2.0);
        println!("{}", v.exp());
        println!("{}", (-1.0_f64).exp() + (-2.0_f64).exp());
    }
}
