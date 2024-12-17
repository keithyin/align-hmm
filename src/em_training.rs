use std::thread;

use crate::{
    cli::TrainingParams,
    supervised_training::common::TrainInstance,
    train_instance::{encode_emit, issue_all_train_instance},
};
use crossbeam::channel;
use fb::{backward, forward};
use model::{encode_2_bases, HmmBuilder, HmmModel, Template, TemplatePos};
pub mod fb;
pub mod model;

pub fn em_training(params: &TrainingParams) {
    let dw_boundaries = params.parse_dw_boundaries();

    let mut hmm_model = HmmModel::new(12);

    for epoch in 0..1000 {
        let new_hmm_model: HmmModel = thread::scope(|s| {
            let aligned_bams = &params.aligned_bams;
            let ref_fastas = &params.ref_fas;
            let dw_boundaries = &dw_boundaries;
            let hmm_model_ref = &hmm_model;

            let (train_ins_sender, train_ins_receiver) = channel::bounded(1000);
            s.spawn(move || {
                issue_all_train_instance(aligned_bams, ref_fastas, dw_boundaries, train_ins_sender);
            });

            let mut handles = vec![];
            for _ in 0..(num_cpus::get() - 4) {
                let train_ins_receiver_ = train_ins_receiver.clone();
                handles.push(s.spawn(move || train(train_ins_receiver_, hmm_model_ref)));
            }
            drop(train_ins_receiver);

            let mut final_hmm_builder = HmmBuilder::new();
            handles.into_iter().for_each(|h| {
                final_hmm_builder.merge(&h.join().unwrap());
            });

            (&final_hmm_builder).into()
        });
        let delta = hmm_model.delta(&new_hmm_model);
        if delta < 1e-6 {
            println!("DONE!!!!");
            break;
        } else {
            println!("epoch:{}, delta:{}", epoch, delta);
        }
        hmm_model = new_hmm_model;
    }
}

pub fn train(
    train_ins_receiver: channel::Receiver<TrainInstance>,
    hmm_model: &HmmModel,
) -> HmmBuilder {
    let mut hmm_builder = HmmBuilder::new();

    for train_ins in train_ins_receiver {
        let rseq = train_ins.ref_aligned_seq().replace('-', "");
        let qseq = train_ins.read_aligned_seq().replace('-', "");
        let dwell_time = train_ins
            .dw()
            .iter()
            .filter(|v| v.is_some())
            .map(|v| v.unwrap())
            .collect::<Vec<u8>>();

        let tpl = Template::from_template_bases(rseq.as_bytes(), hmm_model);
        let encoded_emit = encode_emit(&dwell_time, &qseq);
        let alpha_dp = forward(&encoded_emit, &tpl, hmm_model);
        let beta_dp = backward(&encoded_emit, &tpl, hmm_model);

        let mut prev_trans_probs = TemplatePos::default();
        let mut prev_tpl_base = prev_trans_probs.base();

        for col in 1..(alpha_dp.shape()[1] - 1) {
            let cur_trans_prob = tpl[col - 1];
            let cur_tpl_base = cur_trans_prob.base();
            for row in 1..(alpha_dp.shape()[0] - 1) {
                let cur_read_base_enc = encoded_emit[row - 1];
                // for row1 col1, only match trans to this

                if row > 0 && col > 0 {
                    // match, update emission
                    hmm_builder.add_to_move_ctx_emit_prob_numerator(
                        encode_2_bases(prev_tpl_base, cur_tpl_base),
                        model::Move::Match,
                        cur_read_base_enc,
                        alpha_dp[[row - 1, col - 1]]
                            * prev_trans_probs.prob(model::Move::Match)
                            * beta_dp[[row, col]],
                    );
                }

                if row > 1 && col > 1 {
                    // match, update state
                    hmm_builder.add_to_ctx_move_prob_numerator(
                        encode_2_bases(prev_tpl_base, cur_tpl_base),
                        model::Move::Match,
                        alpha_dp[[row - 1, col - 1]]
                            * prev_trans_probs.prob(model::Move::Match)
                            * beta_dp[[row, col]],
                    );
                }

                if row > 1 {
                    // insertion
                    let next_trans_probs = tpl[col];
                    let next_tpl_base = next_trans_probs.base();

                    hmm_builder.add_to_move_ctx_emit_prob_numerator(
                        encode_2_bases(cur_tpl_base, next_tpl_base),
                        model::Move::Branch,
                        cur_read_base_enc,
                        alpha_dp[[row - 1, col]]
                            * prev_trans_probs.prob(model::Move::Branch)
                            * beta_dp[[row, col]],
                    );

                    hmm_builder.add_to_move_ctx_emit_prob_numerator(
                        encode_2_bases(cur_tpl_base, next_tpl_base),
                        model::Move::Stick,
                        cur_read_base_enc,
                        alpha_dp[[row - 1, col]]
                            * prev_trans_probs.prob(model::Move::Stick)
                            * beta_dp[[row, col]],
                    );

                    hmm_builder.add_to_ctx_move_prob_numerator(
                        encode_2_bases(cur_tpl_base, next_tpl_base),
                        model::Move::Branch,
                        alpha_dp[[row - 1, col]]
                            * prev_trans_probs.prob(model::Move::Branch)
                            * beta_dp[[row, col]],
                    );

                    hmm_builder.add_to_ctx_move_prob_numerator(
                        encode_2_bases(cur_tpl_base, next_tpl_base),
                        model::Move::Stick,
                        alpha_dp[[row - 1, col]]
                            * prev_trans_probs.prob(model::Move::Stick)
                            * beta_dp[[row, col]],
                    );
                }

                if col > 1 {
                    hmm_builder.add_to_ctx_move_prob_numerator(
                        encode_2_bases(prev_tpl_base, cur_tpl_base),
                        model::Move::Dark,
                        alpha_dp[[row, col - 1]]
                            * prev_trans_probs.prob(model::Move::Dark)
                            * beta_dp[[row, col]],
                    );
                }
            }

            prev_trans_probs = cur_trans_prob;
            prev_tpl_base = cur_tpl_base;
        }

        let (tot_row, tot_col) = (alpha_dp.shape()[0], alpha_dp.shape()[1]);
        let cur_tpl_base = tpl.last().unwrap().base();
        let cur_base_enc = *encoded_emit.last().unwrap();
        hmm_builder.add_to_move_ctx_emit_prob_numerator(
            encode_2_bases(prev_tpl_base, cur_tpl_base),
            model::Move::Match,
            cur_base_enc,
            alpha_dp[[tot_row - 2, tot_col - 2]] * beta_dp[[tot_row - 1, tot_col - 1]],
        );
    }

    hmm_builder
}
