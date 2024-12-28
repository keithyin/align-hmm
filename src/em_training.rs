pub mod fb;
pub mod fb_v2;
pub mod model;

use std::{
    sync::{Arc, Mutex},
    thread,
};

use crate::{
    cli::TrainingParams,
    common::{TrainInstance, TransState},
    dataset::{align_record_read_worker, encode_emit, read_refs, train_instance_worker},
    hmm_model::{HmmBuilderV2, HmmModel},
};
use crossbeam::channel;
use fb::veterbi_decode;
use fb_v2::{backward_with_log_sum_exp_trick, forward_with_log_sum_exp_trick};
use gskits::pbar::{get_spin_pb, DEFAULT_INTERVAL};
use model::{decode_emit_base, encode_2_bases, Template, TemplatePos};

pub fn em_training(params: &TrainingParams, mut hmm_model: HmmModel) {
    let aligned_bams = &params.aligned_bams;
    let ref_fastas = &params.ref_fas;
    let dw_boundaries = &params.parse_dw_boundaries();
    assert!(aligned_bams.len() == ref_fastas.len());

    let bam2refs = read_refs(aligned_bams, ref_fastas);

    let mut last_ll: Option<f64> = None;

    for epoch in 0..1000 {
        let bam2refs = &bam2refs;
        let pbar = get_spin_pb(
            format!("epoch:{} --> em training...", epoch),
            DEFAULT_INTERVAL,
        );

        let pbar = Arc::new(Mutex::new(pbar));
        let (new_hmm_model, finished): (HmmModel, bool) = thread::scope(|s| {
            let aligned_bams = &params.aligned_bams;
            let dw_boundaries = &dw_boundaries;
            let hmm_model_ref = &hmm_model;

            let (record_sender, record_receiver) = channel::bounded(1000);
            aligned_bams.iter().for_each(|aligned_bam| {
                let record_sender_ = record_sender.clone();
                let pbar_ = pbar.clone();
                s.spawn(move || {
                    align_record_read_worker(
                        aligned_bam,
                        bam2refs.get(aligned_bam).unwrap(),
                        pbar_,
                        record_sender_,
                    );
                });
            });
            drop(record_sender);
            drop(pbar);

            let (train_ins_sender, train_ins_receiver) = channel::bounded(1000);
            for _ in 0..(num_cpus::get() / 4) {
                let record_receiver_ = record_receiver.clone();
                let train_ins_sender_ = train_ins_sender.clone();
                s.spawn(move || {
                    train_instance_worker(Some(dw_boundaries), record_receiver_, train_ins_sender_)
                });
            }
            drop(record_receiver);
            drop(train_ins_sender);

            let mut handles = vec![];
            for idx in 0..(num_cpus::get() / 4 * 3) {
                let train_ins_receiver_ = train_ins_receiver.clone();
                handles
                    .push(s.spawn(move || train_worker(train_ins_receiver_, hmm_model_ref, idx)));
            }
            drop(train_ins_receiver);

            let mut final_hmm_builder = HmmBuilderV2::new();

            handles.into_iter().for_each(|h| {
                let hb = h.join().unwrap();
                final_hmm_builder.merge(&hb);
            });
            let cur_ll = final_hmm_builder.get_log_likelihood().unwrap();

            tracing::info!(
                "epoch:{}, pre_log_likelihood:{:?}, cur_log_likelihood:{}",
                epoch,
                last_ll,
                cur_ll
            );

            let mut finished = false;
            if let Some(last_ll_) = last_ll {
                let delta = (cur_ll - last_ll_).abs();
                tracing::info!("epoch:{}, delta:{}", epoch, delta);
                if delta < 1e-6 {
                    finished = true;
                }
                last_ll = Some(cur_ll);
            } else {
                last_ll = Some(cur_ll);
            }

            let new_hmm_model = (&final_hmm_builder).into();
            (new_hmm_model, finished)
        });

        if finished {
            new_hmm_model.dump_to_file(&format!("arrow_hg002.em-epoch-{}.params", epoch));
            println!("DONE!!!!");
            break;
        } else {
            new_hmm_model.dump_to_file(&format!("arrow_hg002.em-epoch-{}.params", epoch));
        }
        hmm_model = new_hmm_model;
    }
}

pub fn train_worker(
    train_ins_receiver: channel::Receiver<TrainInstance>,
    hmm_model: &HmmModel,
    idx: usize,
) -> HmmBuilderV2 {
    let mut hmm_builder = HmmBuilderV2::new();
    let mut ins_cnt = 0;
    for train_ins in train_ins_receiver {
        train_with_single_instance(
            &train_ins,
            hmm_model,
            &mut hmm_builder,
            ins_cnt == 0 && idx == 0,
        );
        ins_cnt += 1;
    }

    hmm_builder
}

pub fn train_with_single_instance(
    train_ins: &TrainInstance,
    hmm_model: &HmmModel,
    hmm_builder: &mut HmmBuilderV2,
    print_align_result: bool,
) {
    let rseq = train_ins.ref_aligned_seq().replace('-', "");
    let qseq = train_ins.read_aligned_seq().replace('-', "");
    let dwell_time = train_ins
        .dw()
        .iter()
        .filter(|v| v.is_some())
        .map(|v| v.unwrap())
        .collect::<Vec<u8>>();

    assert_eq!(qseq.len(), dwell_time.len());

    let tpl = Template::from_template_bases(rseq.as_bytes(), hmm_model);
    let encoded_emit = encode_emit(&dwell_time, &qseq);

    if print_align_result {
        tracing::info!(
            "qname: {}, align:\n{}",
            train_ins.name,
            veterbi_decode(&encoded_emit, &tpl, hmm_model)
        );
    }

    let alpha_dp = forward_with_log_sum_exp_trick(&encoded_emit, &tpl, hmm_model);
    let beta_dp = backward_with_log_sum_exp_trick(&encoded_emit, &tpl, hmm_model);
    hmm_builder.add_log_likehood(beta_dp[[0, 0]], 1);

    assert_eq!(alpha_dp.shape(), beta_dp.shape());

    let mut prev_trans_probs = TemplatePos::default();
    let mut prev_tpl_base = prev_trans_probs.base();

    for col in 1..(alpha_dp.shape()[1] - 1) {
        let cur_trans_prob = tpl[col - 1];
        let cur_tpl_base = cur_trans_prob.base();
        for row in 1..(alpha_dp.shape()[0] - 1) {
            let cur_read_base_enc = encoded_emit[row - 1];
            // for row1 col1, only match trans to this

            if (row > 1 && col > 1) || (row == 1 && col == 1) {
                // match, update emission
                let ctx = encode_2_bases(prev_tpl_base, cur_tpl_base);
                let log_prob = alpha_dp[[row - 1, col - 1]]
                    + prev_trans_probs.ln_prob(TransState::Match)
                    + hmm_model.emit_ln_prob(TransState::Match, ctx, cur_read_base_enc)
                    + beta_dp[[row, col]];
                hmm_builder.add_to_state_ctx_emit_prob_numerator(
                    ctx,
                    TransState::Match,
                    cur_read_base_enc,
                    log_prob,
                );
            }

            if row > 1 && col > 1 {
                // match, update state
                let ctx = encode_2_bases(prev_tpl_base, cur_tpl_base);

                let log_prob = alpha_dp[[row - 1, col - 1]]
                    + prev_trans_probs.ln_prob(TransState::Match)
                    + hmm_model.emit_ln_prob(TransState::Match, ctx, cur_read_base_enc)
                    + beta_dp[[row, col]];

                hmm_builder.add_to_ctx_state_prob_numerator(ctx, TransState::Match, log_prob);
            }

            if row > 1 {
                // insertion
                let next_trans_probs = tpl[col];
                let next_tpl_base = next_trans_probs.base();
                let ctx = encode_2_bases(cur_tpl_base, next_tpl_base);

                if decode_emit_base(cur_read_base_enc).as_bytes()[1] == next_tpl_base {
                    let log_prob = alpha_dp[[row - 1, col]]
                        + cur_trans_prob.ln_prob(TransState::Branch)
                        + hmm_model.emit_ln_prob(TransState::Branch, ctx, cur_read_base_enc)
                        + beta_dp[[row, col]];

                    hmm_builder.add_to_state_ctx_emit_prob_numerator(
                        ctx,
                        TransState::Branch,
                        cur_read_base_enc,
                        log_prob,
                    );

                    hmm_builder.add_to_ctx_state_prob_numerator(ctx, TransState::Branch, log_prob);
                } else {
                    let log_prob = alpha_dp[[row - 1, col]]
                        + cur_trans_prob.ln_prob(TransState::Stick)
                        + hmm_model.emit_ln_prob(TransState::Stick, ctx, cur_read_base_enc)
                        + beta_dp[[row, col]];
                    hmm_builder.add_to_state_ctx_emit_prob_numerator(
                        ctx,
                        TransState::Stick,
                        cur_read_base_enc,
                        log_prob,
                    );

                    hmm_builder.add_to_ctx_state_prob_numerator(ctx, TransState::Stick, log_prob);
                }
            }

            if col > 1 {
                let log_prob = alpha_dp[[row, col - 1]]
                    + prev_trans_probs.ln_prob(TransState::Dark)
                    + beta_dp[[row, col]];
                hmm_builder.add_to_ctx_state_prob_numerator(
                    encode_2_bases(prev_tpl_base, cur_tpl_base),
                    TransState::Dark,
                    log_prob,
                );
            }
        }

        prev_trans_probs = cur_trans_prob;
        prev_tpl_base = cur_tpl_base;
    }

    let (tot_row, tot_col) = (alpha_dp.shape()[0], alpha_dp.shape()[1]);
    let cur_tpl_base = tpl.last().unwrap().base();
    let cur_base_enc = *encoded_emit.last().unwrap();
    hmm_builder.add_to_state_ctx_emit_prob_numerator(
        encode_2_bases(prev_tpl_base, cur_tpl_base),
        TransState::Match,
        cur_base_enc,
        (alpha_dp[[tot_row - 2, tot_col - 2]]
            + beta_dp[[tot_row - 1, tot_col - 1]]
            + hmm_model.emit_ln_prob(
                TransState::Match,
                encode_2_bases(prev_tpl_base, cur_tpl_base),
                cur_base_enc,
            )),
    );
}

#[cfg(test)]
mod test {
    use crate::{common::TrainInstance, hmm_model::HmmBuilderV2, hmm_models::boundaries_4_100};

    use super::train_with_single_instance;

    #[test]
    fn test_train_with_single_ins() {
        let train_ins = TrainInstance::new(
            "ACGT".to_string(),
            "AGGT".to_string(),
            vec![Some(1), Some(1), Some(1), Some(1)],
            "q1".to_string(),
        );

        let mut hmm_model = boundaries_4_100::get_hmm_model();
        for _ in 0..10 {
            let mut hmm_builder = HmmBuilderV2::new();
            for idx in 0..100 {
                train_with_single_instance(&train_ins, &hmm_model, &mut hmm_builder, idx == 0);
            }
            hmm_model = (&hmm_builder).into();
            hmm_model.print_params();
        }
    }
}
