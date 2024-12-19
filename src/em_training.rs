use std::{
    sync::{Arc, Mutex},
    thread,
};

use crate::{
    cli::TrainingParams,
    common::{TrainInstance, TransState},
    dataset::{align_record_read_worker, encode_emit, train_instance_worker},
    hmm_model::{HmmBuilderV2, HmmModel},
};
use crossbeam::channel;
use fb::{backward, backward_with_log_sum_exp_trick, forward, forward_with_log_sum_exp_trick, veterbi_decode};
use gskits::pbar::{get_spin_pb, DEFAULT_INTERVAL};
use model::{encode_2_bases, Template, TemplatePos};
pub mod fb;
pub mod model;

pub fn em_training(params: &TrainingParams, mut hmm_model: HmmModel) {
    let aligned_bams = &params.aligned_bams;
    let ref_fastas = &params.ref_fas;
    let dw_boundaries = &params.parse_dw_boundaries();
    assert!(aligned_bams.len() == ref_fastas.len());

    for epoch in 0..1000 {
        let pbar = get_spin_pb(
            format!("epoch:{} --> em training...", epoch),
            DEFAULT_INTERVAL,
        );

        let pbar = Arc::new(Mutex::new(pbar));
        let new_hmm_model: HmmModel = thread::scope(|s| {
            let aligned_bams = &params.aligned_bams;
            let ref_fastas = &params.ref_fas;
            let dw_boundaries = &dw_boundaries;
            let hmm_model_ref = &hmm_model;

            let (record_sender, record_receiver) = channel::bounded(1000);
            aligned_bams
                .iter()
                .zip(ref_fastas.iter())
                .for_each(|(aligned_bam, ref_fasta)| {
                    let record_sender_ = record_sender.clone();
                    let pbar_ = pbar.clone();
                    s.spawn(move || {
                        align_record_read_worker(aligned_bam, ref_fasta, pbar_, record_sender_);
                    });
                });
            drop(record_sender);
            drop(pbar);

            let (train_ins_sender, train_ins_receiver) = channel::bounded(1000);
            for _ in 0..(num_cpus::get() / 4) {
                let record_receiver_ = record_receiver.clone();
                let train_ins_sender_ = train_ins_sender.clone();
                s.spawn(move || {
                    train_instance_worker(dw_boundaries, record_receiver_, train_ins_sender_)
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
            (&final_hmm_builder).into()
        });

        tracing::info!("compute delta");
        let delta = hmm_model.delta(&new_hmm_model);
        if delta < 1e-6 {
            new_hmm_model.dump_to_file(&format!("arrow_hg002.em-epoch-{}.params", epoch));
            println!("DONE!!!!");
            break;
        } else {
            println!("epoch:{}, delta:{}", epoch, delta);
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
    let mut should_print = true;
    for train_ins in train_ins_receiver {
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

        if should_print && idx == 0 {
            should_print = false;
            tracing::info!("qname: {}, align:\n{}", train_ins.name, veterbi_decode(&encoded_emit, &tpl, hmm_model));
        }

        let alpha_dp = forward_with_log_sum_exp_trick(&encoded_emit, &tpl, hmm_model);
        let beta_dp = backward_with_log_sum_exp_trick(&encoded_emit, &tpl, hmm_model);

        assert_eq!(alpha_dp.shape(), beta_dp.shape());

        let mut prev_trans_probs = TemplatePos::default();
        let mut prev_tpl_base = prev_trans_probs.base();

        for col in 1..(alpha_dp.shape()[1] - 1) {
            let cur_trans_prob = tpl[col - 1];
            let cur_tpl_base = cur_trans_prob.base();
            for row in 1..(alpha_dp.shape()[0] - 1) {
                let cur_read_base_enc = encoded_emit[row - 1];
                // for row1 col1, only match trans to this

                if (row > 1 && col > 1) && (row == 1 && col == 1) {
                    // match, update emission
                    hmm_builder.add_to_move_ctx_emit_prob_numerator(
                        encode_2_bases(prev_tpl_base, cur_tpl_base),
                        TransState::Match,
                        cur_read_base_enc,
                        (alpha_dp[[row - 1, col - 1]]
                            + prev_trans_probs.prob(TransState::Match).ln()
                            + beta_dp[[row, col]]).exp(),
                    );
                }

                if row > 1 && col > 1 {
                    // match, update state
                    hmm_builder.add_to_ctx_move_prob_numerator(
                        encode_2_bases(prev_tpl_base, cur_tpl_base),
                        TransState::Match,
                        (alpha_dp[[row - 1, col - 1]]
                            + prev_trans_probs.prob(TransState::Match).ln()
                            + beta_dp[[row, col]]).exp(),
                    );
                }

                if row > 1 {
                    // insertion
                    let next_trans_probs = tpl[col];
                    let next_tpl_base = next_trans_probs.base();

                    hmm_builder.add_to_move_ctx_emit_prob_numerator(
                        encode_2_bases(cur_tpl_base, next_tpl_base),
                        TransState::Branch,
                        cur_read_base_enc,
                        (alpha_dp[[row - 1, col]]
                            + cur_trans_prob.prob(TransState::Branch).ln()
                            + beta_dp[[row, col]]).exp(),
                    );

                    hmm_builder.add_to_move_ctx_emit_prob_numerator(
                        encode_2_bases(cur_tpl_base, next_tpl_base),
                        TransState::Stick,
                        cur_read_base_enc,
                        (alpha_dp[[row - 1, col]]
                            + cur_trans_prob.prob(TransState::Stick).ln()
                            + beta_dp[[row, col]]).exp(),
                    );

                    hmm_builder.add_to_ctx_move_prob_numerator(
                        encode_2_bases(cur_tpl_base, next_tpl_base),
                        TransState::Branch,
                        (alpha_dp[[row - 1, col]]
                            + cur_trans_prob.prob(TransState::Branch).ln()
                            + beta_dp[[row, col]]).exp(),
                    );

                    hmm_builder.add_to_ctx_move_prob_numerator(
                        encode_2_bases(cur_tpl_base, next_tpl_base),
                        TransState::Stick,
                        (alpha_dp[[row - 1, col]]
                            + cur_trans_prob.prob(TransState::Stick).ln()
                            + beta_dp[[row, col]]).exp(),
                    );
                }

                if col > 1 {
                    hmm_builder.add_to_ctx_move_prob_numerator(
                        encode_2_bases(prev_tpl_base, cur_tpl_base),
                        TransState::Dark,
                        (alpha_dp[[row, col - 1]]
                            + prev_trans_probs.prob(TransState::Dark).ln()
                            + beta_dp[[row, col]]).exp(),
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
            TransState::Match,
            cur_base_enc,
            (alpha_dp[[tot_row - 2, tot_col - 2]] + beta_dp[[tot_row - 1, tot_col - 1]]).exp(),
        );
    }

    hmm_builder
}
