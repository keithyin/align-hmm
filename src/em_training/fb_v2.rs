use ndarray::Array2;

use crate::{
    common::IDX_BASE_MAP,
    hmm_model::{log_sum_exp, HmmModel},
};

use super::model::{decode_2_bases, decode_emit_base, encode_2_bases, Template, TemplatePos};
use crate::common::TransState;

/// forward backward

pub fn forward_with_log_sum_exp_trick(
    encoded_query: &[u8],
    template: &Template,
    hmm_model: &HmmModel,
) -> Array2<f64> {
    let dp_rows = encoded_query.len() + 1;
    let dp_cols = template.len() + 1;

    let mut dp_matrix: Array2<f64> = Array2::from_shape_fn((dp_rows, dp_cols), |_| f64::MIN);
    dp_matrix[[0, 0]] = 0.0;

    let mut prev_trans_probs = TemplatePos::default();
    let mut prev_tpl_base = prev_trans_probs.base();

    for col in 1..(dp_cols - 1) {
        let cur_trans_probs = template[col - 1];
        let cur_tpl_base = cur_trans_probs.base();
        for row in 1..(dp_rows - 1) {
            let cur_read_base_enc = encoded_query[row - 1];
            let mut scores = vec![];

            if (row > 1 && col > 1) || (row == 1 && col == 1) {
                let this_move_score = dp_matrix[[row - 1, col - 1]]
                    + prev_trans_probs.prob(TransState::Match).ln()
                    + hmm_model
                        .emit_prob(
                            TransState::Match,
                            encode_2_bases(prev_tpl_base, cur_tpl_base),
                            cur_read_base_enc,
                        )
                        .ln();
                scores.push(this_move_score);
            }

            if row > 1 {
                // ins here
                let next_trans_probs = template[col];
                let next_tpl_base = next_trans_probs.base();

                let this_move_score =
                    if decode_emit_base(cur_read_base_enc).as_bytes()[1] == next_tpl_base {
                        dp_matrix[[row - 1, col]]
                            + cur_trans_probs.prob(TransState::Branch).ln()
                            + hmm_model
                                .emit_prob(
                                    TransState::Branch,
                                    encode_2_bases(cur_tpl_base, next_tpl_base),
                                    cur_read_base_enc,
                                )
                                .ln()
                    } else {
                        dp_matrix[[row - 1, col]]
                            + cur_trans_probs.prob(TransState::Stick).ln()
                            + hmm_model
                                .emit_prob(
                                    TransState::Stick,
                                    encode_2_bases(cur_tpl_base, next_tpl_base),
                                    cur_read_base_enc,
                                )
                                .ln()
                    };

                scores.push(this_move_score);
            }

            if col > 1 {
                // dark here
                let this_move_score =
                    dp_matrix[[row, col - 1]] + prev_trans_probs.prob(TransState::Dark).ln();
                // println!("row:{}, col:{}, prev_score:{}, this_move:{}, score:{}", row, col, dp_matrix[[row, col - 1]], prev_trans_probs.prob(TransState::Dark).ln(), this_move_score);
                scores.push(this_move_score);
            }

            dp_matrix[[row, col]] = log_sum_exp(&scores);
        }

        prev_trans_probs = cur_trans_probs;
        prev_tpl_base = cur_tpl_base;
    }

    let cur_tpl_base = template.last().unwrap().base();
    dp_matrix[[dp_rows - 1, dp_cols - 1]] = dp_matrix[[dp_rows - 2, dp_cols - 2]]
        + hmm_model
            .emit_prob(
                TransState::Match,
                encode_2_bases(prev_tpl_base, cur_tpl_base),
                *encoded_query.last().unwrap(),
            )
            .ln();

    dp_matrix
}

pub fn backward_with_log_sum_exp_trick(
    encoded_query: &[u8],
    template: &Template,
    hmm_model: &HmmModel,
) -> Array2<f64> {
    let dp_rows = encoded_query.len() + 1;
    let dp_cols = template.len() + 1;

    let mut dp_matrix: Array2<f64> = Array2::from_shape_fn((dp_rows, dp_cols), |_| f64::MIN);
    dp_matrix[[dp_rows - 1, dp_cols - 1]] = 0.0;

    for col in (1..(dp_cols - 1)).rev() {
        let next_trans_probs = template[col];
        let next_tpl_base = next_trans_probs.base();

        let cur_trans_probs = template[col - 1];
        let cur_tpl_base = cur_trans_probs.base();

        for row in (1..(dp_rows - 1)).rev() {
            let next_read_base_enc = encoded_query[row];
            let mut scores = vec![];
            // match
            if (row + 1) == (dp_rows - 1) && (col + 1) == (dp_cols - 1) {
                let this_move_score = hmm_model
                    .emit_prob(
                        TransState::Match,
                        encode_2_bases(cur_tpl_base, next_tpl_base),
                        next_read_base_enc,
                    )
                    .ln()
                    + dp_matrix[[row + 1, col + 1]];
                scores.push(this_move_score);
            } else if (row + 1) < (dp_rows - 1) && (col + 1) < (dp_cols - 1) {
                let this_move_score = cur_trans_probs.prob(TransState::Match).ln()
                    + hmm_model
                        .emit_prob(
                            TransState::Match,
                            encode_2_bases(cur_tpl_base, next_tpl_base),
                            next_read_base_enc,
                        )
                        .ln()
                    + dp_matrix[[row + 1, col + 1]];
                scores.push(this_move_score);
            }

            if row < (dp_rows - 2) {
                // ins here
                let this_move_score = if decode_emit_base(next_read_base_enc).as_bytes()[0] == next_tpl_base {
                    cur_trans_probs.prob(TransState::Branch).ln()
                    + hmm_model
                        .emit_prob(
                            TransState::Branch,
                            encode_2_bases(cur_tpl_base, next_tpl_base),
                            next_read_base_enc,
                        )
                        .ln()
                    + dp_matrix[[row + 1, col]]
                } else {
                    cur_trans_probs.prob(TransState::Stick).ln()
                    + hmm_model
                        .emit_prob(
                            TransState::Stick,
                            encode_2_bases(cur_tpl_base, next_tpl_base),
                            next_read_base_enc,
                        )
                        .ln()
                    + dp_matrix[[row + 1, col]]
                };

                scores.push(this_move_score);
            }

            if col < (dp_cols - 2) {
                // dark here
                let this_move_score =
                    cur_trans_probs.prob(TransState::Dark).ln() + dp_matrix[[row, col + 1]];
                scores.push(this_move_score);
            }

            dp_matrix[[row, col]] = log_sum_exp(&scores);
        }
    }
    let default_tpl_pos = TemplatePos::default();
    let default_tpl_base = default_tpl_pos.base();
    dp_matrix[[0, 0]] = hmm_model
        .emit_prob(
            TransState::Match,
            encode_2_bases(default_tpl_base, template.first().unwrap().base()),
            *encoded_query.first().unwrap(),
        )
        .ln()
        + dp_matrix[[1, 1]];

    dp_matrix
}


#[cfg(test)]
mod test {

    use crate::{
        dataset::encode_emit,
        em_training::{
            fb::veterbi_decode, fb_v2::{backward_with_log_sum_exp_trick, forward_with_log_sum_exp_trick}, model::Template
        },
        hmm_models::v1,
    };


    #[test]
    fn test_forward_backward() {
        let query_bases = vec![0, 1, 2, 3];
        let templates = vec![0, 1, 2, 3];

        let hmm_model = v1::get_hmm_model();

        let templates = Template::from_template_bases(&templates, &hmm_model);

        let alpha = forward_with_log_sum_exp_trick(&query_bases, &templates, &hmm_model);
        let beta = backward_with_log_sum_exp_trick(&query_bases, &templates, &hmm_model);
        println!("{:?}", alpha);
        println!("{:?}", beta);
        println!("{}", beta[[0, 0]].exp());
        println!("{}", (alpha + beta).mapv(|v| v.exp()));
    }

    #[test]
    fn test_forward_backward_0() {
        let query_bases = vec![0, 1, 2];
        let templates = vec![0, 1, 2];

        let hmm_model = v1::get_hmm_model();

        let templates = Template::from_template_bases(&templates, &hmm_model);

        let alpha = forward_with_log_sum_exp_trick(&query_bases, &templates, &hmm_model);
        let beta = backward_with_log_sum_exp_trick(&query_bases, &templates, &hmm_model);
        println!("{:?}", alpha);
        println!("{:?}", beta);
        println!("{}", beta[[0, 0]].exp());
        println!("{}", (alpha + beta).mapv(|v| v.exp()));
    }

    #[test]
    fn test_forward_backward_2() {
        let hmm_model = v1::get_hmm_model();

        let query_bases = vec![0, 1, 2, 3];
        let templates = b"ACGTAAGT".to_vec();
        let templates = Template::from_template_bases(&templates, &hmm_model);
        println!("{}", veterbi_decode(&query_bases, &templates, &hmm_model));

        let templates = b"CCGAACTAGTCTCAGGCTTCAACATCGAATACGCCGCAGGCCCCTTCGCCCTATTCTTCATAGCCGAATACACAAACATTATTATAATAAACACCCTCACCACTACAATCTTCCTAGGAACAACATATAACGCACTCTCCCCTGAACTCTACACAACATATTTTGTCACCAAGACCCTACTTCTGACCTCCCTGTTCTTATGAATTCGAACAGCATACCCCCGATTCCGC";
        let query = "GGGAGCCGAACTAGTCTCAGGCTTCACATCGAGTTTCCCGCAGCCTTCGCCCTATCTTCCATAGCCGAATACACAAACATATATAATAAACACCCTCACCACTACAATCTTCCTAGGAACAACATATGACGCACTCTCCCCTGAACTCCTACACAACATATTTTGTCACCAAGACCCTACTTCTAACCTCCCTGTTCTTATAATTCGAACAGCATCACCCCCGATTCCGC";
        let encoded_emit = encode_emit(&vec![0; 230], query);
        let templates = Template::from_template_bases(templates, &hmm_model);
        println!("{}", veterbi_decode(&encoded_emit, &templates, &hmm_model));

        let templates = b"CCGAACTAGTCTCAGGCTTCAACATCGAATACGCCGCAGGCCCCTTCGCCCTATTCTTCATAGCCGAATACACAAACATTATTATAATAAACACCCTCACCACTACAATCTTCCTAGGAACAACATATAACGCACTCTCCCCTGAACTCTACACAACATATTTTGTCACCAAGACCCTACTTCTGACCTCCCTGTTCTTATGAATTCGAACAGCATACCCCCGATTCCGC";
        let query = "GGGAGCCGAACTAGTCTCAGGCTTCACATCGAGTTTCCCGCAGCCTTCGCCCTATCTTCCATAGCCGAATACACAAACATATATAATAAACACCCTCACCACTACAATCTTCCTAGGAACAACATATGACGCACTCTCCCCTGAACTCCTACACAACATATTTTGTCACCAAGACCCTACTTCTAACCTCCCTGTTCTTATAATTCGAACAGCATCACCCCCGATTCCGC";
        let encoded_emit = encode_emit(&vec![0; 230], query);
        let hmm_model = v1::get_hmm_model();
        let templates = Template::from_template_bases(templates, &hmm_model);
        println!("{}", veterbi_decode(&encoded_emit, &templates, &hmm_model));
    }

    #[test]
    pub fn test_max_by() {
        let values = vec![0.6, 0.7, 0.8];
        let v = values
            .iter()
            .max_by(|&&a, &&b| a.partial_cmp(&b).unwrap())
            .copied()
            .unwrap();
        println!("{}", v);
    }
}
