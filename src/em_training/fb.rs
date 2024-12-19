use ndarray::Array2;

use crate::{
    common::IDX_BASE_MAP,
    hmm_model::{log_sum_exp, HmmModel},
};

use super::model::{decode_2_bases, decode_emit_base, encode_2_bases, Template, TemplatePos};
use crate::common::TransState;

/// forward backward

/// 2D DP, 假设 template 是 ATC, called 是 ATC，那么会构建如下DP
///  ' AT TC C-
/// '  
/// A  *
/// T
/// C        *
/// 将坐标看作状态，初始情况下 在 a_11 的概率为 1， 所以可以理解为 初始状态 a_11 的概率是 1.
///  因为已经pin了开始和结束，所以 到第一个 baes 的转移，和 到最后一个 base 的转移，都应该是 match！！！ln prob
pub fn forward(encoded_query: &[u8], template: &Template, hmm_model: &HmmModel) -> Array2<f64> {
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
            let mut score = 0.0;
            let cur_read_base_enc = encoded_query[row - 1];

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
                score += this_move_score.exp();
            }

            if row > 1 {
                // ins here
                let next_trans_probs = template[col];
                let next_tpl_base = next_trans_probs.base();
                let this_move_score = dp_matrix[[row - 1, col]]
                    + cur_trans_probs.prob(TransState::Branch).ln()
                    + hmm_model
                        .emit_prob(
                            TransState::Branch,
                            encode_2_bases(cur_tpl_base, next_tpl_base),
                            cur_read_base_enc,
                        )
                        .ln();

                score += this_move_score.exp();
                let this_move_score = dp_matrix[[row - 1, col]]
                    + cur_trans_probs.prob(TransState::Stick).ln()
                    + hmm_model
                        .emit_prob(
                            TransState::Stick,
                            encode_2_bases(cur_tpl_base, next_tpl_base),
                            cur_read_base_enc,
                        )
                        .ln();

                score += this_move_score.exp();
            }

            if col > 1 {
                // dark here
                let this_move_score =
                    dp_matrix[[row, col - 1]] + prev_trans_probs.prob(TransState::Dark).ln();
                score += this_move_score.exp();
            }

            dp_matrix[[row, col]] = score.ln();
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
                let this_move_score = dp_matrix[[row - 1, col]]
                    + cur_trans_probs.prob(TransState::Branch).ln()
                    + hmm_model
                        .emit_prob(
                            TransState::Branch,
                            encode_2_bases(cur_tpl_base, next_tpl_base),
                            cur_read_base_enc,
                        )
                        .ln();
                scores.push(this_move_score);

                let this_move_score = dp_matrix[[row - 1, col]]
                    + cur_trans_probs.prob(TransState::Stick).ln()
                    + hmm_model
                        .emit_prob(
                            TransState::Stick,
                            encode_2_bases(cur_tpl_base, next_tpl_base),
                            cur_read_base_enc,
                        )
                        .ln();
                scores.push(this_move_score);
            }

            if col > 1 {
                // dark here
                let this_move_score =
                    dp_matrix[[row, col - 1]] + prev_trans_probs.prob(TransState::Dark).ln();
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

pub fn backward(encoded_query: &[u8], template: &Template, hmm_model: &HmmModel) -> Array2<f64> {
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
            let mut score = 0.0;
            let next_read_base_enc = encoded_query[row];

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

                score += this_move_score.exp();
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

                score += this_move_score.exp();
            }

            if row < (dp_rows - 2) {
                // ins here
                let this_move_score = cur_trans_probs.prob(TransState::Branch).ln()
                    + hmm_model
                        .emit_prob(
                            TransState::Branch,
                            encode_2_bases(cur_tpl_base, next_tpl_base),
                            next_read_base_enc,
                        )
                        .ln()
                    + dp_matrix[[row + 1, col]];

                score += this_move_score.exp();
                let this_move_score = cur_trans_probs.prob(TransState::Stick).ln()
                    + hmm_model
                        .emit_prob(
                            TransState::Stick,
                            encode_2_bases(cur_tpl_base, next_tpl_base),
                            next_read_base_enc,
                        )
                        .ln()
                    + dp_matrix[[row + 1, col]];

                score += this_move_score.exp();
            }

            if col < (dp_cols - 2) {
                // dark here
                let this_move_score =
                    cur_trans_probs.prob(TransState::Dark).ln() + dp_matrix[[row, col + 1]];
                score += this_move_score.exp();
            }

            dp_matrix[[row, col]] = score.ln();
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
                let this_move_score = cur_trans_probs.prob(TransState::Branch).ln()
                    + hmm_model
                        .emit_prob(
                            TransState::Branch,
                            encode_2_bases(cur_tpl_base, next_tpl_base),
                            next_read_base_enc,
                        )
                        .ln()
                    + dp_matrix[[row + 1, col]];
                scores.push(this_move_score);
                let this_move_score = cur_trans_probs.prob(TransState::Stick).ln()
                    + hmm_model
                        .emit_prob(
                            TransState::Stick,
                            encode_2_bases(cur_tpl_base, next_tpl_base),
                            next_read_base_enc,
                        )
                        .ln()
                    + dp_matrix[[row + 1, col]];
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

pub fn veterbi_decode(encoded_query: &[u8], template: &Template, hmm_model: &HmmModel) -> String {
    let dp_rows = encoded_query.len() + 1;
    let dp_cols = template.len() + 1;

    let mut dp_matrix: Array2<f64> = Array2::from_shape_fn((dp_rows, dp_cols), |_| f64::MIN);
    dp_matrix[[0, 0]] = 0.0;

    let mut state_matrix: Array2<TransState> =
        Array2::from_shape_fn((dp_rows, dp_cols), |_| TransState::Match);
    state_matrix[[0, 0]] = TransState::Match;

    let mut prev_trans_probs = TemplatePos::default();
    let mut prev_tpl_base = prev_trans_probs.base();

    for col in 1..(dp_cols - 1) {
        let cur_trans_probs = template[col - 1];
        let cur_tpl_base = cur_trans_probs.base();
        for row in 1..(dp_rows - 1) {
            let cur_read_base_enc = encoded_query[row - 1];
            let mut match_score = f64::MIN;
            if (row > 1 && col > 1) || (row == 1 && col == 1) {
                match_score = dp_matrix[[row - 1, col - 1]]
                    + prev_trans_probs.prob(TransState::Match).ln()
                    + hmm_model
                        .emit_prob(
                            TransState::Match,
                            encode_2_bases(prev_tpl_base, cur_tpl_base),
                            cur_read_base_enc,
                        )
                        .ln();
            }

            let mut branch_score = f64::MIN;
            let mut stick_score = f64::MIN;
            if row > 1 {
                // ins here

                let next_trans_probs = template[col];
                let next_tpl_base = next_trans_probs.base();
                branch_score = dp_matrix[[row - 1, col]]
                    + cur_trans_probs.prob(TransState::Branch).ln()
                    + hmm_model
                        .emit_prob(
                            TransState::Branch,
                            encode_2_bases(cur_tpl_base, next_tpl_base),
                            cur_read_base_enc,
                        )
                        .ln();

                stick_score = dp_matrix[[row - 1, col]]
                    + cur_trans_probs.prob(TransState::Stick).ln()
                    + hmm_model
                        .emit_prob(
                            TransState::Stick,
                            encode_2_bases(cur_tpl_base, next_tpl_base),
                            cur_read_base_enc,
                        )
                        .ln();
            }
            let mut dark_score = f64::MIN;
            if col > 1 {
                // dark here
                dark_score =
                    dp_matrix[[row, col - 1]] + prev_trans_probs.prob(TransState::Dark).ln();
                // if row == 1 {
                //     println!("col:{}, dark_score:{}, pre_score:{}, trans_score:{}", col, dark_score, dp_matrix[[row, col - 1]], prev_trans_probs.prob(TransState::Dark).ln());
                // }
            }

            let mut scores = vec![
                (branch_score, TransState::Branch),
                (stick_score, TransState::Stick),
                (dark_score, TransState::Dark),
                (match_score, TransState::Match),
            ];
            scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            // if scores[0].0 < (-1e307) {
            //     println!("row:{}-col:{}", row, col);
            //     break;
            // }
            dp_matrix[[row, col]] = scores[0].0;
            state_matrix[[row, col]] = scores[0].1;
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

    // backtrace
    let mut cur_col = dp_cols - 1;
    let mut cur_row = dp_rows - 1;
    let mut aligned_seq = vec![];
    let mut aligned_ref = vec![];
    while cur_col > 0 && cur_row > 0 {
        let cur_state = state_matrix[[cur_row, cur_col]];
        let ref_base = template[cur_col - 1].base() as char;
        let read_base = decode_emit_base(encoded_query[cur_row - 1]).as_bytes()[1] as char;
        match cur_state {
            TransState::Match => {
                aligned_seq.push(read_base);
                aligned_ref.push(ref_base);
                cur_col -= 1;
                cur_row -= 1;
            }
            TransState::Branch | TransState::Stick => {
                aligned_seq.push(read_base);
                aligned_ref.push('-');
                cur_row -= 1;
            }

            TransState::Dark => {
                aligned_seq.push('-');
                aligned_ref.push(ref_base);
                cur_col -= 1;
            }
        }
    }

    let eq = aligned_ref
        .iter()
        .zip(aligned_seq.iter())
        .map(|(&v1, &v2)| if v1 == v2 { 1 } else { 0 })
        .sum::<u32>();
    let identity = (eq as f64) / (aligned_ref.len() as f64);

    format!(
        "identity={}\n{}\n{}",
        identity,
        aligned_ref
            .into_iter()
            .rev()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(""),
        aligned_seq
            .into_iter()
            .rev()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(""),
    )
}

#[cfg(test)]
mod test {

    use crate::{
        dataset::encode_emit,
        em_training::{
            fb::{backward_with_log_sum_exp_trick, forward_with_log_sum_exp_trick, veterbi_decode},
            model::Template,
        },
        hmm_models::v1,
    };

    use super::{backward, forward};

    #[test]
    fn test_forward_backward() {
        let query_bases = vec![0, 1, 2, 3];
        let templates = vec![0, 1, 2, 3];

        let hmm_model = v1::get_hmm_model();

        let templates = Template::from_template_bases(&templates, &hmm_model);

        let alpha = forward(&query_bases, &templates, &hmm_model);
        let beta = backward(&query_bases, &templates, &hmm_model);
        println!("{:?}", alpha);
        println!("{:?}", beta);
    }

    #[test]
    fn test_forward_backward_2() {
        let hmm_model = v1::get_hmm_model();

        let query_bases = vec![0, 1, 2, 3];
        let templates = vec![0, 1, 2, 3];
        let templates = Template::from_template_bases(&templates, &hmm_model);

        let alpha = forward(&query_bases, &templates, &hmm_model);
        let beta = backward(&query_bases, &templates, &hmm_model);
        println!("{:?}", alpha);
        println!("{:?}", beta);

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
    fn test_fb_3() {
        let templates = b"CCGAACTAGTCTCAGGCTTCAACATCGAATACGCCGCAGGCCCCTTCGCCCTATTCTTCATAGCCGAATACACAAACATTATTATAATAAACACCCTCACCACTACAATCTTCCTAGGAACAACATATAACGCACTCTCCCCTGAACTCTACACAACATATTTTGTCACCAAGACCCTACTTCTGACCTCCCTGTTCTTATGAATTCGAACAGCATACCCCCGATTCCGC";
        let query = "GGGAGCCGAACTAGTCTCAGGCTTCACATCGAGTTTCCCGCAGCCTTCGCCCTATCTTCCATAGCCGAATACACAAACATATATAATAAACACCCTCACCACTACAATCTTCCTAGGAACAACATATGACGCACTCTCCCCTGAACTCCTACACAACATATTTTGTCACCAAGACCCTACTTCTAACCTCCCTGTTCTTATAATTCGAACAGCATCACCCCCGATTCCGC";
        let encoded_emit = encode_emit(&vec![0; 230], query);
        let hmm_model = v1::get_hmm_model();
        let templates = Template::from_template_bases(templates, &hmm_model);

        println!("{:?}", forward(&encoded_emit, &templates, &hmm_model));
        println!(
            "{:?}",
            forward_with_log_sum_exp_trick(&encoded_emit, &templates, &hmm_model)
        );
        println!("{:?}", backward(&encoded_emit, &templates, &hmm_model));
        println!(
            "{:?}",
            backward_with_log_sum_exp_trick(&encoded_emit, &templates, &hmm_model)
        );
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
