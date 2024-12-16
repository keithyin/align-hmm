use ndarray::Array2;

use crate::model::{encode_2_bases, HmmModel, Move, Template, TemplatePos};

/// forward backward

/// 2D DP, 假设 template 是 ATC, called 是 ATC，那么会构建如下DP
///  ' AT TC C-
/// '  
/// A  *
/// T
/// C        *
/// 将坐标看作状态，初始情况下 在 a_11 的概率为 1， 所以可以理解为 初始状态 a_11 的概率是 1.
///  因为已经pin了开始和结束，所以 到第一个 baes 的转移，和 到最后一个 base 的转移，都应该是 match！！！
pub fn forward(encoded_query: &[u8], template: &Template, hmm_model: &HmmModel) -> Array2<f32> {
    let dp_rows = encoded_query.len() + 1;
    let dp_cols = template.len() + 1;

    let mut dp_matrix: Array2<f32> = Array2::zeros((dp_rows, dp_cols));
    dp_matrix[[0, 0]] = 1.0;

    let mut prev_trans_probs = TemplatePos::default();
    let prev_tpl_base = prev_trans_probs.base();

    for col in 1..(dp_cols - 1) {
        let cur_trans_probs = template[col - 1];
        let cur_tpl_base = cur_trans_probs.base();
        for row in 1..(dp_rows - 1) {
            let mut score = 0.0;
            let cur_read_base_enc = encoded_query[row - 1];
            if row > 0 && col > 0 {
                score = dp_matrix[[row - 1, col - 1]]
                    * prev_trans_probs.prob(Move::Match)
                    * hmm_model.emit_prob(
                        Move::Match,
                        encode_2_bases(prev_tpl_base, cur_tpl_base),
                        cur_read_base_enc,
                    );
            }

            if row > 1 {
                // ins here

                let next_trans_probs = template[col];
                let next_tpl_base = next_trans_probs.base();
                let this_move_score = dp_matrix[[row - 1, col]]
                    * cur_trans_probs.prob(Move::Branch)
                    * hmm_model.emit_prob(
                        Move::Branch,
                        encode_2_bases(cur_tpl_base, next_tpl_base),
                        cur_read_base_enc,
                    );

                score += this_move_score;
                let this_move_score = dp_matrix[[row - 1, col]]
                    * cur_trans_probs.prob(Move::Stick)
                    * hmm_model.emit_prob(
                        Move::Stick,
                        encode_2_bases(cur_tpl_base, next_tpl_base),
                        cur_read_base_enc,
                    );

                score += this_move_score;
            }

            if col > 1 {
                // dark here
                score += dp_matrix[[row, col - 1]] * prev_trans_probs.prob(Move::Dark);
            }

            dp_matrix[[row, col]] = score;
        }

        prev_trans_probs = cur_trans_probs;
    }

    let cur_tpl_base = template.last().unwrap().base();
    dp_matrix[[dp_rows - 1, dp_cols - 1]] = dp_matrix[[dp_rows - 2, dp_cols - 2]]
        * hmm_model.emit_prob(
            Move::Match,
            encode_2_bases(prev_tpl_base, cur_tpl_base),
            *encoded_query.last().unwrap(),
        );

    dp_matrix
}

pub fn backward(encoded_query: &[u8], template: &Template, hmm_model: &HmmModel) -> Array2<f32> {
    let dp_rows = encoded_query.len() + 1;
    let dp_cols = template.len() + 1;

    let mut dp_matrix: Array2<f32> = Array2::zeros((dp_rows, dp_cols));
    dp_matrix[[dp_rows - 1, dp_cols - 1]] = 1.0;

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
                let this_move_score = hmm_model.emit_prob(
                    Move::Match,
                    encode_2_bases(cur_tpl_base, next_tpl_base),
                    next_read_base_enc,
                ) * dp_matrix[[row + 1, col + 1]];

                score += this_move_score;
            } else {
                let this_move_score = cur_trans_probs.prob(Move::Match)
                    * hmm_model.emit_prob(
                        Move::Match,
                        encode_2_bases(cur_tpl_base, next_tpl_base),
                        next_read_base_enc,
                    )
                    * dp_matrix[[row + 1, col + 1]];

                score += this_move_score;
            }

            if row < (dp_rows - 2) {
                // ins here
                let this_move_score = cur_trans_probs.prob(Move::Branch)
                    * hmm_model.emit_prob(
                        Move::Branch,
                        encode_2_bases(cur_tpl_base, next_tpl_base),
                        next_read_base_enc,
                    )
                    * dp_matrix[[row + 1, col]];

                score += this_move_score;
                let this_move_score = cur_trans_probs.prob(Move::Stick)
                    * hmm_model.emit_prob(
                        Move::Stick,
                        encode_2_bases(cur_tpl_base, next_tpl_base),
                        next_read_base_enc,
                    )
                    * dp_matrix[[row + 1, col]];

                score += this_move_score;
            }

            if col < (dp_cols - 2) {
                // dark here
                score += cur_trans_probs.prob(Move::Dark) * dp_matrix[[row, col + 1]];
            }

            dp_matrix[[row, col]] = score;
        }
    }
    let default_tpl_pos = TemplatePos::default();
    let default_tpl_base = default_tpl_pos.base();
    dp_matrix[[0, 0]] = hmm_model.emit_prob(
        Move::Match,
        encode_2_bases(default_tpl_base, template.first().unwrap().base()),
        *encoded_query.last().unwrap(),
    ) * dp_matrix[[1, 1]];

    dp_matrix
}

#[cfg(test)]
mod test {
    use crate::model::{HmmModel, Template};

    use super::{backward, forward};

    #[test]
    fn test_forward_backward() {
        let query_bases = vec![0, 1, 2, 3];
        let templates = vec![0, 1, 2, 3];

        let hmm_model = HmmModel::new(4);
        let templates = Template::from_template_bases(&templates, &hmm_model);

        let alpha = forward(&query_bases, &templates, &hmm_model);
        let beta = backward(&query_bases, &templates, &hmm_model);
        println!("{:?}", alpha);
        println!("{:?}", beta);
    }
}
