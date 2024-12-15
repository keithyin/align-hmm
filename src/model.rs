use std::ops::Deref;

use ndarray::{Array2, Array3};

pub fn encode_2_bases(prev: u8, cur: u8) -> u8 {
    let mut enc = gskits::dna::SEQ_NT4_TABLE[prev as usize];
    enc <<= 2;
    enc += gskits::dna::SEQ_NT4_TABLE[cur as usize];
    enc
}

#[derive(Debug, Clone, Copy)]
pub enum Move {
    Match = 0,
    Dark = 1,   // del
    Branch = 2, // poly ins
    Stick = 3,  // non-poly ins
}

#[derive(Debug, Clone, Copy)]
pub struct TemplatePos {
    base: u8,
    probs: [f32; 4], // match dark, branch, stick
}

impl TemplatePos {
    pub fn new(base: u8, probs: [f32; 4]) -> Self {
        Self { base, probs }
    }

    pub fn base(&self) -> u8 {
        self.base
    }

    pub fn prob(&self, movement: Move) -> f32 {
        self.probs[movement as usize]
    }
}

impl Default for TemplatePos {
    fn default() -> Self {
        Self {
            base: 'A' as u8,
            probs: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

pub struct Template(Vec<TemplatePos>);

impl Template {
    pub fn from_template_bases(bases: &[u8], model: &HmmModel) -> Self {
        let mut prev_base = bases[0];

        let mut template = bases
            .iter()
            .skip(1)
            .into_iter()
            .map(|&cur_base| {
                let mut enc = gskits::dna::SEQ_NT4_TABLE[prev_base as usize];
                enc <<= 2;
                enc += gskits::dna::SEQ_NT4_TABLE[cur_base as usize];
                let match_prob = model.ctx_move(enc, Move::Match);
                let dark_prob = model.ctx_move(enc, Move::Dark);
                let branch_prob = model.ctx_move(enc, Move::Branch);
                let stick_prob = model.ctx_move(enc, Move::Stick);
                let tpl_pos =
                    TemplatePos::new(prev_base, [match_prob, dark_prob, branch_prob, stick_prob]);
                prev_base = cur_base;
                tpl_pos
            })
            .collect::<Vec<_>>();

        template.push(TemplatePos::new(
            *bases.last().unwrap(),
            [1.0, 0.0, 0.0, 0.0],
        ));

        Self(template)
    }
}

impl Deref for Template {
    type Target = Vec<TemplatePos>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug)]
pub struct HmmModel {
    ctx_move_prob: Array2<f32>,
    move_ctx_emit_prob: Array3<f32>,
}

impl HmmModel {
    pub fn new(emit_num: usize) -> Self {
        let ctx_move_prob = Array2::from_shape_fn((16, 4), |_| 1.0 / 4.0);

        let move_ctx_emit_prob =
            Array3::from_shape_fn((4, 16, emit_num), |_| 1.0 / emit_num as f32);
        Self {
            ctx_move_prob,
            move_ctx_emit_prob,
        }
    }

    pub fn from_params(ctx_move_prob: Array2<f32>, move_ctx_emit_prob: Array3<f32>) -> Self {
        Self {
            ctx_move_prob,
            move_ctx_emit_prob,
        }
    }

    pub fn emit_prob(&self, movement: Move, ctx: u8, emit: u8) -> f32 {
        self.move_ctx_emit_prob[[movement as usize, ctx as usize, emit as usize]]
    }

    pub fn ctx_move(&self, ctx: u8, movement: Move) -> f32 {
        self.ctx_move_prob[[ctx as usize, movement as usize]]
    }
}

#[cfg(test)]
mod test {
    use super::HmmModel;

    #[test]
    fn test_model() {
        let model = HmmModel::new(12);
        println!("{:?}", model);
    }
}
