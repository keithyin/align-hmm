use std::ops::Deref;


use crate::{common::TransState, hmm_model::HmmModel};

pub fn encode_2_bases(prev: u8, cur: u8) -> u8 {
    let enc1 = gskits::dna::SEQ_NT4_TABLE[prev as usize];
    assert!(enc1 < 4, "invalid base, prev: {}", prev as char);

    let enc2 = gskits::dna::SEQ_NT4_TABLE[cur as usize];
    assert!(enc2 < 4, "invalid base, cur: {}", cur as char);

    (enc1 << 2) + enc2
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

    pub fn prob(&self, movement: TransState) -> f32 {
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
                let match_prob = model.ctx_move(enc, TransState::Match);
                let dark_prob = model.ctx_move(enc, TransState::Dark);
                let branch_prob = model.ctx_move(enc, TransState::Branch);
                let stick_prob = model.ctx_move(enc, TransState::Stick);
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

    #[allow(unused)]
    pub fn init_ctx(&self) -> u8 {
        encode_2_bases('A' as u8, self.0[0].base)
    }
    
    #[allow(unused)]
    pub fn ctx(&self, pos: usize) -> u8 {
        assert!(pos < self.0.len());
        encode_2_bases(self.0[pos].base, self.0[pos + 1].base)
    }
}

impl Deref for Template {
    type Target = Vec<TemplatePos>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod test {

    #[test]
    fn test_model() {}
}
