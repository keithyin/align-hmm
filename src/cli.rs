use std::str::FromStr;

use clap::{self, Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub commands: Subcommands,
}

#[derive(Debug, Subcommand, Clone)]
pub enum Subcommands {
    SupervisedTraining(TrainingParams),
    EmTraining(TrainingParams),
    TrainData(TrainDataParams),
}

#[derive(Debug, Args, Clone)]
pub struct TrainingParams {
    #[arg(long = "aligned-bams")]
    pub aligned_bams: Vec<String>,

    #[arg(long = "ref-fas", help = "fasta")]
    pub ref_fas: Vec<String>,

    #[arg(long="dw_boundaries", default_value_t=String::from_str("18,46").unwrap()   )]
    pub dw_boundaries: String,
}

impl TrainingParams {
    pub fn parse_dw_boundaries(&self) -> Vec<u8> {
        self.dw_boundaries
            .split(",")
            .map(|v| v.trim())
            .map(|v| v.parse::<u8>().unwrap())
            .collect::<Vec<u8>>()
    }
}

#[derive(Debug, Args, Clone)]
pub struct TrainDataParams {
    #[arg(long = "sbr-bam")]
    pub sbr_bam: String,

    #[arg(long = "ref-fa")]
    pub ref_fa: String,

    #[arg(long = "vcf-file")]
    pub vcf_file: Option<String>,

    #[arg(long = "bed-file")]
    pub bed_file: Option<String>,
}
