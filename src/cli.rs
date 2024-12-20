use std::{fs, path, str::FromStr};

use clap::{self, Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub commands: Subcommands,
}

#[derive(Debug, Subcommand, Clone)]
pub enum Subcommands {
    Eda(TrainingParams),
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

    #[arg(long="outdir")]
    pub out_dir: Option<String>,

    #[arg(long = "vcf-file")]
    pub vcf_file: Option<String>,

    #[arg(long = "bed-file")]
    pub bed_file: Option<String>,
}

impl TrainDataParams {
    pub fn get_out_dir(&self) -> String {
        let out = if let Some(out_dir) = &self.out_dir {
            out_dir.to_string()
        } else {
            let sbr_path = path::Path::new(&self.sbr_bam);
            sbr_path.parent().map(|v| v.to_string_lossy().into_owned()).unwrap()
        };

        if !path::Path::new(&out).exists() {
            fs::create_dir_all(&out).expect(&format!("create dir error: {}", &out));
        }

        out

    }
}