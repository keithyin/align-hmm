mod cli;
mod em_training;
mod supervised_training;
mod dataset;
mod hmm_model;
mod common;
mod train_data;
mod hmm_model_params;

use clap::Parser;
use cli::Subcommands;
use em_training::em_training;
use hmm_model_params::v1;
use supervised_training::train_model_entrance_parallel;
use train_data::train_data_main;


fn main() {
    let time_fmt = time::format_description::parse(
        "[year]-[month padding:zero]-[day padding:zero] [hour]:[minute]:[second]",
    )
    .unwrap();

    let time_offset =
        time::UtcOffset::current_local_offset().unwrap_or_else(|_| time::UtcOffset::UTC);
    let timer = tracing_subscriber::fmt::time::OffsetTime::new(time_offset, time_fmt);

    tracing_subscriber::fmt::fmt().with_timer(timer).init();

    let params = cli::Cli::parse();
    match params.commands {
        Subcommands::TrainData(train_data_param) => {
            train_data_main(&train_data_param);
        }

        Subcommands::SupervisedTraining(training_param) => {
            // train_model_entrance(&training_param);
            train_model_entrance_parallel(&training_param);
        }

        Subcommands::EmTraining(trainig_param) => {
            // tracing::info!("init hmm model");
            // let init_hmm_model = train_model_entrance_parallel(&trainig_param);
            let init_hmm_model = v1::get_hmm_model();
            tracing::info!("init hmm model done, start em training");
            em_training(&trainig_param, init_hmm_model);
        }
    }
}
