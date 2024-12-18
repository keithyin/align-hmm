use clap::Parser;
use cli::Subcommands;
use em_training::em_training;
use supervised_training::{train::train_model_entrance, train_data::train_data_main};

mod cli;
mod em_training;
mod supervised_training;
mod train_instance;

fn main() {
    let time_fmt = time::format_description::parse(
        "[year]-[month padding:zero]-[day padding:zero] [hour]:[minute]:[second]",
    )
    .unwrap();

    // let timer = tracing_subscriber::fmt::time::OffsetTime::new(time_offset, time_fmt);
    // let timer = tracing_subscriber::fmt::time::LocalTime::new(time_fmt);
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
            train_model_entrance(&training_param);
        }

        Subcommands::EmTraining(trainig_param) => {
            em_training(&trainig_param);
        }
    }
}
