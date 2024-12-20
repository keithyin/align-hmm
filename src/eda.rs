use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    thread,
};

use crossbeam::channel::{self, Receiver};
use gskits::pbar::{get_spin_pb, DEFAULT_INTERVAL};
use ndarray::Array1;

use crate::{
    cli::TrainingParams,
    common::TrainInstance,
    dataset::{align_record_read_worker, train_instance_worker},
    supervised_training::build_train_events_for_stat,
};

pub fn eda_entrance_parallel(params: &TrainingParams) {
    let aligned_bams = &params.aligned_bams;
    let ref_fastas = &params.ref_fas;
    assert!(aligned_bams.len() == ref_fastas.len());

    let pbar = get_spin_pb(format!("training..."), DEFAULT_INTERVAL);

    let pbar = Arc::new(Mutex::new(pbar));

    thread::scope(|s| {
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
        for _ in 0..(num_cpus::get() / 2) {
            let record_receiver_ = record_receiver.clone();
            let train_ins_sender_ = train_ins_sender.clone();
            s.spawn(move || train_instance_worker(None, record_receiver_, train_ins_sender_));
        }
        drop(record_receiver);
        drop(train_ins_sender);
        let mut counter_parts = vec![];
        for _ in 0..(num_cpus::get() / 2) {
            let train_ins_recv_ = train_ins_receiver.clone();
            counter_parts.push(s.spawn(move || counter_worker(train_ins_recv_)));
        }

        let counter_parts = counter_parts
            .into_iter()
            .map(|hmm_model| hmm_model.join().unwrap())
            .collect::<Vec<_>>();

        let mut final_counter = HashMap::new();
        counter_parts.into_iter().for_each(|counter| {
            counter.into_iter().for_each(|(key, v)| {
                *final_counter
                    .entry(key)
                    .or_insert(Array1::<usize>::from_elem((256,), 0)) += &v;
            });
        });
        print_counter(final_counter);
    });
}

fn counter_worker(recv: Receiver<TrainInstance>) -> HashMap<String, Array1<usize>> {
    let mut counter = HashMap::new();

    for instance in recv {
        let curcount = build_train_events_for_stat(&instance);
        curcount.into_iter().for_each(|(key, v)| {
            *counter
                .entry(key)
                .or_insert(Array1::<usize>::from_elem((256,), 0)) += &v;
        });
    }
    counter
}

fn print_counter(counter: HashMap<String, Array1<usize>>) {
    let mut counter = counter.into_iter().collect::<Vec<_>>();
    counter.sort_by_key(|v| v.0.clone());

    for (k, counts) in counter {
        println!(
            "key:{}:{}",
            k,
            counts
                .to_vec()
                .into_iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
    }
}
