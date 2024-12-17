use std::collections::HashMap;

use crossbeam::channel;
use gskits::{
    fastx_reader::{fasta_reader::FastaFileReader, read_fastx},
    pbar::{get_spin_pb, DEFAULT_INTERVAL},
};
use rust_htslib::bam::{self, Read};

use crate::supervised_training::common::TrainInstance;

pub fn issue_train_instance(
    aligned_bam: &str,
    ref_fasta: &str,
    dw_boundaries: &Vec<u8>,
    sender: channel::Sender<TrainInstance>,
) {
    let fasta_rader = FastaFileReader::new(ref_fasta.to_string());
    let refname2refseq = read_fastx(fasta_rader)
        .into_iter()
        .map(|read_info| (read_info.name, read_info.seq))
        .collect::<HashMap<_, _>>();

    let mut aligned_bam_reader = bam::Reader::from_path(aligned_bam).unwrap();
    aligned_bam_reader.set_threads(10).unwrap();
    let aligned_bam_header = bam::Header::from_template(aligned_bam_reader.header());
    let aligned_bam_header_view = bam::HeaderView::from_header(&aligned_bam_header);

    let mut tid2refname = HashMap::new();

    let pbar = get_spin_pb(
        format!("training... using {}", aligned_bam),
        DEFAULT_INTERVAL,
    );

    for align_record in aligned_bam_reader.records() {
        pbar.inc(1);
        let align_record = align_record.unwrap();
        let tid = align_record.tid();
        if !tid2refname.contains_key(&tid) {
            tid2refname.insert(
                tid,
                String::from_utf8(aligned_bam_header_view.tid2name(tid as u32).to_vec()).unwrap(),
            );
        }
        let refname = tid2refname.get(&tid).unwrap();
        let refseq = refname2refseq.get(refname).unwrap();

        let train_instance = TrainInstance::from_aligned_record_and_ref_seq_and_pin_start_end(
            &align_record,
            refseq,
            dw_boundaries,
        );

        sender.send(train_instance).unwrap();
    }
}

pub fn issue_all_train_instance(
    aligned_bams: &Vec<String>,
    ref_fastas: &Vec<String>,
    dw_boundaries: &Vec<u8>,
    sender: channel::Sender<TrainInstance>,
) {
    aligned_bams
        .iter()
        .zip(ref_fastas.iter())
        .for_each(|(aligned_bam, ref_fasta)| {
            issue_train_instance(aligned_bam, ref_fasta, dw_boundaries, sender.clone())
        });
}

pub fn encode_emit(dw_buckets: &Vec<u8>, bases: &str) -> Vec<u8> {
    let bases = bases
        .as_bytes()
        .iter()
        .map(|v| *v as usize)
        .map(|v| gskits::dna::SEQ_NT4_TABLE[v])
        .collect::<Vec<u8>>();
    bases
        .into_iter()
        .zip(dw_buckets.iter())
        .map(|(base_enc, &bucket)| bucket << 2 + base_enc)
        .collect()
}
