use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crossbeam::channel;
use gskits::{
    fastx_reader::{fasta_reader::FastaFileReader, read_fastx},
    gsbam::bam_record_ext::BamRecord,
};
use indicatif::ProgressBar;
use rust_htslib::bam::{self, Read};

use crate::common::TrainInstance;

pub fn align_record_read_worker(
    aligned_bam: &str,
    ref_fasta: &str,
    pbar: Arc<Mutex<ProgressBar>>,
    sender: channel::Sender<(BamRecord, Arc<String>)>,
) {
    let fasta_rader = FastaFileReader::new(ref_fasta.to_string());
    let refname2refseq = read_fastx(fasta_rader)
        .into_iter()
        .map(|read_info| (read_info.name, Arc::new(read_info.seq)))
        .collect::<HashMap<_, _>>();

    let mut aligned_bam_reader = bam::Reader::from_path(aligned_bam).unwrap();
    aligned_bam_reader.set_threads(4).unwrap();
    let aligned_bam_header = bam::Header::from_template(aligned_bam_reader.header());
    let aligned_bam_header_view = bam::HeaderView::from_header(&aligned_bam_header);

    let mut tid2refname = HashMap::new();

    let mut align_record = BamRecord::new();
    let mut cnt = 0;
    while let Some(v) = aligned_bam_reader.read(&mut align_record) {
        pbar.lock().unwrap().inc(1);
        cnt += 1;
        if cnt > 10000 {
            break;
        }
        if v.is_ok() {
            let tid = align_record.tid();
            if !tid2refname.contains_key(&tid) {
                tid2refname.insert(
                    tid,
                    String::from_utf8(aligned_bam_header_view.tid2name(tid as u32).to_vec())
                        .unwrap(),
                );
            }
            let refname = tid2refname.get(&tid).unwrap();
            let refseq = refname2refseq.get(refname).unwrap();

            sender.send((align_record, refseq.clone())).unwrap();
            align_record = BamRecord::new();
        }
    }
    tracing::info!("records in {} are all issued. ", aligned_bam);
}

pub fn train_instance_worker(
    dw_boundaries: &Vec<u8>,
    receiver: channel::Receiver<(BamRecord, Arc<String>)>,
    sender: channel::Sender<TrainInstance>,
) {
    for (align_record, ref_seq) in receiver {
        let train_instance = TrainInstance::from_aligned_record_and_ref_seq_and_pin_start_end(
            &align_record,
            &ref_seq,
            dw_boundaries,
        );
        if let Some(train_instance) = train_instance {
            sender.send(train_instance).unwrap();
        }
    }
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
        .map(|(base_enc, &bucket)| (bucket << 2) + base_enc)
        .collect()
}

#[cfg(test)]
mod test {
    use crate::em_training::model::decode_emit_base;

    use super::encode_emit;

    #[test]
    fn test_encode_emit() {
        let dw_buckets = vec![0, 1, 2, 0, 1, 2];
        let bases = "ACCGAT";
        let encoded = encode_emit(&dw_buckets, bases);
        let res_str = encoded
            .into_iter()
            .map(|enc| decode_emit_base(enc))
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(res_str, "0A1C2C0G1A2T");
    }
}
