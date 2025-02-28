use std::{
    collections::{HashMap, HashSet}, fs, io::BufReader, path, process
};

use anyhow::anyhow;
use gskits::{
    file_reader::{bed_reader::BedInfo, vcf_reader::VcfInfo},
    pbar::{get_spin_pb, DEFAULT_INTERVAL},
};
use rust_htslib::bam::{self, ext::BamRecordExtensions, Read};

use crate::cli::TrainDataParams;

pub fn train_data_main(params: &TrainDataParams) -> Option<()> {
    let subreads_bam = &params.sbr_bam;
    let ref_fa = &params.ref_fa;

    // variant
    let vcf_file = params.vcf_file.as_ref().and_then(|v| Some(v.as_str()));
    // confidence region
    let bed_file = params.bed_file.as_ref().and_then(|v| Some(v.as_str()));
    let out_dir = params.get_out_dir();

    // 1. do alignment
    // let align_res_bam = alignment(subreads_bam, ref_fa, None).unwrap();
    let align_res_bam = gsmm2_alignment(subreads_bam, ref_fa, &out_dir, None).unwrap();

    // 2. get whitelist according to vcf, bed, and single mapping
    let whitelist = query_name_whitelist(&align_res_bam, vcf_file, bed_file);
    tracing::info!("remaining records: {}", whitelist.len());

    // 3. dump
    let res_bam = dump_filtered_bam(&align_res_bam, &whitelist);

    tracing::info!(?res_bam);

    Some(())
}

#[tracing::instrument(skip(subreads_bam, ref_fa, threads))]
fn gsmm2_alignment(
    subreads_bam: &str,
    ref_fa: &str,
    out_dir: &str,
    threads: Option<usize>,
) -> anyhow::Result<String> {
    tracing::info!("do alignment");
    let threads = threads.unwrap_or(num_cpus::get());

    let sbr_path = path::Path::new(subreads_bam);

    let o_file_prefix = format!("{}/{}.align4arrow", out_dir, sbr_path.file_stem().map(|v| v.to_string_lossy().into_owned()).unwrap());
    let o_filepath = format!("{o_file_prefix}.bam");
    let mut cmd = process::Command::new("gsmm2");
    cmd.args([
        "--threads",
        threads.to_string().as_str(),
        "--preset",
        "map-ont",
        "align",
        "-q",
        subreads_bam,
        "-t",
        ref_fa,
        "-p",
        o_file_prefix.as_str(),
        "--noMar",
        "--oupIyT", "0.8",
        "--oupCovT", "0.7"
    ]);

    let status = cmd.status()?;
    if !status.success() {
        return Err(anyhow!("run alignment command error"));
    }

    Ok(o_filepath)
}

#[tracing::instrument(skip(align_res_bam, vcf_file, bed_file))]
fn query_name_whitelist(
    align_res_bam: &str,
    vcf_file: Option<&str>,
    bed_file: Option<&str>,
) -> HashSet<String> {
    let mut qname_cnt = HashMap::new();

    let mut whitelist = HashSet::new();
    let mut bam_reader = bam::Reader::from_path(align_res_bam).unwrap();
    bam_reader.set_threads(num_cpus::get()).unwrap();

    let header = bam::Header::from_template(bam_reader.header());
    let head_view = bam::HeaderView::from_header(&header);

    let mut tid2refname = HashMap::new();

    let vcf_info = vcf_file.and_then(|f_path| {
        tracing::info!("processing vcf file");
        Some(VcfInfo::new(&mut BufReader::new(
            fs::File::open(f_path).unwrap(),
        )))
    });
    let bed_info = bed_file.and_then({
        tracing::info!("processing bed file");
        |f_path| {
            Some(BedInfo::new(&mut BufReader::new(
                fs::File::open(f_path).unwrap(),
            )))
        }
    });

    let pbar = get_spin_pb("generate whitelist".to_string(), DEFAULT_INTERVAL);
    for record in bam_reader.records() {
        pbar.inc(1);
        let record = record.unwrap();
        let qname = String::from_utf8(record.qname().to_vec()).unwrap();
        let tid = record.tid();

        if !tid2refname.contains_key(&tid) {
            tid2refname.insert(
                tid,
                String::from_utf8(head_view.tid2name(tid as u32).to_vec()).unwrap(),
            );
        }

        let ref_name = tid2refname.get(&tid).unwrap();

        let ref_start = record.reference_start() as usize;
        let ref_end = record.reference_end() as usize;

        let mut pass = true;
        if let Some(ref vcf_info_) = vcf_info {
            if pass && vcf_info_.range_hit(ref_name, &(ref_start, ref_end)) {
                pass = false;
            }
        }

        if let Some(ref bed_info_) = bed_info {
            if pass && !bed_info_.within_the_range(ref_name, &(ref_start, ref_end)) {
                pass = false
            }
        }

        if pass {
            *qname_cnt.entry(qname.clone()).or_insert(0) += 1;
            whitelist.insert(qname);
        } else {
            *qname_cnt.entry(qname).or_insert(0) += 1;
        }
    }
    pbar.finish();

    tracing::info!("tot_queries:{}", qname_cnt.len());

    qname_cnt
        .into_iter()
        .filter(|(k, v)| *v == 1 && whitelist.contains(k))
        .map(|(k, _)| k)
        .collect::<HashSet<String>>()
}

fn dump_filtered_bam(align_res_bam: &str, query_whitelist: &HashSet<String>) -> String {
    let res_bam = format!(
        "{}.align_hmm_train_data.bam",
        align_res_bam.rsplit_once(".").unwrap().0
    );

    let mut bam_reader = bam::Reader::from_path(align_res_bam).unwrap();
    bam_reader.set_threads(num_cpus::get() / 2).unwrap();

    let header = bam::Header::from_template(bam_reader.header());
    let mut bam_writer = bam::Writer::from_path(&res_bam, &header, bam::Format::Bam).unwrap();
    bam_writer.set_threads(num_cpus::get() / 2).unwrap();

    let pbar = get_spin_pb("dumping result bam".to_string(), DEFAULT_INTERVAL);
    for record in bam_reader.records() {
        pbar.inc(1);
        let record = record.unwrap();

        let qname = String::from_utf8(record.qname().to_vec()).unwrap();
        if query_whitelist.contains(&qname) {
            bam_writer.write(&record).unwrap();
        }
    }
    pbar.finish();

    res_bam
}

// #[tracing::instrument(skip(subreads_bam, ref_fa, threads))]
// fn alignment(subreads_bam: &str, ref_fa: &str, threads: Option<usize>) -> anyhow::Result<String> {
//     tracing::info!("do alignment");
//     let threads = threads.unwrap_or(num_cpus::get());
//     let o_file_prefix = format!("{}.align4arrow", subreads_bam.rsplit_once(".").unwrap().0);
//     let o_filepath = format!("{o_file_prefix}.bam");
//     let mut cmd = process::Command::new("gsmm2-rs");
//     cmd.args([
//         "-q",
//         subreads_bam,
//         "-r",
//         ref_fa,
//         "-p",
//         o_file_prefix.as_str(),
//         "--aligner",
//         "minimap2",
//         "-t",
//         threads.to_string().as_str(),
//         "--forceIndex",
//         // "--num_querys_per_fa", "4000000",
//         "--kmer",
//         "15",
//         "--wins",
//         "10",
//         "--s_m",
//         "2",
//         "--s_mm",
//         "4",
//         "--s_go",
//         "4,24",
//         "--s_ge",
//         "2,1",
//         "--force_alignment",
//         "--no_extra_info",
//     ]);

//     let status = cmd.status()?;
//     if !status.success() {
//         return Err(anyhow!("run alignment command error"));
//     }

//     Ok(o_filepath)
// }
