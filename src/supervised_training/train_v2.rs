// /// query 使用call 出来的方向，reverse 参考序列


// use core::str;
// use std::{collections::HashMap, io::{BufWriter, Write}, thread};

// use clap::ArgMatches;
// use rust_htslib::{bam::{self, Read}, faidx};

// use crate::{pb_tools::{self, get_spin_pb, DEFAULT_INTERVAL},
//     samtools_facade::samtools_fai, utils::{self, dna::reverse_complement}};
// use ndarray::{Array2, Array3, Axis, s};
// use crate::arrow_hmm::common::*;

// use bio;


// fn state_identification_v2(align_op: bio::alignment::AlignmentOperation, ref_seq: &[u8], query_seq: &[u8], ref_pos: usize, query_pos: usize) -> ArrowState {

//     return match align_op {
//         bio::alignment::AlignmentOperation::Match | bio::alignment::AlignmentOperation::Subst => ArrowState::Match,
//         bio::alignment::AlignmentOperation::Del => ArrowState::Dark,
//         bio::alignment::AlignmentOperation::Ins => {
//             if query_seq[query_pos] == ref_seq[ref_pos] || query_seq[query_pos] == ref_seq[ref_pos + 1] {
//                 ArrowState::Branch
//             } else {
//                 ArrowState::Stick
//             }
//         },

//         _ => panic!("invalid align_op"),
//     }
// }

// struct TrainInstance {
//     query_aligned_seg: Vec<u8>,
//     query_aligned_dw_feats: Vec<u8>,
//     ref_aligned_seg: Vec<u8>,
//     align_ops: Option<Vec<bio::alignment::AlignmentOperation>>,
//     qname: String,
//     is_rev: bool
// }

// impl TrainInstance {
//     fn build_train_instance(aligned_record: &bam::Record, subread_record: &SubreadRecord, ref_str: &str, ref_str_rev_comp: &str) -> Self {
//         let ref_str = ref_str.as_bytes();
//         let ref_str_rev_comp = ref_str_rev_comp.as_bytes();
//         let is_rev = aligned_record.is_reverse();
//         let align_pairs = pin_start_end(aligned_record);

//         let ori_query_start = if is_rev {
//             aligned_record.seq_len() - (align_pairs.last().unwrap()[0].unwrap() as usize + 1)
//         } else {
//             align_pairs[0][0].unwrap() as usize
//         };

//         let ori_query_end = if is_rev {
//             aligned_record.seq_len() - align_pairs[0][0].unwrap() as usize
//         } else {
//             align_pairs.last().unwrap()[0].unwrap() as usize + 1
//         };

//         let ref_start = if is_rev {
//             ref_str.len() - (align_pairs.last().unwrap()[1].unwrap() as usize + 1)
//         } else {
//             align_pairs[0][1].unwrap() as usize

//         };

//         let ref_end = if is_rev {
//             ref_str.len() -  align_pairs[0][1].unwrap() as usize
//         } else {
//             align_pairs.last().unwrap()[1].unwrap() as usize + 1
//         };

//         let ref_aligned_seg = if is_rev {
//             &ref_str_rev_comp[ref_start..ref_end]
//         } else {
//             &ref_str[ref_start..ref_end]
//         };

//         let query_aligned_seg = subread_record.get_seq_range(ori_query_start, ori_query_end);
//         let query_dw_feats = subread_record.get_dw_range(ori_query_start, ori_query_end);

//         Self { 
//             query_aligned_seg: query_aligned_seg.to_vec(), 
//             query_aligned_dw_feats: query_dw_feats.to_vec(), 
//             ref_aligned_seg: ref_aligned_seg.to_vec(), 
//             align_ops: None,
//             qname: String::from_utf8(aligned_record.qname().to_vec()).unwrap(),
//             is_rev: aligned_record.is_reverse()
//         }

//     }

//     fn fill_align_ops(&mut self, left_align_indel_version: Option<i32>) {
//         let left_align_indel_version = left_align_indel_version.unwrap_or(0);
//         let scoring = bio::alignment::pairwise::Scoring::from_scores(-4, -2, 2, -4);
//         let mut aligner = bio::alignment::pairwise::Aligner::with_capacity_and_scoring(
//             self.query_aligned_seg.len(), self.ref_aligned_seg.len(), scoring);
//         let alignment = aligner.custom(&self.query_aligned_seg, &self.ref_aligned_seg);
//         let mut align_ops = alignment.operations;
//         match left_align_indel_version {
//             0 => {},
//             1 => left_align_indel_v1(&self.query_aligned_seg, &self.ref_aligned_seg, &mut align_ops),
//             2 => left_align_indel_v2(&self.query_aligned_seg, &self.ref_aligned_seg, &mut align_ops),
//             _ => panic!("invalid version, {}", left_align_indel_version),
//         }
//         self.align_ops = Some(align_ops);
//     }
// }


// struct ArrowHmm {
//     num_state: usize,
//     num_emit_state: usize,
//     num_ctx: usize,
//     num_emit: usize,

//     ctx_trans_cnt: Array2<usize>,
//     emission_cnt: Array3<usize>,

//     ctx_trans_prob: Array2<f32>,
//     emission_prob: Array3<f32>,

//     dw_buckets: Vec<u8>
// }

// impl ArrowHmm {
//     fn new(dw_buckets: Vec<u8>) -> Self{
//         let num_state = 4_usize;
//         let num_emit_state = 3_usize;
//         let num_ctx = 16_usize;
//         let num_emit = 4 * 4 * 4;
//         ArrowHmm { 
//             num_state,
//             num_emit_state,
//             num_ctx,
//             num_emit,
//             ctx_trans_cnt: Array2::<usize>::zeros((num_ctx, num_state)),  // zeros: no laplace smooth, ones: laplace smooth
//             emission_cnt: Array3::<usize>::ones((num_emit_state, num_ctx, num_emit)) ,
//             ctx_trans_prob: Array2::<f32>::zeros((num_ctx, num_state)), 
//             emission_prob: Array3::<f32>::zeros((num_emit_state, num_ctx, num_emit)),
//             dw_buckets
//         }

//     }

//     fn update_with_instance(&mut self, instance: &TrainInstance) {
//         let mut query_cursor: i32 = -1;
//         let mut ref_cursor: i32 = -1;

//         for op in instance.align_ops.as_ref().unwrap() {
//             match op {
//                 bio::alignment::AlignmentOperation::Match | bio::alignment::AlignmentOperation::Subst => {
//                     query_cursor += 1;
//                     ref_cursor += 1;
//                 },
//                 bio::alignment::AlignmentOperation::Ins => {
//                     query_cursor += 1;
//                 },
//                 bio::alignment::AlignmentOperation::Del =>  {
//                     ref_cursor += 1;
//                 },
//                 _ => panic!("invalid op"),
//             };

//             if ref_cursor <= 0 {
//                 continue;
//             }

//             let arrow_state = state_identification_v2(*op, &instance.ref_aligned_seg, &instance.query_aligned_seg, ref_cursor as usize, query_cursor as usize);

//             match arrow_state {
//                 ArrowState::Match => {
                
//                     let pre_base = instance.ref_aligned_seg[ref_cursor as usize - 1];
//                     let cur_base = instance.ref_aligned_seg[ref_cursor as usize];
//                     let ctx = encode_ctx(pre_base, cur_base);
//                     let emit = self.encode_emit_feat(instance.query_aligned_dw_feats[query_cursor as usize], instance.query_aligned_seg[query_cursor as usize]);
                    
//                     self.ctx_trans_cnt[[ctx, arrow_state as usize]] += 1;
//                     self.emission_cnt[[arrow_state as usize, ctx, emit]] += 1;


//                 },
//                 ArrowState::Dark => {
//                     let pre_base = instance.ref_aligned_seg[ref_cursor as usize - 1];
//                     let cur_base = instance.ref_aligned_seg[ref_cursor as usize];
//                     let ctx = encode_ctx(pre_base, cur_base);
//                     self.ctx_trans_cnt[[ctx, arrow_state as usize]] += 1;

//                 },
//                 ArrowState::Stick | ArrowState::Branch => {
//                     let cur_base = instance.ref_aligned_seg[ref_cursor as usize];
//                     let next_base = *instance.ref_aligned_seg.get(ref_cursor as usize + 1).expect(
//                         &format!("query:{}\nref:{}\n{}\nrev:{}", 
//                             String::from_utf8(instance.query_aligned_seg.to_vec()).unwrap(), 
//                             String::from_utf8(instance.ref_aligned_seg.to_vec()).unwrap(),
//                             instance.qname,
//                             instance.is_rev
//                         ));
//                     let ctx = encode_ctx(cur_base, next_base);
//                     let emit = self.encode_emit_feat(instance.query_aligned_dw_feats[query_cursor as usize], instance.query_aligned_seg[query_cursor as usize]);
                    
//                     self.ctx_trans_cnt[[ctx, arrow_state as usize]] += 1;
//                     self.emission_cnt[[arrow_state as usize, ctx, emit]] += 1;
//                 }
//             };
            
//         }
//     }

//     fn finish(&mut self) {
//         let ctx_aggr = self.ctx_trans_cnt.sum_axis(Axis(1));
//         let emission_aggr = self.emission_cnt.sum_axis(Axis(2));

//         for i in 0..self.num_ctx {
//             if ctx_aggr[[i]] == 0 {
//                 continue;
//             }

//             for j in 0..self.num_state {
//                 self.ctx_trans_prob[[i, j]] = (self.ctx_trans_cnt[[i, j]] as f32) / (ctx_aggr[[i]] as f32);
//             }
//         }

//         for i in 0..self.num_emit_state {
//             for j in 0..self.num_ctx {
//                 if emission_aggr[[i, j]] == 0 {
//                     continue;
//                 }
//                 for k in 0..self.num_emit {
//                     self.emission_prob[[i, j, k]] = (self.emission_cnt[[i, j, k]] as f32) / (emission_aggr[[i, j]] as f32);
//                 }
//             }
//         }

//     }

//     fn encode_dw_feat(&self, dw: u8) -> u8 {
//         match self.dw_buckets.binary_search(&dw) {
//             Ok(n) => n as u8,
//             Err(n) => n as u8
//         }
//     }

//     fn encode_emit_feat(&self, mut dw: u8, base: u8) -> usize {
//         dw = self.encode_dw_feat(dw);
//         ((dw as usize) << 2) + (*utils::dna::BASE_MAP.get(&base).unwrap())
//     }

//     fn ctx_trans_prob_to_string(&self) -> String {
//         let mut param_strs = vec![];

//         for i in 0..self.num_ctx {
//             let second = utils::dna::IDX_BASE_MAP.get(& ((i & 0b0011) as u8)).unwrap();
//             let first = utils::dna::IDX_BASE_MAP.get(& (((i >> 2) & 0b0011) as u8)).unwrap();
//             let ctx = String::from_iter(vec![(*first) as char, (*second) as char]);
//             let prob_str = format!("{{{}}}, // {}",
//                 self.ctx_trans_prob.slice(s![i, ..]).iter()
//                     .map(|v| v.to_string()).collect::<Vec<String>>()
//                     .join(","),
//                     ctx
//                 );
//             param_strs.push(prob_str);
//         }

//         param_strs.join("\n")
//     }

//     fn emit_prob_to_string(&self) -> String {
//         let mut param_strs = vec![];

//         for state_idx in 0..self.num_emit_state {
//             let mut state_string = vec![];
//             for ctx_idx in 0..self.num_ctx {
//                 let emit = self.emission_prob.slice(s![state_idx, ctx_idx, ..]);
//                 let emit = emit.iter().map(|v| v.to_string()).collect::<Vec<String>>().join(",");
                
//                 state_string.push(format!("{{{}}}", emit));

//             }
//             param_strs.push(format!("{{\n {} \n}}", state_string.join(",\n")));
//         }

//         param_strs.join(",\n")
//     }

//     pub fn print_params(&self) {
//         println!("{}", self.ctx_trans_prob_to_string());
//         println!("{}", self.emit_prob_to_string());
//     }

//     pub fn dump_to_file(&self, filename: &str) {
//         let file = std::fs::File::create(filename).unwrap();
//         let mut writer = BufWriter::new(file);
//         writeln!(&mut writer, "{}", self.ctx_trans_prob_to_string()).unwrap();
//         writeln!(&mut writer, "\n\n\n\n").unwrap();
//         writeln!(&mut writer, "{}", self.emit_prob_to_string()).unwrap();
//     }
    
// }

// fn train_model_parallel(aligned_bam: &str, subreads_bam: &str, ref_fasta: &str, arrow_hmm: &mut ArrowHmm, left_align_indel_version: Option<i32>) -> anyhow::Result<()>{
//     samtools_fai(ref_fasta, false)?;

//     let ref_fasta = faidx::Reader::from_path(ref_fasta).unwrap();

//     let mut subreads_info = read_subreads_bam_v2(subreads_bam);
//     let mut refname2refseq = HashMap::new();
//     let mut refname2refseq_rev_comp = HashMap::new();

//     (1..23).into_iter().map(|idx| format!("chr{}", idx)).for_each(|refname| {
//         let seq = ref_fasta.fetch_seq_string(&refname, 0, 3_000_000_000_000).unwrap();
//         refname2refseq_rev_comp.insert(refname.clone(), reverse_complement(&seq));
//         refname2refseq.insert(refname, seq);
//     });

//     thread::scope(|scope| {

//         let (init_sender, init_receiver) = crossbeam::channel::bounded(2000);

//         scope.spawn( move || {
//             let mut aligned_bam_reader = bam::Reader::from_path(aligned_bam).unwrap();
//             aligned_bam_reader.set_threads(10).unwrap();
//             let aligned_bam_header  = bam::Header::from_template(aligned_bam_reader.header());
//             let aligned_bam_header_view  = bam::HeaderView::from_header(&aligned_bam_header);

//             let mut tid2refname = HashMap::new();
//             for align_record in aligned_bam_reader.records() {
//                 let align_record = align_record.unwrap();
//                 let tid = align_record.tid();
//                 if !tid2refname.contains_key(&tid) {
//                     tid2refname.insert(tid, String::from_utf8(aligned_bam_header_view.tid2name(tid as u32).to_vec()).unwrap());
//                 }
//                 let refname = tid2refname.get(&tid).unwrap().to_string();
//                 let qname = String::from_utf8(align_record.qname().to_vec()).unwrap();
//                 let subread_record = subreads_info.get_mut(&qname).unwrap().take().unwrap();
//                 let ref_str = refname2refseq.get(&refname).unwrap();
//                 let ref_str_rev_comp = refname2refseq_rev_comp.get(&refname).unwrap();
                
//                 let ins = TrainInstance::build_train_instance(&align_record, &subread_record, ref_str, ref_str_rev_comp);

//                 init_sender.send(ins).unwrap()
//             }
//         });

//         let (instance_sender, instance_receiver) = crossbeam::channel::bounded(2000);
//         for _ in 0..20 {
//             let instance_sender_ = instance_sender.clone();
//             let init_receiver_ = init_receiver.clone();
//             scope.spawn(move || {
//                 for mut ins in init_receiver_ {
//                     ins.fill_align_ops(left_align_indel_version);
//                     instance_sender_.send(ins).unwrap();
//                 }
//             });
//         }
//         drop(instance_sender);
//         drop(init_receiver);

//         let pbar = get_spin_pb(format!("training... using {}", aligned_bam), DEFAULT_INTERVAL);
//         for ins in instance_receiver {
//             pbar.inc(1);
//             arrow_hmm.update_with_instance(&ins);
//         }
//         pbar.finish();        

//     });

    

//     Ok(())

// }


// fn read_subreads_bam_v2(subreads_bam: &str) -> HashMap<String, Option<SubreadRecord>> {
//     let mut res = HashMap::new();
//     let mut subreads_bam_reader = bam::Reader::from_path(subreads_bam).unwrap();
//     subreads_bam_reader.set_threads(10).unwrap();

//     let pbar = pb_tools::get_spin_pb(format!("reading {subreads_bam}"), DEFAULT_INTERVAL);
//     for record in subreads_bam_reader.records() {
//         pbar.inc(1);
//         let record = record.unwrap();
        
//         let mut subread_record = SubreadRecord::new(&record);
//         let qname = subread_record.qname.take().unwrap();
//         res.insert(qname, Some(subread_record));
//     }
//     pbar.finish();
//     res
// }


// pub fn train_model_entrance(arg_matches: &ArgMatches) -> Option<()> {
//     let aligned_bams = arg_matches.get_many::<String>("aligned_bams").unwrap().collect::<Vec<&String>>();
//     let subreads_bams = arg_matches.get_many::<String>("subreads_bams").unwrap().collect::<Vec<&String>>();
//     let ref_fastas = arg_matches.get_many::<String>("ref_fastas").unwrap().collect::<Vec<&String>>();
//     let left_align_indel_version = arg_matches.get_one::<i32>("left_align_indel_version").and_then(|v| Some(*v));

//     let dw_boundaries: &String = arg_matches.get_one::<String>("dw_boundaries").unwrap();
//     let dw_buckets = dw_boundaries.split(",")
//         .map(|v| v.trim())
//         .map(|v| v.parse::<u8>().unwrap())
//         .collect::<Vec<u8>>();
//     assert!(aligned_bams.len() == subreads_bams.len());
//     assert!(aligned_bams.len() == ref_fastas.len());
//     let mut arrow_hmm = ArrowHmm::new(dw_buckets);

//     for idx in 0..aligned_bams.len() {
//         // train_model(aligned_bams[idx], subreads_bams[idx], ref_fastas[idx], &mut arrow_hmm).unwrap();
//         train_model_parallel(aligned_bams[idx], subreads_bams[idx],
//             ref_fastas[idx], &mut arrow_hmm, left_align_indel_version).unwrap();
//     }
//     arrow_hmm.finish();
//     arrow_hmm.print_params();
//     arrow_hmm.dump_to_file("arrow_hg002.params");

//     Some(())
// }