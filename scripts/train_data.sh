
run_names=(
    "20240402_Sync_Y0003_02_H01_Run0002" 
    "20240408_Sync_Y0002_01_H01_Run0002" 
    "20240408_Sync_Y0002_02_H01_Run0003"
    "20240409_Sync_Y0006_01_H01_Run0002"
    "20240411_Sync_Y0701_01_H01_Run0002"
    "20240411_Sync_Y0701_02_H01_Run0003"
    "20240416_Sync_Y0002_01_H01_Run0002"
    "20240416_Sync_Y0003_01_H01_Run0003"
    "20240416_Sync_Y0006_04_H01_Run0002"
    "20240416_Sync_Y0701_01_H01_Run0003"
    )

out_root="/data/ccs_data/HG002/smc-train-data"

for run_name in "${run_names[@]}"; do
    echo $run_name
    align-hmm train-data \
        --sbr-bam /data/ccs_data/HG002/${run_name}_called_subreads.bam \
        --ref-fa /data/ccs_data/HG002/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta \
        --outdir ${out_root} \
        --bed-file /data/ccs_data/HG002/HG002_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed \
        --vcf-file /data/ccs_data/HG002/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf 
done
