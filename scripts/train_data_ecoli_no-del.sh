
# run_names=(
#     "20241015_240101Y0002_Run0006_called"
#     "20241015_240101Y0002_Run0008_called"
#     "20241015_240601Y0004_Run0002_called"
#     "20241015_240601Y0004_Run0003_called"
#     "20241015_240601Y0004_Run0004_called"
#     "20241015_240601Y0007_Run0002_called"
#     "20241015_240601Y0007_Run0003_called"
#     "20241015_240601Y0007_Run0006_called"
#     "20241015_240601Y0007_Run0008_called"
#     )

# out_root="/data/ccs_data/smc-upgrade/sf-train-data"

# for run_name in "${run_names[@]}"; do
#     echo $run_name
#     align-hmm train-data \
#         --sbr-bam /data/ccs_data/smc-upgrade/${run_name}.adapter.bam \
#         --ref-fa /data/ccs_data/MG1655.fa \
#         --outdir ${out_root}
# done



run_names=(
    "20241015_240101Y0002_Run0006_called_dbscan"
    "20241015_240101Y0002_Run0008_called_dbscan"
    "20241015_240601Y0004_Run0002_called_dbscan"
    "20241015_240601Y0004_Run0003_called_dbscan"
    "20241015_240601Y0004_Run0004_called_dbscan"
    "20241015_240601Y0007_Run0002_called_dbscan"
    "20241015_240601Y0007_Run0003_called_dbscan"
    "20241015_240601Y0007_Run0006_called_dbscan"
    "20241015_240601Y0007_Run0008_called_dbscan"
    )

out_root="/data/ccs_data/smc-upgrade/dbscan-train-data"

for run_name in "${run_names[@]}"; do
    echo $run_name
    align-hmm train-data \
        --sbr-bam /data/ccs_data/smc-upgrade/${run_name}.adapter.bam \
        --ref-fa /data/ccs_data/MG1655.fa \
        --outdir ${out_root}
done





