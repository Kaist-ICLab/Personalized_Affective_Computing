for dataset in kemowork; do
    python tune_one_cluster.py "$1" "$dataset"
done