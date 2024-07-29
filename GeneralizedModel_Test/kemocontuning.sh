# This script was created because of probable memory leakage in version of tensorflow/keras used in this project
for clas in fcnM resnetM mlpLstmM; do
  for dataset in kemocon; do
    for max_eval in $(seq 1 1); do
      for i_fold in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20; do
        python tune_one.py "$1" "$dataset"_fold_21_"$i_fold" $clas "$max_eval"
      done
    done
  done
done