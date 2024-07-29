# This script was created because of probable memory leakage in version of tensorflow/keras used in this project
for clas in fcnM resnetM mlpLstmM; do
  for dataset in kemophone; do
    for max_eval in $(seq 1 1); do
      for i_fold in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46; do
        python tune_one.py "$1" "$dataset"_fold_47_"$i_fold" $clas "$max_eval"
      done
    done
  done
done