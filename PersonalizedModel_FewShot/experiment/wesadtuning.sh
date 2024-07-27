# This script was created because of probable memory leakage in version of tensorflow/keras used in this project
# for clas in fcnM resnetM mlpLstmM; do
  # for dataset in wesad; do
    # for max_eval in $(seq 1 1); do
      for i_fold in 2 3 4 5 6 7 8 9 10 11 13 14 15 16 17; do
        python experiment/main.py --train_idx "$i_fold"
      done
    # done
  # done
# done