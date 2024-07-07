# This script was created because of probable memory leakage in version of tensorflow/keras used in this project
# for clas in fcnM resnetM mlpLstmM; do
  # for dataset in wesad; do
    # for max_eval in $(seq 1 1); do
      # for i_fold in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
      for i_fold in 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
        python experiment/main.py --train_idx "$i_fold"
      done
    # done
  # done
# done