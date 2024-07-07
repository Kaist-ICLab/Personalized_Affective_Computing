import tensorflow as tf
import random
import sys
sys.path.append("./")
from model.fcn import ClassifierFcn
from train import *
from dataset import *
import argparse
# Case
# SUBJECTS_IDS = list(it.chain(range(2, 12), range(13, 18)))
# WESAD
# SUBJECTS_IDS = list(it.chain(range(2, 12), range(13, 18)))

def main():
    """
    Load data and train a model on it.
    """

    parser = argparse.ArgumentParser(description='Process training index.')
    parser.add_argument('--train_idx', type=int, required=True, help='An index for the training set')

    args = parser.parse_args()

    dataset = CustomDataset()
    # 데이터셋 읽기 함수가 필요합니다. 예제를 위해 `read_dataset` 함수를 정의해야 합니다.
    # `read_dataset(DATA_DIR)` 대신 실제 데이터셋 경로를 사용하세요.
   
    result_dir = os.path.join('./results/ascertain')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)  # 결과 디렉토리가 없으면 생성
    print('Training...')
    # for i in SUBJECTS_IDS:
    FCN = ClassifierFcn(dataset.input_shapes,2)
    model = FCN.model
    train(model, meta_batch_size=30, test_idx=args.train_idx, save_dir=result_dir)

if __name__ == '__main__':
    main()