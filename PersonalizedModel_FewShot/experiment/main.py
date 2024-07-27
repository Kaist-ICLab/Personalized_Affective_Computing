import tensorflow as tf
import random
import sys
sys.path.append("./")
from model.fcn import ClassifierFcn
from model.mlp_lstm import ClassifierMlpLstm
from model.resnet import ClassifierResnet
from train import *
from dataset import *
import argparse


def main():

    parser = argparse.ArgumentParser(description='Process training index.')
    parser.add_argument('--train_idx', type=int, required=True, help='An index for the training set')

    args = parser.parse_args()

    dataset = CustomDataset()
   
    result_dir = os.path.join('./results/ascertain')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    print('Training...')

    FCN = ClassifierFcn(dataset.input_shapes, 2)
    model = FCN.model

    # LSTM = ClassifierMlpLstm(dataset.input_shapes, 2)
    # model = LSTM.model

    # Resnet = ClassifierResnet(dataset.input_shapes, 2)
    # model = Resnet.model

    train(model, meta_batch_size=30, test_idx=args.train_idx, save_dir=result_dir)

if __name__ == '__main__':
    main()