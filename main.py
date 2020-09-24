import argparse
from model.Mlp import *
from dataset.Loader import *

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'validation', 'test'])
    parser.add_argument('--feature_num', type=int, default=2)
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--hidden_layer_num', type=int, default=1)
    parser.add_argument('--hidden_layer_neurons', type=int, nargs='+', default=[10])
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--train_data_dir', type=str, default='./dataset/custom_two_moon/custom_two_moon_train.txt')
    parser.add_argument('--train_data_num', type=int, default=700)
    parser.add_argument('--validation_data_dir', type=str, default='./dataset/custom_two_moon/custom_two_moon_val.txt')
    parser.add_argument('--validation_data_num', type=int, default=100)
    parser.add_argument('--test_data_dir', type=str, default='./dataset/custom_two_moon/custom_two_moon_test.txt')
    parser.add_argument('--test_data_num', type=int, default=200)
    parser.add_argument('--test_output_dir', type=str, default='./result.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/custom_two_moon')
    return parser.parse_args()


if __name__ == "__main__":
    args = args()
    model = Mlp(args)
    print(model)
    try:
        model.load()
    except:
        print('from scratch')

    if args.mode == 'train':
        model.fit()

    elif args.mode == 'validation':
        model.validation(verbose=True)

    elif args.mode == 'test':
        test_dataset = Loader(args.test_data_dir, args.feature_num, args.class_num, data_num=args.test_data_num, mode='test')
        model.predict(test_dataset, result_file_dir='./result.txt')


