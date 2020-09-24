import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


def make_txt(x, y, name, split, dir):
    file = open(os.path.join(dir, name + '_' + split + '.txt'), 'w')
    for feat, lab in zip(x, y):
        feats = ""
        for f in feat:
            feats += str(f) + ' '
        feats += str(lab)
        file.write(feats + '\n')
    file.close()


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default='../dataset')
    parser.add_argument('--name', type=str, default='custom_two_moon')
    return parser.parse_args()


if __name__ == "__main__":
    flags = args()
    x, y = make_moons(noise=flags.noise, random_state=flags.random_state, n_samples=flags.n_samples)
    tvx, test_x, tvy, test_y = train_test_split(x, y, test_size=0.2)
    train_x, val_x, train_y, val_y = train_test_split(tvx, tvy, test_size=0.25)

    print('train: ', len(train_y), 'val: ', len(val_y), 'test: ', len(test_y))

    zeros = x[np.where(y == 0), :][0]
    ones = x[np.where(y == 1), :][0]
    plt.scatter(zeros[:, 0], zeros[:, 1])
    plt.scatter(ones[:, 0], ones[:, 1])
    plt.show()

    if os.path.isdir(os.path.join(flags.save_dir, flags.name)) is False:
        os.makedirs(os.path.join(flags.save_dir, flags.name))

    make_txt(train_x, train_y, flags.name, 'train', os.path.join(flags.save_dir, flags.name))
    make_txt(val_x, val_y, flags.name, 'val', os.path.join(flags.save_dir, flags.name))
    make_txt(test_x, test_y, flags.name, 'test', os.path.join(flags.save_dir, flags.name))