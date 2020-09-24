import numpy as np
import os
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

x, y = make_moons(noise=0.2, random_state=0, n_samples=1000)
tvx, test_x, tvy, test_y = train_test_split(x, y, test_size=0.2)
train_x, val_x, train_y, val_y = train_test_split(tvx, tvy, test_size=1/8)

xx = x[np.where(y == 0), :][0]
xxx = x[np.where(y == 1), :][0]
print(xx)
plt.scatter(xx[:, 0], xx[:, 1])
plt.scatter(xxx[:, 0], xxx[:, 1])
plt.show()

make_txt(train_x, train_y, 'custom_two_moon', 'train', '../')
make_txt(val_x, val_y, 'custom_two_moon', 'val', '../')
make_txt(test_x, test_y, 'custom_two_moon', 'test', '../')