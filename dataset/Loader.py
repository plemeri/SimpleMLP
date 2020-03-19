import numpy as np
from nn.dataset.Dataloader import *

class Loader(Dataloader):
    def __init__(self, file_dir, feature_num, class_num, data_num, mode='train'):
        super().__init__()
        self.file_dir = file_dir
        self.mode = mode
        self.feature_num = feature_num
        self.class_num = class_num
        self.data_num = data_num

        self.data = np.zeros((self.data_num, self.feature_num))
        if self.mode == 'train' or self.mode == 'val':
            self.label = np.zeros((self.data_num, self.class_num))

    @staticmethod
    def one_hot(input, class_num):
        one_hot_output = np.zeros(class_num)
        one_hot_output[input] = 1
        return one_hot_output

    def read(self):
        with open(self.file_dir, 'r') as f:
            for i, data in enumerate(f.readlines()):
                data = data.split(' ')
                self.data[i] = np.array(data[:self.feature_num], dtype=float)
                if self.mode == 'train' or self.mode == 'val':
                    self.label[i] = self.one_hot(int(data[-1]), self.class_num)


if __name__ == "__main__":
    dataset = Loader('./two_moon/two_moon_train.txt', 2, 2, 150, 'train')
    dataset.read()
    print(dataset.data, dataset.label)
