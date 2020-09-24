from nn.network.Losses.Loss import *
from nn.network.Model import Model
from utils.utils import *

class Classifier(Model):
    def __init__(self, train_dataset, validation_dataset, Loss, criteria, epochs, learning_rate):
        super().__init__(train_dataset, validation_dataset, Loss, criteria, epochs, learning_rate)

    def train(self, verbose=True):
        out, loss = super().train()
        acc = accuracy(out, self.train_label)
        if verbose is True:
            print('train_loss: {0:2f}, train_accuracy: {1}'.format(loss, acc), end='')

    def fit(self, verbose_step=50):
        best_acc = 0
        for i in range(self.epochs):
            if i % verbose_step == 0:
                print('\repoch: {0} '.format(i + 1), end='')
            self.train(not(i % verbose_step))
            acc = self.validation(verbose=False)
            if acc > best_acc:
                best_acc = acc
                self.save()
        self.validation()
        print("\ndone")

    def validation(self, verbose=True):
        out, loss = super().validation()
        acc = accuracy(out, self.validation_label)
        if verbose is True:
            print('\nvalidation_loss: ', loss, 'validation accuracy: ', acc)
        return acc

    def predict(self, test_dataset, *args, **kwargs):
        result_file_dir = kwargs['result_file_dir']
        out = super().predict(test_dataset)

        with open(result_file_dir, 'w') as f:
            for out_ in np.argmax(out, axis=-1):
                f.writelines(str(out_) + '\n')


