import numpy as np

def sigmoid(input):
    return 1/(1 + np.exp(-input))

def d_sigmoid(input):
    return sigmoid(input) * (1 - sigmoid(input))

def relu(input):
    return np.maximum(input, 0)

def d_relu(input):
    return (relu(input) > 0).astype(float)

def softmax1d(input):
    input_copy = np.copy(input)
    if len(input_copy.shape) == 1:
        divisor = np.exp(input_copy)
        divisor = np.sum(divisor)
        for _class in range(input_copy.shape[0]):
            input_copy[_class] = np.exp(input_copy[_class]) / divisor
    else:
        for batch in range(input_copy.shape[0]):
            divisor = np.exp(input_copy[batch, :])
            divisor = np.sum(divisor)
            for _class in range(input_copy.shape[1]):
                input_copy[batch, _class] = np.exp(input_copy[batch, _class]) / divisor
    return input_copy


def CEloss(logit, label):
    if logit.shape != label.shape:
        raise AttributeError("logit and label shape are not equal")
    logit = softmax1d(logit)
    loss = np.sum(np.mean(-label * np.log(logit), axis=0))
    return loss

def d_CEloss(logit, label):
    return logit - label

def accuracy(logit, label):
    prediction = np.argmax(logit, axis=-1)
    label = np.argmax(label, axis=-1)
    return np.mean(prediction == label)


if __name__ == "__main__":
    x = np.zeros((10, 3))
    x[0:3, 0] = 0.9
    x[0:3, 1] = 0.05
    x[0:3, 2] = 0.05
    x[3:6, 0] = 0.05
    x[3:6, 1] = 0.9
    x[3:6, 2] = 0.05
    x[6:, 0] = 0.05
    x[6:, 1] = 0.05
    x[6:, 2] = 0.9
    y = np.zeros((10, 3), dtype=int)
    y[0:3, 0] = 1
    y[3:6, 2] = 1
    y[6:, 1] = 1
    print(x, y)
    print(d_l2loss(x, y))
    print(d_CEloss(x, y))
    # print(delta_cross_entropy(x, y))
