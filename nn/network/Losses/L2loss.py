import numpy as np
from .Loss import *


def l2loss(logit, label):
    if logit.shape != label.shape:
        raise AttributeError("logit and label shape are not equal")
    loss = np.sum(np.mean((label - logit) ** 2, axis=0)) / 2
    return loss


def d_l2loss(logit, label):
    # return np.expand_dims(np.mean(logit - label, axis=0), axis=0)
    return logit - label

class L2Loss(Loss):
    def __init__(self):
        super().__init__(l2loss, d_l2loss)

