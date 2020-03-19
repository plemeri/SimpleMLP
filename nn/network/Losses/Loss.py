class Loss:
    def __init__(self, loss_fn, d_loss_fn):
        self.loss_fn = loss_fn
        self.d_loss_fn = d_loss_fn

    def forward(self, logits, labels):
        self.logits = logits
        self.labels = labels
        self.loss = self.loss_fn(self.logits, self.labels)
        return self.loss

    def backward(self):
        self.gradient = self.d_loss_fn(self.logits, self.labels)
        return self.gradient
