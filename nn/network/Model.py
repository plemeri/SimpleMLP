class Model:
    def __init__(self, train_dataset, validation_dataset, Loss, criteria, epochs, learning_rate):
        self.layers = []
        self.loss = Loss
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.criteria = criteria
        self.epochs = epochs
        self.learning_rate = learning_rate

        if self.train_dataset is not None:
            self.train_data, self.train_label = self.read_dataset(self.train_dataset)
        if self.validation_dataset is not None:
            self.validation_data, self.validation_label = self.read_dataset(self.validation_dataset)

    def read_dataset(self, dataset, *args, **kwargs):
        dataset.read()
        return dataset.data, dataset.label

    def append_layer(self, layer, *args, **kwargs):
        self.layers.append(layer)

    def __str__(self):
        out = ""
        out += "=" * 80 + "\n"
        for i, layer_ in enumerate(self.layers):
            out += layer_.__str__()
        out += "=" * 80 + "\n"
        return out

    def save(self, *args, **kwargs):
        for layer in self.layers:
            layer.save()

    def load(self, *args, **kwargs):
        for layer in self.layers:
            layer.load()

    def train(self, *args, **kwargs):
        out = self.train_data
        for layer_ in self.layers:
            out = layer_.forward(out)

        loss_ = self.loss.forward(out, self.train_label)
        layer_grad = self.loss.backward()
        for layer_ in self.layers[::-1]:
            layer_grad = layer_.backward(layer_grad, self.learning_rate)

        return out, loss_

    def fit(self, *args, **kwargs):
        for i in range(self.epochs):
            self.train()

    def validation(self, *args, **kwargs):
        out = self.validation_data
        for layer_ in self.layers:
            out = layer_.forward(out)
        loss_ = self.loss.forward(out, self.validation_label)
        return out, loss_

    def predict(self, test_dataset, *args, **kwargs):
        test_dataset.read()
        out = test_dataset.data
        for layer_ in self.layers:
            out = layer_.forward(out)

        return out

