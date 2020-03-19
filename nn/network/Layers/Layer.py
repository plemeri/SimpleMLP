class Layer:
    def __init__(self, checkpoint_dir=None, scope=""):
        self.checkpoint_dir = checkpoint_dir
        self.scope = scope
        super(Layer, self).__init__()

    def visualize(self):
        raise NotImplementedError('visualize function not implemented')

    def load(self, *args, **kwargs):
        raise NotImplementedError('load function not implemented')

    def save(self, *args, **kwargs):
        raise NotImplementedError('save function not implemented')

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward function not implemented')

    def backward(self, *args, **kwargs):
        raise NotImplementedError('backward function not implemented')
