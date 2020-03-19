class Dataloader:
    def __init__(self):
        super(Dataloader, self).__init__()

    def read(self, *args, **kwargs):
        raise NotImplementedError("implement read function")
