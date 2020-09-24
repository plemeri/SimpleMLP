from nn.network.Layers.Dense import *
from nn.network.Losses.L2loss import *
from network.Classifier import *
from utils.utils import *
from dataset.Loader import *

def Mlp(args):
    loader_spec = (args.feature_num, args.class_num)

    if args.mode == 'train':
        train_dataset = Loader(args.train_data_dir, *loader_spec, data_num=args.train_data_num)
        validation_dataset = Loader(args.validation_data_dir, *loader_spec, data_num=args.validation_data_num)
    elif args.mode == 'validation':
        train_dataset = None
        validation_dataset = Loader(args.validation_data_dir, *loader_spec, data_num=args.validation_data_num)
    else:
        train_dataset = None
        validation_dataset = None

    model = Classifier(train_dataset, validation_dataset, L2Loss(), accuracy, args.epochs, args.learning_rate)

    layer_spec = (sigmoid, d_sigmoid, args.checkpoint_dir)
    model.append_layer(Dense(args.feature_num, args.hidden_layer_neurons[0], *layer_spec, scope="input_layer"))
    for i in range(args.hidden_layer_num - 1):
        model.append_layer(Dense(args.hidden_layer_neurons[i], args.hidden_layer_neurons[i + 1], *layer_spec, scope="hidden_layer_" + str(i)))
    model.append_layer(Dense(args.hidden_layer_neurons[-1], args.class_num, *layer_spec, scope="output_layer"))
    return model
