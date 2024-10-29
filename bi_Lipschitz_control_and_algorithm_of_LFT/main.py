import os
import warnings
from argparse import ArgumentParser
from train import *
from evaluate import *
warnings.filterwarnings("ignore")

def main(args):

    config = args

    config.lip_batch_size = 64
    config.print_freq = 10
    config.save_freq = 5


    config.in_channels = 1
    config.img_size = 0
    config.num_classes = 1
    config.train_batch_size = 50
    config.test_batch_size = 200
    config.num_workers = 0

    config.loss = 'mse'
    if config.scale is None:
        config.scale = 'small'
    if config.layer is None:
        config.layer = 'Plain'

    if config.gamma is None:
        config.train_dir = f"{config.root_dir}_seed{config.seed}/{config.dataset}/{config.model}-{config.layer}-{config.scale}"
    elif config.LLN:
        config.train_dir = f"{config.root_dir}_seed{config.seed}/{config.dataset}/{config.model}-{config.layer}-{config.scale}-LLN-gamma{config.gamma:.1f}"
    else:
        config.train_dir = f"{config.root_dir}_seed{config.seed}/{config.dataset}/{config.model}-{config.layer}-{config.scale}-gamma{config.gamma:.1f}"

    os.makedirs("./data", exist_ok=True)
    os.makedirs(config.train_dir, exist_ok=True)
    if config.mode == 'train':
        if config.model == 'Toy':
            train_toy(config)
        else:
            train(config)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('-m', '--model', type=str, default='Resnet',
                        help="[DNN, KWL, Resnet, Toy]")
    parser.add_argument('-d', '--dataset', type=str, default='square_wave',
                        help="dataset [square_wave, linear1, linear50]")
    parser.add_argument('-g', '--gamma', type=float, default=1.0,
                        help="Network Lipschitz bound")
    parser.add_argument('-s', '--seed', type=int, default=123)
    parser.add_argument('-e','--epochs', type=int, default=100)

    parser.add_argument('--layer', type=str, default='SLL')
    parser.add_argument('--scale', type=str, default='xlarge')
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--loss', type=str, default='xent')
    parser.add_argument('--root_dir', type=str, default='./saved_models')
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--LLN', action='store_true')
    parser.add_argument('--normalized', action='store_true')
    parser.add_argument('--cert_acc', action='store_true')

    parser.add_argument('-smooth', '--smooth', \
            type=float, default=1.0)
    parser.add_argument('-convex', '--convex', \
            type=float, default=1.0)
    parser.add_argument('-hd', '--h_dim', \
            type=int, default=10)
    parser.add_argument('-od', '--out_dim', \
            type=int, default=1)
    parser.add_argument('-cd', '--contiz_dim', \
            type=int, default=2)
    parser.add_argument('-nhl', '--num_hidden_layers', \
            type=int, default=2)
    parser.add_argument('-comp', '--composite', \
            type=bool, default=False)
    parser.add_argument('-eval', '--eval', \
            type=bool, default=False)
    parser.add_argument('-resume', '--resume', \
            type=bool, default=False)
    parser.add_argument('-brute', '--brute_force', \
            type=bool, default=False)
    args = parser.parse_args()

    main(args)
