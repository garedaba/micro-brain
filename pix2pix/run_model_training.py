import os

from models import define_discriminator
from models import define_generator
from models import define_gan
from models import train

from utils import load_dataset
from utils import summarise_performance
from utils import get_res_and_size

from tensorflow.keras.utils import plot_model

import json

def main(train_dir='.', depth = 3, batch_number = 1, num_epochs = 100, loss_ratio = 100, lr1=0.0002, lr2 = 0.0002, jitter = False, labels = False, config_args=None):

    ### Extracts size and resolution from name of directory containing training data - can hard code if required
    patch_size, patch_resolution = get_res_and_size(train_dir)

    ### NOTE DIRECTORY NAMING STRUCTURE - EDIT AS NEEDED ###
    outdir = 'model_output/model_res{:}_size{:}_depth{:02d}_batch{:02d}_epochs{:03d}_lr1_{:}_lr2_{:}_lossweight{:03d}'.format(patch_resolution, patch_size, depth, batch_number, num_epochs, lr1, lr2, loss_ratio)
    if jitter:
        outdir = outdir + '_jitter'
    if labels:
        outdir = outdir + '_classlabels'

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir + '/model_plots', exist_ok=True)

    if config_args:
        with open(outdir + '/commandline_args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    #------------------------------------------------------------------------------------------------#
    # set up models
    # discriminator
    print('')
    print('DISCRIMINATOR')
    image_trg_shape = (patch_size, patch_size, 3)
    if labels:
        image_src_shape = (patch_size, patch_size, 21)
    else:
        image_src_shape = image_trg_shape

    d_model = define_discriminator(image_src_shape, image_trg_shape, initial_filters=64, depth=depth, lr=lr1)
    d_model.summary()

    # GENERATOR
    print('')
    print('GENERATOR')
    g_model = define_generator(image_src_shape)
    g_model.summary()

    # gan
    print('')
    print('GAN')
    gan_model = define_gan(g_model, d_model, image_src_shape, lr=lr2, loss_ratio=loss_ratio)
    gan_model.summary()

    #------------------------------------------------------------------------------------------------#
    # LOAD DATASET
    ### training_blocks.pkl contains pickled collection of paired image patches - edit as needed ###
    dataset = load_dataset(train_dir + '/training_blocks.pkl')

    #------------------------------------------------------------------------------------------------#
    # TRAIN
    loss = train(d_model, g_model, gan_model, dataset, jitter=jitter, labels=labels, n_epochs=num_epochs, n_batch=batch_number, savepath=outdir)
    loss.to_csv('{:}/model_loss.csv'.format(outdir), index=None)


if __name__ == '__main__':
    import argparse, os

    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)

    parser = argparse.ArgumentParser(description='Run pix2pix generation of histology from labels')

    parser.add_argument('-t', '--train_dir', type=dir_path, default='.', required=True,
            help='location of training data')
    parser.add_argument('-d', '--depth', type=int, metavar='n_layers', default=3, choices=[1, 2, 3], required=True,
            help='depth of patchGAN discriminator')
    parser.add_argument('-b', '--batch_num', type=int,  default=1, required=True,
            help='batch size for training')
    parser.add_argument('-n', '--number_epochs', type=int,  default=100, required=True,
            help='number of training epochs')
    parser.add_argument('-l', '--loss_ratio', type=int,  default=100, required=False,
                help='upweighting of G loss, loss_weight=[1:l]')
    parser.add_argument('--learning_rate1', type=float,  default=0.0002, required=False,
                    help='learning rate for discriminator')
    parser.add_argument('--learning_rate2', type=float,  default=0.0002, required=False,
                        help='learning rate for GAN')
    parser.add_argument('-j', '--jitter', action='store_true',
                help='apply data augmentation by jitter, random crop and flip')
    parser.add_argument('-c', '--rgb_to_labels', action='store_true',
            help='use class channels instead of RGB for label patches')

    args = parser.parse_args()

    main(train_dir = args.train_dir,
         depth = args.depth,
         batch_number = args.batch_num,
         num_epochs = args.number_epochs,
         loss_ratio = args.loss_ratio,
         lr1 = args.learning_rate1,
         lr2 = args.learning_rate2,
         jitter = args.jitter,
         labels = args.rgb_to_labels, config_args=args)
