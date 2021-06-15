import argparse
import os
from util import util
import torch
import models
import data
import easydict

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self):
        """Define the common options that are used in both training and test."""
        # basic parameters
        
        opt = easydict.EasyDict({
            'dataroot' : '/opt/ml/input/data/forgan/testA',
            'name' : 'graycleaning',
            'gpu_ids' : 0,
            'checkpoints_dir' : './checkpoints',
            'model' : 'test',
            'input_nc' : 1,
            'output_nc' : 1,
            'ngf' : 64,
            'ndf' : 64,
            'netD' : 'basic',
            'netG' : 'resnet_9blocks',
            'n_layers_D' : 3,
            'norm' : 'instance',
            'init_type' : 'normal',
            'init_gain' : 0.02,
            'no_dropout' : 'True',
            'dataset_mode' : 'single',
            'direction' : 'AtoB',
            'serial_batches' : True,
            'num_threads' : 4,
            'batch_size' : 3,
            'load_size' : 512,
            'crop_size' : 512,
            'max_dataset_size' : float("inf"),
            'preprocess' : 'none',
            'no_flip' : True,
            'display_winsize' : 256,
            'epoch' : 'latest',
            'load_iter' : 0,
            'verbose' : True,
            'suffix' : '',
            'results_dir' : './results/',
            'aspect_ratio' : 1.0,
            'phase' : 'test',
            'eval' : True,
            'model_suffix' : ''
        })
        self.initialized = True
        self.isTrain = False
        return opt

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            opt = self.initialize()

        # get the basic options
        # modify model-related parser options
        
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(opt, self.isTrain)

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(opt, self.isTrain)

        # save and return the parser
        self.parser = opt
        return opt

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in list(opt.items()):
            comment = ''
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        #str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = [0]
        torch.cuda.set_device(opt.gpu_ids[0])
        
        self.opt = opt
        return self.opt
