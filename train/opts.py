import argparse
import os

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--load_model', type=bool, default=False,
                             help='load model flag')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--gpus', default='0', 
                             help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=4,
                                help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317, 
                                help='random seed') # from CornerNet
        self.parser.add_argument('--test', type=bool, default=False)
        
        self.parser.add_argument('--print_iter', type=int, default=0, 
                             help='disable progress bar and print to screen.')
        self.parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
        self.parser.add_argument('--metric', default='loss', 
                             help='main metric to save best model')
        
        self.parser.add_argument('--arch', default='res_18', 
                             help='model architecture. Currently tested'
                                  'res_18 | res_101 | resdcn_18 | resdcn_101 |'
                                  'dlav0_34 | dla_34 | hourglass')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                             help='output stride. Currently only supports 4.')
        
        # input
        self.parser.add_argument('--input_res', type=int, default=-1, 
                                help='input height and width. -1 for default from '
                                'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h', type=int, default=-1, 
                                help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=-1, 
                                help='input width. -1 for default from dataset.')

        self.parser.add_argument('--lr', type=float, default=1.25e-4, 
                             help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=str, default='200',
                                help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=140,
                                help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,
                                help='batch size on the master gpu.')
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                help='default: #samples / batch_size.')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                help='number of epochs to run validation.')
        self.parser.add_argument('--trainval', action='store_true',
                                help='include validation in training and '
                                    'test on test set')
        self.parser.add_argument('--resume', action='store_true',
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.') 
        
        # test
        self.parser.add_argument('--flip_test', action='store_true',
                                help='flip data augmentation.')
        self.parser.add_argument('--test_scales', type=str, default='1',
                                help='multi scale test augmentation.')
        self.parser.add_argument('--nms', action='store_true',
                                help='run nms in testing.')
        self.parser.add_argument('--K', type=int, default=100,
                                help='max number of output objects.') 
        self.parser.add_argument('--not_prefetch_test', action='store_true',
                                help='not use parallal data pre-processing.')
        self.parser.add_argument('--fix_res', action='store_true',
                                help='fix testing resolution or keep '
                                    'the original resolution')
        self.parser.add_argument('--keep_res', action='store_true',
                                help='keep the original resolution'
                                    ' during validation.')
    
    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.test_scales = [float(i) for i in opt.test_scales.split(',')]


        opt.root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        print(opt.root_dir)
        opt.data_dir = os.path.join(opt.root_dir, 'data')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp')
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        opt.num_stacks = 1
        
        print('The output will be saved to ', opt.save_dir)

        return opt
        # print(os.listdir(opt.save_dir))