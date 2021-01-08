
import argparse
import logging
import os
import time

import torch
import torch.optim as optim
from thop import profile
from thop import clever_format
from models import DPTNet_base
from others.optimizer_dptnet import TransformerOptimizer
from others.data import AudioDataset, AudioDataLoader, EvalDataset, EvalDataLoader

from utils import device

from solver import Solver

parser = argparse.ArgumentParser( "DPTNet")
# General config
# Task related
parser.add_argument('--train_dir', type=str, default='/data/WSJ/wsj_data/min/tr',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default='/data/WSJ/wsj_data/min/cv',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4.0, type=float,
                    help='Segment length (seconds)')
parser.add_argument('--cv_maxlen', default=8.0, type=float,
                    help='max audio length (seconds) in cv, to avoid OOM issue.')
# Network architecture
parser.add_argument('--N', default=64, type=int,
                    help='Dim of feature to the DPT blocks')
parser.add_argument('--W', default=2, type=int,
                    help='Filter length in encoder, stride is W//2')
parser.add_argument('--K', default=250, type=int,
                    help='Chunk size in frames')
parser.add_argument('--D', default=6, type=int,
                    help='Number of DPT blocks')
parser.add_argument('--C', default=2, type=int,
                    help='Number of speakers')
parser.add_argument('--E', default=256, type=int,
                    help='Number of channels before bottleneck 1x1-conv block')
parser.add_argument('--H', default=128, type=int,
                    help='Number of hidden units in LSTM and linear layer after MHA')
# Training config
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--shuffle', default=1, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int,
                    help='Enables checkpoint saving of model after every epoch.')
parser.add_argument('--continue_from', default='',
                    help='Continue from the checkpointed model specified.')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='The first epoch number in this run. Important when '
                    'loading a checkpoint since lr depends on epoch. Ignored '
                    'if a checkpoint is not loaded. Unneccessary if eval only.')
parser.add_argument('--warmup', default=1, type=int,
                    help='Perform the warmup. Important for when loading a ' 
                    'checkpoint since the warmup should be skipped if the '
                    'checkpoint is advanced enough.')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument('--mode', type=str, default='train_and_eval',
                    choices=['train_and_eval', 'train', 'eval'])
# logging
parser.add_argument('--print_freq', default=10, type=int,
                    help='Frequency in itrs of printing training infomation')

def main(args):
    # setup logging that prints and saves to a file
    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger('pytorch')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    os.makedirs(args.save_folder, exist_ok=True)
    fh = logging.FileHandler(os.path.join(args.save_folder,
        '{}-{}.log'.format(args.mode, time.strftime("%Y%m%d-%H%M%S"))))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    logger.info(args)
    
    # data
    tr_dataset = AudioDataset(args.train_dir, batch_size=1,
                              sample_rate=args.sample_rate, segment=args.segment)
    cv_dataset = AudioDataset(args.valid_dir, batch_size=args.batch_size,
                              sample_rate=args.sample_rate,
                              segment=-1, cv_maxlen=args.cv_maxlen)
    tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                shuffle=args.shuffle)
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1, shuffle=False)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    
    model = DPTNet_base(enc_dim=args.E, feature_dim=args.N, hidden_dim=args.H, layer=args.D, segment_size=args.K, nspk=2, win_len=args.W)
    logger.info(model)
    
    if args.use_cuda:
        # model = torch.nn.DataParallel(model)
        model.cuda()
        model.to(device)
    
    optimizier = TransformerOptimizer(optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9), k=0.2, d_model=args.N, warmup_steps=4000, warmup=bool(args.warmup))

    solver = Solver(data, model, optimizier, args)
    solver.run()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)