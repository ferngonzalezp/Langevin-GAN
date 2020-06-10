# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:42:25 2020

@author: fgp35
"""

import os
from argparse import ArgumentParser
import torch
import pytorch_lightning as pl


import turboGAN3d

def main(hparams):     
    save_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(os.getcwd(),save_top_k=-1,period=hparams.max_epochs-1) 
    save_callback.format_checkpoint_name({},{})
    
    model = turboGAN3d.GAN3d(hparams)

    trainer = pl.Trainer(gpus=int(hparams.gpus), max_epochs = hparams.max_epochs, checkpoint_callback=save_callback,
                         num_nodes = hparams.num_nodes, auto_select_gpus = True,
                         distributed_backend=hparams.distributed_backend,
                         resume_from_checkpoint=hparams.resume_from_checkpoint)    
    trainer.fit(model)
    

if __name__ == '__main__':
    __author__ = 'Fernando Gonzalez'
    parser = ArgumentParser(description='Trainig GAN for 2D turbulence')
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('-bs' , '--batch_size', help='batch size',default=8, type=int)
    parser.add_argument('-lr' , '--lr'  , help='learning rate',default=0.0001,  type=float)
    parser.add_argument('-b1' , '--b1' , help='b1 for adam', default = 0, type=float)
    parser.add_argument('-b2', '--b2', help='b2 dor adam ', default = 0.99, type=float)
    parser.add_argument('-ld','--latent_dim', help='latent vector dimension',default = 192, type = int)
    parser.add_argument('-sc','--sc',help='use lr scheduler scheduler',required=False,action='store_true')
    parser.add_argument('-ml' , '--milestones'  , help='lr milestones', default=10, nargs='+', type=int)
    parser.add_argument('-g' , '--gamma'  , help='lr sche gamma', default=0.05, type=float)
    parser.add_argument('-nv','--nv',help='dont use vorticity',required=False,action='store_true')
    hparams = parser.parse_args()
    
    main(hparams)