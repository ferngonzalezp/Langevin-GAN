# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:28:57 2020

@author: fgp35
"""

import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torchvision
import pytorch_lightning as pl
from scipy.linalg import sqrtm
from turboGAN2d import *

class mirror3d(object):
 def __init__(self):
   super().__init__()
 def __call__(self,field):
    p = torch.rand(1)
    if p < 0.25:
      return torch.flip(field,[0,1,2])
    elif 0.25 <= p < 0.5:
      return torch.flip(field,[0,1,3])
    elif 0.5 <= p < 0.75:
      return torch.flip(field,[0,1,2,3])
    else:
      return field

class transform3d(object):
  def __init__(self):
    self.transform = torchvision.transforms.Compose([
                    mirror3d(),
    ])
  def __call__(self,field):
    return self.transform(field)

def mse1(x,y):
    s = y;
    t = x.shape[2]
    s_hat = 0
    for i in range(t):
        s_hat += spec(x[:,:,i])[1]/t
    if x.is_cuda:
        s_hat = s_hat.cuda(x.device.index)
        s = s.cuda(x.device.index)
    return torch.norm(s-s_hat)

def t_correlation(x):
    m, n = x[0,0].shape
    bs = x.shape[0]
    t = x.shape[1]
    x = x.cpu().detach().numpy()
    r = np.zeros((t,m,n))
    for i in range(m):
     for j in range(n):
         for b in range(bs):
                r[:,i,j] += np.correlate(x[b,:,i,j],x[b,:,i,j],mode='full')[t-1:]/bs
         r[:,i,j] /= max(r[:,i,j])
    return r


def s2(x):
    nf = x.shape[1]
    t = x.shape[2]
    m,n = x[0,0,0].shape
    s = np.zeros((nf,t,m,n))
    for i in range(nf):
        s[i] = t_correlation(x[:,i])
    s = torch.tensor(s,requires_grad=x.requires_grad)
    return s.to(x.device)

def mse2(x,y):
    s_hat = y
    s = s2(x)
    if x.is_cuda:
        s = s.cuda(x.device.index)
        s_hat = s_hat.cuda(x.device.index)
    return torch.norm(s-s_hat)


def s3(latent_vector):
    mean = torch.mean(latent_vector)
    rms = torch.sqrt(torch.mean(latent_vector**2))
    sk = torch.mean(((latent_vector-mean)/torch.std(latent_vector))**3)
    k = sk = torch.mean(((latent_vector-mean)/torch.std(latent_vector))**4)
    return torch.tensor((mean,rms,sk,k)).type_as(latent_vector)

def mse3(x,y):
    s = s3(x)
    s_hat = s3(y)
    return torch.norm(s-s_hat)
                        
    
class Discriminator_norm(nn.Module):
    def __init__(self,latent_dim):
        super(Discriminator_norm,self).__init__()
        
        self.main = nn.Sequential(
                nn.Linear(latent_dim+4,255),
                nn.LeakyReLU(0.2,True),
                nn.Linear(255,255),
                nn.LeakyReLU(0.2,True),
                nn.Linear(255,255),
                nn.LeakyReLU(0.2,True),
                nn.Linear(255,255),
                nn.LeakyReLU(0.2,True),
                )
    def forward(self,latent_vector):
        bs = latent_vector.shape[0]
        mean = torch.mean(latent_vector)
        rms = torch.sqrt(torch.mean(latent_vector**2))
        sk = torch.mean(((latent_vector-mean)/torch.std(latent_vector))**3)
        k = sk = torch.mean(((latent_vector-mean)/torch.std(latent_vector))**4)
        moments = torch.tensor((mean,rms,sk,k)).type_as(latent_vector)
        moments = moments.expand(bs,4)
        latent_vector = torch.cat((latent_vector,moments),dim=1)
        return self.main(latent_vector)

class Discriminator_time(nn.Module):
   def __init__(self,use_vorticity=True):
    super(Discriminator_time,self).__init__()
    if use_vorticity:
        self.input_features = 4
    else:
        self.input_features = 3

    def block(in_feats,out_feats):
      layers = [nn.ConvTranspose3d(in_feats,in_feats,3,padding=1)]
      layers.append(nn.LeakyReLU(0.2,True))
      layers.append(nn.ConvTranspose3d(in_feats,out_feats,3,padding=1))
      layers.append(nn.LeakyReLU(0.2,True))
      layers.append(nn.AvgPool3d((1,2,2)))
      return layers
    
    self.main = nn.Sequential(
        nn.ConvTranspose3d(self.input_features,24,1),
        # 128 x 128 x 4 x 24
        *block(24,48),
        # 64 x 64 x x 4 x 96
        *block(48,96),
        # 32 x 32 x 4 x 96
        *block(96,96),
        # 16 x 16 x 4 x 96
        *block(96,96),
        # 8 x 8 x 4 x 96
        *block(96,96),
        # 4 x 4 x  4 x 96
        nn.ConvTranspose3d(96,96,3,padding=1),
        nn.LeakyReLU(0.2,True),
        # 4 x 4 x 4 x 96
    )
    self.last_block = nn.Sequential(
        nn.Conv3d(96+1,96,3,padding=1),
        nn.LeakyReLU(0.2,True),
        nn.Conv3d(96,96,4),
        nn.LeakyReLU(0.2,True),
    )
    self.fc = nn.Linear(96,1,bias=False)

   def forward(self,field):
     b_size = field.shape[0]
     field = self.main(field)
     mstd = torch.std(field,dim=1).unsqueeze(1)
     field = torch.cat((field,mstd),dim=1)
     field = self.last_block(field)
     field = field.reshape(b_size,96)
     return self.fc(field)
 

class RNN(nn.Module):
    def __init__(self,hidden_size):
        super(RNN,self).__init__()
        self.hs = hidden_size
        self.main = nn.LSTM(192,self.hs,num_layers=3, batch_first=True)
        self.fc = nn.Linear(self.hs,192)
    def forward(self,z,hidden):
        z = z.view(z.shape[0],1,z.shape[1])
        out,(hn,cn) = self.main(z,hidden)
        return self.fc(out), (hn,cn)
    def init_hidden(self, batch_size):
          ''' Initialize hidden state '''
          # create NEW tensor with SAME TYPE as weight
          weight = next(self.parameters()).data
          hidden = (weight.new(3,batch_size, self.hs).normal_(mean=0,std=0.1),
                     weight.new(3,batch_size, self.hs).normal_(mean=0,std=0.1))        
          return hidden

class GAN3d(pl.LightningModule):
    def __init__(self,hparams):
        super(GAN3d,self).__init__()
        torch.cuda.seed_all()
        self.hparams = hparams
        
        #networks
        GAN2d = GAN.load_from_checkpoint(os.getcwd()+'/pre_trainGan.ckpt')
        self.netG = GAN2d.netG
        self.netD = GAN2d.netD
        
        self.D_time = Discriminator_time()
        self.D_time.apply(weights_init)
        self.D_norm = Discriminator_norm(hparams.latent_dim)
        self.D_norm.apply(weights_init)
        self.RNN = RNN(500)
    def evaluate_lstm(self,z,t):
        hidden = self.RNN.init_hidden(z.shape[0])
        output = z.view(z.shape[0],1,z.shape[1])
        ot = z
        for i in range(1,t):
          ot, hidden = self.RNN(ot.view_as(z),hidden)
          output = torch.cat((output,ot),dim=1)
        ot = None
        return output
    def forward(self,z,t):
        bs = z.shape[0]
        zt = self.evaluate_lstm(z,t)
        field = self.netG(zt[:,0]).reshape(bs,3,1,128,128)
        for i in range(1,t):
            field_i = self.netG(zt[:,i]).reshape(bs,3,1,128,128)
            field = torch.cat((field,field_i),dim=2)
        return field
    def adversarial_loss(self, y, y_hat):
        return -torch.mean((y)) + torch.mean((y_hat)) 
  
    def training_step(self, batch, batch_nb, optimizer_idx):
        real_field = batch
        self.s1 = self.s1.type_as(real_field)
        self.s2 = self.s2.type_as(real_field)
        t = real_field.shape[2]
        if not self.hparams.nv:
          omega = stream_vorticity(real_field[:,:,0]).type_as(real_field[:,:,0])
          for i in range(1,t):
              omega = torch.cat((omega,stream_vorticity(real_field[:,:,i]).type_as(real_field[:,:,i])),dim=0)
          real_field = torch.cat((real_field,omega.view(real_field.shape[0],1,t,128,128)),dim=1)
        
        if optimizer_idx == 0:
            z = torch.randn(real_field.shape[0],self.hparams.latent_dim).type_as(real_field)
            gen_field = self.netG(z)
            if not self.hparams.nv:
              omega = stream_vorticity(gen_field).type_as(gen_field)
              gen_field = torch.cat((gen_field,omega),1)
            grad_penalty = calc_gradient_penalty(self.netD,real_field[:,:,0],gen_field,l=100)
            d_loss = self.adversarial_loss(self.netD(real_field[:,:,0]),self.netD(gen_field)) + grad_penalty
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
              'loss': d_loss,
              'progress_bar': tqdm_dict,
              'log': tqdm_dict,
              })
            return output
        
        if optimizer_idx == 1:
          z = torch.randn(real_field.shape[0],self.hparams.latent_dim).type_as(real_field)
          gen_field = self.netG(z)
          if not self.hparams.nv:
              omega = stream_vorticity(gen_field).type_as(gen_field)
              gen_field = torch.cat((gen_field,omega),1)
          gen_field_t = self(z,4)
          if not self.hparams.nv:
              omega = stream_vorticity(gen_field_t[:,:,0]).type_as(gen_field)
              for i in range(1,4):
                  omega = torch.cat((omega,stream_vorticity(gen_field_t[:,:,i]).type_as(gen_field)),dim=0)
              gen_field_t = torch.cat((gen_field_t,omega.view(real_field.shape[0],1,t,128,128)),dim=1)
        
          g_loss = (-torch.mean(self.netD(gen_field)) -torch.mean(self.D_time(gen_field_t)) 
                     + 10*mse1(gen_field_t,self.s1) +1*mse2(gen_field_t[:,0:3],self.s2))
          fid = score(real_field[:,:,0],gen_field_t[:,:,0]).detach()
          for i in range(1,4):
              fid += score(real_field[:,:,i],gen_field_t[:,:,i]).detach()
          fid = fid/4
          tqdm_dict = {'g_loss': g_loss,'score': fid}
          output = OrderedDict({
              'loss': g_loss,
              'progress_bar': tqdm_dict,
              'log': tqdm_dict,
              })
          return output
      
        if optimizer_idx ==2:
            z = torch.randn(real_field.shape[0],self.hparams.latent_dim).type_as(real_field)
            gen_field_t = self(z,4)
            
            if not self.hparams.nv:
              omega = stream_vorticity(gen_field_t[:,:,0]).type_as(gen_field_t)
              for i in range(1,4):
                  omega = torch.cat((omega,stream_vorticity(gen_field_t[:,:,i]).type_as(gen_field_t)),dim=0)
              gen_field_t = torch.cat((gen_field_t,omega.view(real_field.shape[0],1,t,128,128)),dim=1)
              
            grad_penalty = calc_gradient_penalty(self.D_time,real_field,gen_field_t,l=400)
            d_time_loss = self.adversarial_loss(self.D_time(real_field),self.D_time(gen_field_t)) + grad_penalty
            fid = score(real_field[:,:,0],gen_field_t[:,:,0]).detach()
            for i in range(1,4):
                  fid += score(real_field[:,:,i],gen_field_t[:,:,i]).detach()
            fid = fid/4
            tqdm_dict = {'d_time_loss': d_time_loss, 'score': fid}
            output = OrderedDict({
              'loss': d_time_loss,
              'progress_bar': tqdm_dict,
              'log': tqdm_dict,
               })
            return output
        
        if optimizer_idx == 3:
             z = torch.randn(real_field.shape[0],self.hparams.latent_dim).type_as(real_field)
             zt = self.evaluate_lstm(z,500)
             zt = zt[:,np.random.randint(50,500)].view_as(z)
             grad_penalty = calc_gradient_penalty(self.D_norm,z,zt)
             d_norm_loss = self.adversarial_loss(self.D_norm(z),self.D_norm(zt)) + grad_penalty
             tqdm_dict = {'d_norm_loss': d_norm_loss}
             output = OrderedDict({
              'loss': d_norm_loss,
              'progress_bar': tqdm_dict,
              'log': tqdm_dict,
               })
             return output
         
        if optimizer_idx == 4:
            z = torch.randn(real_field.shape[0],self.hparams.latent_dim).type_as(real_field)
            zt = self.evaluate_lstm(z,500)
            zt = zt[:,np.random.randint(50,500)].view_as(z)
            gen_field_t = self(z,4)
            
            if not self.hparams.nv:
              omega = stream_vorticity(gen_field_t[:,:,0]).type_as(gen_field_t)
              for i in range(1,4):
                  omega = torch.cat((omega,stream_vorticity(gen_field_t[:,:,i]).type_as(gen_field_t)),dim=0)
              gen_field_t = torch.cat((gen_field_t,omega.view(real_field.shape[0],1,t,128,128)),dim=1)
            
            rnn_loss = (-torch.mean(self.D_time(gen_field_t)) -torch.mean(self.D_norm(zt)) + 10*mse1(gen_field_t,self.s1)
                        +1*mse2(gen_field_t[:,0:3],self.s2) +100*mse3(z,zt))
            fid = score(real_field[:,:,0],gen_field_t[:,:,0]).detach()
            for i in range(1,4):
              fid += score(real_field[:,:,i],gen_field_t[:,:,i]).detach()
            fid = fid/4
            tqdm_dict = {'rnn_loss': rnn_loss, 'score': fid}
            output = OrderedDict({
              'loss': rnn_loss,
              'progress_bar': tqdm_dict,
              'log': tqdm_dict,
               })
            return output
        
    def configure_optimizers(self):

      lr = self.hparams.lr
      b1 = self.hparams.b1
      b2 = self.hparams.b2

      opt_g = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(b1, b2))
      opt_d = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(b1, b2))
      opt_d_time = torch.optim.Adam(self.D_time.parameters(), lr=lr, betas=(b1, b2))
      opt_d_norm = torch.optim.Adam(self.D_norm.parameters(), lr=lr, betas=(b1, b2))
      opt_rnn = torch.optim.Adam(self.RNN.parameters(), lr=lr, betas=(b1, b2))
      if self.hparams.sc:
          scheduler_d = torch.optim.lr_scheduler.MultiStepLR(opt_d,milestones=self.hparams.milestones,gamma=self.hparams.gamma)
          scheduler_g = torch.optim.lr_scheduler.MultiStepLR(opt_g,milestones=self.hparams.milestones,gamma=self.hparams.gamma)
          scheduler_dt = torch.optim.lr_scheduler.MultiStepLR(opt_d_time,milestones=self.hparams.milestones,gamma=self.hparams.gamma)
          scheduler_dn = torch.optim.lr_scheduler.MultiStepLR(opt_d_norm,milestones=self.hparams.milestones,gamma=self.hparams.gamma)
          scheduler_rnn = torch.optim.lr_scheduler.MultiStepLR(opt_rnn,milestones=self.hparams.milestones,gamma=self.hparams.gamma)
          return [opt_d, opt_g, opt_d_time, opt_d_norm, opt_rnn], [scheduler_d,scheduler_g,scheduler_dt,scheduler_dn,scheduler_rnn]
      else:
          return opt_d, opt_g, opt_d_time, opt_d_norm, opt_rnn
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.hparams.batch_size,)
    
    def prepare_data(self):
        path = os.getcwd()
        field = torch.load(path+'/field.pt')
        dataset = mydataset(field, transform=transform3d())
        self.dataset = dataset
        t = field.shape[2]
        s_hat = 0
        for i in range(t):
           s_hat += spec(field[0:100,:,i])[1]/t
        self.s1 = torch.mean(s_hat,dim=0).unsqueeze(0)   
        self.s2 = s2(field)