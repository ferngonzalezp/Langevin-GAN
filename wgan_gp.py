# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:32:47 2020

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
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

class mirror(object):
 def __init__(self):
   super().__init__()
 def __call__(self,field):
    p = torch.rand(1)
    if p < 0.25:
      return torch.flip(field,[0,1])
    elif 0.25 <= p < 0.5:
      return torch.flip(field,[0,2])
    elif 0.5 <= p < 0.75:
      return torch.flip(field,[0,1,2])
    else:
      return field

class transform(object):
  def __init__(self):
    self.transform = torchvision.transforms.Compose([
                    mirror(),
    ])
  def __call__(self,field):
    return self.transform(field)


class mydataset(torch.utils.data.Dataset):
  def __init__(self,field,transform=None):
    self.field = field
    self.transform = transform
  def __len__(self):
    return self.field.shape[0]
  
  def __getitem__(self,idx):
    if torch.is_tensor(idx):
            idx = idx.tolist()
    sample = self.field[idx,:]
    if self.transform:
        sample = self.transform(sample)
    return sample

def stream_vorticity(field):
  b_size = field.shape[0]
  w = np.zeros((b_size,128,128))
  for i in range(b_size):
    w[i,:,:] = np.gradient(field[i,1,:,:].cpu().detach(),axis=0)-np.gradient(field[i,0,:,:].cpu().detach(),axis=1)
  w = torch.tensor(w)
  return w.reshape(b_size,1,128,128)

def cov(y):
  mu = torch.mean(y,dim=(0))
  covu = torch.mean(torch.matmul(y[:,0],y[:,0].transpose(-1,-2)),dim=0)-torch.matmul(mu[0],mu[0].transpose(-1,-2))
  covv = torch.mean(torch.matmul(y[:,1],y[:,1].transpose(-1,-2)),dim=0)-torch.matmul(mu[1],mu[1].transpose(-1,-2))
  covw = torch.mean(torch.matmul(y[:,2],y[:,2].transpose(-1,-2)),dim=0)-torch.matmul(mu[2],mu[2].transpose(-1,-2))
  sigma = torch.cat((covu.unsqueeze(0),covv.unsqueeze(0),covw.unsqueeze(0)))
  return sigma

def spec(field,lx=9*np.pi/50,smooth=True):
  n = field.shape[2]
  uh = torch.rfft(field[:,0],1,onesided=False)/n
  vh = torch.rfft(field[:,1],1,onesided=False)/n
  wh = torch.rfft(field[:,2],1,onesided=False)/n
  uspec = 0.5 * (uh[:,:,:,0]**2+uh[:,:,:,1]**2)
  vspec = 0.5 * (vh[:,:,:,0]**2+vh[:,:,:,1]**2)
  wspec = 0.5 * (wh[:,:,:,0]**2+wh[:,:,:,1]**2)
  uspec = uspec.reshape(uspec.shape[0],1,n,n)
  vspec = vspec.reshape(vspec.shape[0],1,n,n)
  wspec = wspec.reshape(wspec.shape[0],1,n,n)
  k = 2.0 * np.pi / lx
  wave_numbers = k*np.arange(0,n)
  spec = torch.cat((uspec,vspec,wspec),dim=1)
  spec[:,:,:,int(n/2+1):,] = 0

  if smooth == True:
    window = torch.ones(3,3,5,5).type_as(spec)/ 5
    specsmooth = nn.functional.conv2d(spec,window,padding=2)
    #specsmooth[:,:,:,0:4] = spec[:,:,:,0:4]
    spec = specsmooth

  return wave_numbers, spec

def stat_cosntraint2(x,y):
  _, specx = spec(x)
  _, specy = spec(y)
  return torch.norm(specx-specy)

def score(x,y):
  covx = cov(x)
  covy = cov(y)
  nf = x.shape[2]
  nc = x.shape[1]
  mux = torch.mean(x,dim=0)
  muy = torch.mean(y,dim=0)
  term1 = torch.norm(mux-muy)
  covmean = np.zeros((nc,nf,nf))
  for i in range(nc):
    covmean[i] = sqrtm((torch.matmul(covx[i],covy[i])).cpu().detach())
  covmean = torch.tensor(covmean).to(x.device).type_as(x)
  term2 = (torch.diagonal(covx+covy - 2*covmean, dim1=-2, dim2=-1).sum(-1).view(nc,1))
  return torch.mean(term1 + torch.abs(term2))


def calc_gradient_penalty(netD, real_data, generated_data,l = 10):
    # GP strengt
    LAMBDA = l

    b_size = real_data.size()[0]

    # Calculate interpolation
    bs = [b_size]
    for i in range(len(real_data[0].shape)):
        bs.append(1)
    alpha = torch.rand(bs).to(real_data.device)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.type_as(real_data)

    interpolated = alpha * real_data + (1 - alpha) * generated_data
    interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data,0.2,mode='fan_out')
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data,0.2,mode='fan_out')

class PixelNormLayer(nn.Module):
    """
    Pixelwise feature vector normalization.
    """
    def __init__(self, eps=1e-8):
        super(PixelNormLayer, self).__init__()
        self.eps = eps
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % (self.eps)

class Generator(nn.Module):

  def __init__(self,latent_dim):
    super(Generator, self).__init__()

    self.input_layer = nn.Sequential(
      nn.ConvTranspose2d(latent_dim,latent_dim,4),
      nn.LeakyReLU(0.2,True),
      PixelNormLayer()
    )

    def block(in_feats,out_feats):
      layers = [nn.UpsamplingBilinear2d(scale_factor=2)]
      layers.append(nn.Conv2d(in_feats,out_feats,3,padding=1))
      layers.append(nn.LeakyReLU(0.2,True))
      layers.append(PixelNormLayer())
      layers.append(nn.Conv2d(out_feats,out_feats,3,padding=1))
      layers.append(nn.LeakyReLU(0.2,True))
      layers.append(PixelNormLayer())
      return layers

    self.main = nn.Sequential(
        nn.Conv2d(192,192,3,padding=1),
        nn.LeakyReLU(0.2,True),
        PixelNormLayer(),
        # size 4 x 4 x 192
        *block(192,192),
        # size 8 x 8 x 192
        *block(192,192),
        # size 16 x 16 x 192
        *block(192,96),
        # size 32 x 32 x 96
        *block(96,48),
        # size 64 x 64 x 48
        #*block(96,1),
        #size 128 x 128 x 3
        nn.Conv2d(48,3,1),
    )

  def forward(self,z):
    nz = z.shape[1]
    b_size = z.shape[0]
    z = z.reshape(b_size,nz,1,1)
    z = self.input_layer(z)
    return self.main(z)

class Discriminator(nn.Module):

  def __init__(self,use_vorticity=True):
    super(Discriminator,self).__init__()
    if use_vorticity:
        self.input_features = 4
    else:
        self.input_features = 3

    def block(in_feats,out_feats):
      layers = [nn.Conv2d(in_feats,in_feats,3,padding=1)]
      layers.append(nn.LeakyReLU(0.2,True))
      layers.append(nn.Conv2d(in_feats,out_feats,3,padding=1))
      layers.append(nn.LeakyReLU(0.2,True))
      layers.append(nn.AvgPool2d(2))
      return layers
    
    self.main = nn.Sequential(
        nn.ConvTranspose2d(self.input_features,48,1),
        # 128 x 128 x 48
        *block(48,96),
        # 64 x 64 x 96
        *block(96,192),
        # 32 x 32 x 192
        *block(192,192),
        # 16 x 16 x 192
        *block(192,192),
        # 8 x 8 x 192
        #*block(192,192),
        # 4 x 4 x 192
    )

    self.last_block = nn.Sequential(
        nn.Conv2d(192+1,192,3,padding=1),
        nn.LeakyReLU(0.2,True),
        nn.Conv2d(192,192,4),
        nn.LeakyReLU(0.2,True),
    )
    self.fc = nn.Linear(192,1,bias=False)

  def forward(self,field):
      b_size = field.shape[0]
      field = self.main(field)
      mstd = torch.std(field,dim=1).unsqueeze(1)
      field = torch.cat((field,mstd),dim=1)
      field = self.last_block(field)
      field = field.reshape(b_size,192)
      return self.fc(field)

class GAN(pl.LightningModule):
  def __init__(self, hparams):
    super(GAN, self).__init__()
    torch.cuda.seed_all()
    self.hparams = hparams

    # networks
    self.netG = Generator(latent_dim=hparams.latent_dim)
    self.netG.apply(weights_init)
    if hparams.nv:
        self.netD = Discriminator(use_vorticity=False)
    else:
        self.netD = Discriminator()
    self.netD.apply(weights_init)
    # cache for generated images
    self.generated_imgs = None
    self.last_imgs = None
    
    self.G_losses = []
    self.D_losses = []
    self.score = []
    self.iters = 0
  
  def forward(self, z):
      return self.netG(z)

  def adversarial_loss(self, y, y_hat):
      return -torch.mean((y)) + torch.mean((y_hat)) 

  def training_step(self, batch, batch_nb, optimizer_idx):
      
      real_field = batch[0]
      self.last_imgs = real_field
      if not self.hparams.nv:
          omega = stream_vorticity(real_field).type_as(real_field)
          real_field = torch.cat((real_field,omega),1)
      if self.on_gpu:    
        real_field = real_field.cuda(real_field.device.index)

      if optimizer_idx == 0:
          z = torch.randn(real_field.shape[0],self.hparams.latent_dim).type_as(real_field)
          if self.on_gpu:
            z = z.cuda(real_field.device.index)
          gen_field = self(z)
          if not self.hparams.nv:
              omega = stream_vorticity(gen_field).type_as(gen_field)
              gen_field = torch.cat((gen_field,omega),1)

          grad_penalty = calc_gradient_penalty(self.netD,real_field,gen_field)
          d_loss = self.adversarial_loss(self.netD(real_field),self.netD(gen_field)) + grad_penalty
          fid = score(real_field,gen_field).detach()
          tqdm_dict = {'d_loss': d_loss, 'score': fid}
          self.score.append(fid)
          output = OrderedDict({
              'loss': d_loss,
              'progress_bar': tqdm_dict,
              'log': tqdm_dict,
              })
          self.D_losses.append(d_loss.detach())
          self.iters += 1
          return output
      
      if optimizer_idx == 1:
          #if batch_nb % 5 == 0:
              z = torch.randn(real_field.shape[0],self.hparams.latent_dim).type_as(real_field)
              gen_field = self(z)
              self.generated_imgs = gen_field
              if not self.hparams.nv:
                  omega = stream_vorticity(gen_field).type_as(gen_field)
                  gen_field = torch.cat((gen_field,omega),1)
    
              g_loss = -torch.mean(self.netD(gen_field))
              fid = score(real_field,gen_field).detach()
              tqdm_dict = {'g_loss': g_loss,'score': fid}
              self.score.append(fid)
              output = OrderedDict({
                  'loss': g_loss,
                  'progress_bar': tqdm_dict,
                  'log': tqdm_dict,
                  })
              self.G_losses.append(g_loss.detach())
              self.iters += 1
              return output
      
  def configure_optimizers(self):

      lr = self.hparams.lr
      b1 = self.hparams.b1
      b2 = self.hparams.b2

      opt_g = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(b1, b2))
      opt_d = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(b1, b2))
      if self.hparams.sc:
          scheduler_d = torch.optim.lr_scheduler.MultiStepLR(opt_d,milestones=self.hparams.milestones,gamma=self.hparams.gamma)
          scheduler_g = torch.optim.lr_scheduler.MultiStepLR(opt_g,milestones=self.hparams.milestones,gamma=self.hparams.gamma)
          return [opt_d, opt_g], [scheduler_d,scheduler_g]
      else:
          return opt_d, opt_g

  def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
    # update generator opt every 5 steps
    if optimizer_i == 0:
        if batch_nb % 1 == 0 :
            optimizer.step()
            optimizer.zero_grad()

    # update discriminator opt every step
    if optimizer_i == 1:
        if batch_nb % 1 == 0 :
            optimizer.step()
            optimizer.zero_grad()
    
  def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.hparams.batch_size,shuffle=True)
  
  def prepare_data(self):
    path = os.getcwd() 
    dataset = dset.CelebA(path, split='train', transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), target_transform=None, target_type='attr',
                               download = False)
    self.dataset = dataset
    
  def on_epoch_end(self):
        z = torch.randn(8, self.hparams.latent_dim)
        # match gpu device (or keep as cpu)
        if self.on_gpu:
            z = z.cuda(self.last_imgs.device.index)

        # log sampled images
        sample_imgs = self(z)
        grid = vutils.make_grid(sample_imgs)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)
