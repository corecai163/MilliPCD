#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:42:00 2021
Train Color GAN
@author: pingping
"""
import torch
import torch.optim as optim

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch_geometric.transforms as T
import numpy as np
import os
from datetime import datetime
import shutil

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from chamfer_distance import ChamferDistance
from Emd.emd_module import emdFunction
from MMNet_V1 import Generator
from DatasetUofSC_All import DatasetUofSC

def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = ChD(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2

def emd(p1,p2):
    emdist, _ = emdFunction.apply(p1, p2, 0.01, 500)
    return torch.sqrt(emdist).mean()
def noise_score(d):
    
    return torch.exp(-d)
    
def train():
    G.train()
    loss_gen = 0
    loss_ini = 0
    loss_emd = 0
    with tqdm(train_data_loader) as t:
        for step, data in enumerate(t):
            #print(data)
            data = data.to(device)
            #[ele.squeeze().to(device) for ele in data]
            gt_pcd = data.y     # 10000 by 3
            #x_fft = data.fft   #N:9000 M:256 H:4  W:3
            x_pos = data.x   #N:9000 M:3
            #x_img = data.imgs   #N:  M: 1   H: 80 W:80
            #x_ang = data.ang      #N:9000 M:3
            x_ini = data.ini
            batch_size = torch.max(data.y_batch)+1
    
            G.zero_grad()
            #init,pred = G(x_ini,x_pos,data.x_batch)
            score,init,pred = G(x_ini,x_pos,data.x_batch)
            dist1, dist2, idx1, idx2 = ChD(pred.view(batch_size,-1,3), gt_pcd.view(batch_size,-1,3))  # Train G 
            dist3, dist4, idx1, idx2 = ChD(init.view(batch_size,-1,3), gt_pcd.view(batch_size,-1,3))  # Train G 
            dist5, dist6, idx3, idx4 = ChD(x_pos.view(batch_size,-1,3), gt_pcd.view(batch_size,-1,3))  # Train G 
            #print(dist5.size())
            dist5 = 2*torch.sqrt(dist5)
            #gt_score = 1/(1+dist5)
            #gt_score = 1-dist5/torch.max(dist5)
            gt_score = torch.exp(-dist5)
            loss_s = torch.mean(torch.square(score.view(batch_size,-1)-gt_score))
            loss_i = (torch.mean(dist3)) #+ (torch.mean(dist4))
            loss_e = emd(pred.view(batch_size,-1,3),gt_pcd.view(batch_size,-1,3))
            #loss_g = 0.5*(torch.mean(torch.sqrt(dist1))) + 0.5*(torch.mean(torch.sqrt(dist2)))
            loss_g = 0.5*(torch.mean(dist1)) + 2*(torch.mean(dist2)) + 2*loss_e + 2*loss_s
            
            loss_g.backward()
            g_optimizer.step()  # optimizes G's parameters
            loss_ini += loss_i.item()
            loss_gen += loss_g.item()
            loss_emd += loss_e.item()
    
    return loss_gen/step, loss_ini/step, loss_emd/step

def valid():
    G.eval()
    loss_g = 0
    loss_e = 0
    with tqdm(test_data_loader) as t:
        for step, data in enumerate(t):
            # introduce noise for y_color
            data =data.to(device)
            # 1. Test G 
            gt_pcd = data.y     # 10000 by 3
            #x_fft = data.fft   #N:9000 M:256 H:4  W:3
            x_pos = data.x   #N:9000 M:3
            #x_img = data.imgs   #N:  M: 1   H: 80 W:80
            #x_ang = data.ang      #N:9000 M:3
            x_ini = data.ini
            batch_size = torch.max(data.y_batch)+1
            #init,pred = G(x_ini,x_pos,data.x_batch)
            score,init,pred = G(x_ini,x_pos,data.x_batch)
            dist1, dist2, idx1, idx2 = ChD(pred.view(batch_size,-1,3), gt_pcd.view(batch_size,-1,3))  # test G 
            g_error = 0.5*(torch.mean(torch.sqrt(dist1))) + 0.5*(torch.mean(torch.sqrt(dist2)))
            e_error = emd(pred.view(batch_size,-1,3),gt_pcd.view(batch_size,-1,3))
            #print(g_error.size())
            loss_g += g_error.item()
            loss_e += e_error.item()
    writer.add_mesh('input', vertices=x_pos.reshape(1,-1,3))
    writer.add_mesh('predicted', vertices=pred.reshape(1,-1,3))
    writer.add_mesh('gt', vertices=gt_pcd.reshape(1,-1,3))
    #writer.add_scalar('Test/generator_loss', loss_g/len(valid_dataset))
    return loss_g/len(valid_dataset), loss_e/len(valid_dataset)

def init_weights(m):
    if isinstance(m,torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.05)
        m.bias.data.fill_(0.01)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    set_seed(5)
    
    output_dir = os.path.join('trained', '%s', datetime.now().isoformat())
    CHECKPOINTS = output_dir % 'checkpoints'
    LOGS = output_dir % 'logs'
    if not os.path.exists(CHECKPOINTS):
        os.makedirs(CHECKPOINTS)
    # backup model
    savefile = CHECKPOINTS+'/model.py'
    shutil.copyfile('MMNet_V1.py', savefile)
    #shutil.copyfile('config.py', CHECKPOINTS+'/config.py')
    #tensorborad writer
    writer = SummaryWriter(comment="MMPCD_score_2048_1e4_TV4")
    #dataset = Completion3D('../data/Completion3D', split='train', categories='Airplane')
    dataset = DatasetUofSC('../UofSC_Train_V4', split='train')
    valid_dataset = DatasetUofSC('../UofSC_Test_V4', split='test')
    print(dataset[0])

    train_data_loader = DataLoader(dataset, batch_size=16, follow_batch=['y', 'x'],shuffle=True,drop_last=True)
    test_data_loader = DataLoader(valid_dataset, batch_size=1, follow_batch=['y', 'x'],shuffle=False,drop_last=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ## model training parameter
    g_learning_rate = 1e-4

    G = Generator().to(device)
    
    fine_tune = False    
    if fine_tune:    
        # load model parameter
        model_path = './trained_old/MMNetet_Ch1000.pt'
        checkpoint = torch.load(model_path)
        G.load_state_dict(checkpoint['Gen_state_dict'])
    else:
            # initialize weight
        init_weights(G)

        
    ChD = ChamferDistance()  # chamfer loss
    g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate)
    scheduler_steplr = StepLR(g_optimizer, step_size=100, gamma=0.5)
    best_loss = 1e10
    print('Training started:')
    for epoch in range(0, 701):
        print('Train/Epoch:',epoch)
        g_loss,i_loss,e_loss = train()
        scheduler_steplr.step()
        print('GenLoss: {:.4f} \n'.format(
            g_loss))
        writer.add_scalar('Train/ini_loss', i_loss, epoch)
        writer.add_scalar('Train/generator_loss', g_loss, epoch)
        writer.add_scalar('Train/emd_loss', e_loss, epoch)
        if epoch % 5 ==0:
            print('Valid/Epoch')
            g_loss,e_loss = valid()
            print('GenLoss: {:.4f} \n'.format(g_loss))
            writer.add_scalar('Test/L1ChD_loss', g_loss, epoch)
            writer.add_scalar('Test/emd_loss', e_loss, epoch)
            if g_loss < best_loss:
                best_loss = g_loss
                path = CHECKPOINTS+'/MMNet_ChBest.pt'
                torch.save({
                'Gen_state_dict': G.state_dict(),
                'optimizerGen_state_dict': g_optimizer.state_dict(),
                }, path)
        if epoch % 100 ==0:
            path = CHECKPOINTS+'/MMNet_Ch'+'{}'.format(epoch)+'.pt'
            torch.save({
            'Gen_state_dict': G.state_dict(),
            'optimizerGen_state_dict': g_optimizer.state_dict(),
            }, path)            
        #torch.save(model.state_dict(),'./trained/Color_net_Ch'+'{}'.format(epoch)+'.pt')


