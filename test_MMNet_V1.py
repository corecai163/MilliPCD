#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 16:04:51 2021
Test ColorGAN.py
@author: pingping
"""

import torch
import torch.nn as nn
import numpy as np
from MMNet_V1 import Generator
from DatasetUofSC import DatasetUofSC
from chamfer_distance import ChamferDistance
from scipy.io import savemat, loadmat
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
#import config as cfg
from Emd.emd_module import emdFunction

def save_txt(path,pred_pcd):
    '''
    pred_pcd: N by 3
    '''
    np.savetxt(path + '.txt', pred_pcd, fmt='%.6f')
    
def emd(p1,p2):
    emdist, _ = emdFunction.apply(p1, p2, 0.01, 500)
    return torch.sqrt(emdist).mean()

if __name__ == '__main__':
    
    #tensorborad writer
    #writer = SummaryWriter(comment="color_GAN_test")
    #dataset = Completion3D('../data/Completion3D', split='train', categories='Airplane')
    test_dataset = DatasetUofSC('../Test_250', split='test')
    
    test_data_loader = DataLoader(test_dataset, batch_size=1, follow_batch=['y', 'x'],shuffle=False,drop_last=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    G = Generator().to(device)
    
    ChD = ChamferDistance()  # chamfer loss for 

    # load model parameter
    model_path = './trained/MMNet_Ch700.pt'
    checkpoint = torch.load(model_path)
    G.load_state_dict(checkpoint['Gen_state_dict'])


    G.eval()
    step = 0
    print ('Valid: ')
    loss_g =0
    each_chd = []
    each_emd = []
    for data in test_data_loader:
        print(step)
        data =data.to(device)
        # 1. Test G 
        gt_pcd = data.y     # 10000 by 3
        #x_fft = data.fft   #N:9000 M:256 H:4  W:3
        x_pos = data.x   #N:9000 M:3
        #x_img = data.imgs   #N:  M: 1   H: 80 W:80
        #x_ang = data.ang      #N:9000 M:3
        x_ini = data.ini
        batch_size = torch.max(data.y_batch)+1

        score,init,pred = G(x_ini,x_pos,data.x_batch)
        dist1, dist2, idx1, idx2 = ChD(pred, gt_pcd.view(batch_size,-1,3))  # test G 

        g_error = 0.5*(torch.mean(torch.sqrt(dist1))) + 0.5*(torch.mean(torch.sqrt(dist2)))
        #print(g_error.size())
        loss_g += g_error.item()
        emd_error = emd(pred,gt_pcd.view(batch_size,-1,3))
        gen_data = {
        'input': x_pos.cpu().numpy().reshape((-1,3)),
        'pred_pcd': pred.detach().cpu().numpy().reshape((-1,3)),
        'gt_pcd': gt_pcd.cpu().numpy().reshape((-1,3)),
        'Chd':g_error.item(),
        'EMD':emd_error.item(),
        }
        each_chd.append(g_error.item())
        each_emd.append(emd_error.item())

        savemat("output/result"+str(step)+".mat", gen_data)
        step = step + 1
    print(loss_g/len(test_dataset))
save_txt("output/chd_loss.txt",np.array(each_chd))
save_txt("output/emd_loss.txt",np.array(each_emd))
