#!/usr/bin/env python

"""
Created on Fri Dec 7 14:42:00 2021
@author: pingping
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Sequential as Seq, ReLU, GELU
from torch.nn import Dropout, Softmax, Linear, LayerNorm
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, DynamicEdgeConv
from torch_geometric.nn.pool.topk_pool import topk
#from torch_geometric.nn.models import MLP
from torch.autograd import Variable


# ##### MODELS: Generator model and discriminator model
def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Linear(channels[i - 1], channels[i]), ReLU())
        #               ,BN(channels[i]))
        for i in range(1, len(channels))
    ])
    
class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class TopK_PointNetPP(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(TopK_PointNetPP, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn,add_self_loops = True)

    def forward(self, x, score, pos, batch, num_samples=16):
        #find top k score
        idx = topk(score, self.ratio, batch, min_score=None)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=num_samples)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch
   
class PointNetPP(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(PointNetPP, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn,add_self_loops = True)

    def forward(self, x, pos, batch, num_samples=16):

        idx = fps(pos, batch, self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=num_samples)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class DGCNN(torch.nn.Module):
    def __init__(self,nn, k=16, aggr='max'):
        super(DGCNN, self).__init__()
        self.conv = DynamicEdgeConv(nn,k,aggr)
        self.activ = ReLU()
    def forward(self, x, pos, batch):
        x = self.conv(x, batch)
        x = self.activ(x)
        return x
    
class GlobalPool(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalPool, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)

        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

    
class MMGraphExtractor(torch.nn.Module):
    '''
    shape code extractor based on graph neural network
    '''
    def __init__(self):
        super(MMGraphExtractor,self).__init__()
        # pointNet++ module
        #self.top_module = TopK_PointNetPP(0.7, 0.5, MLP([576 + 3,256, 128]))
        self.pn1_module = PointNetPP(0.25, 1, MLP([1 + 3,64, 256]))
        self.pn2_module = PointNetPP(0.25, 2, MLP([256+3,256, 384]))
        #self.pn3_module = PointNetPP(0.25, 1, MLP([256 + 3, 384]))
        self.gn1_module = GlobalPool(MLP([384+3, 384, 512]))
        
    def forward(self, fea, pos, score, pi):
        '''

        '''
        # fea, pos, batch
        #top_out = self.top_module(fea, score, pos, pi)
        score = score.view(-1,1)
        pn1_out = self.pn1_module(score, pos, pi)
        #pn1_out = self.pn1_module(*top_out)
        pn2_out = self.pn2_module(*pn1_out)
        #pn3_out = self.pn3_module(*pn2_out)
        gn1_out = self.gn1_module(*pn2_out)
        #pn3_out = self.pn3_module(*pn2_out)
        return gn1_out[0]
        
class NoiseScoreNet(torch.nn.Module):
    '''
    vision extractor with for 2D images
    '''
    def __init__(self):
        super(NoiseScoreNet,self).__init__()
        self.dg1 = DGCNN(MLP([3*2, 32, 64]))
        self.dg2 = DGCNN(MLP([64*2, 128,256]))
        #self.dg3 = DGCNN(MLP([128*2,256,384]))
        #self.mlp = MLP([256,256,384])
        self.head = nn.Sequential(
                nn.Linear(576,256),
                nn.ReLU(),
                nn.Linear(256,64),
                nn.Linear(64,1)
        )
        self.activ = nn.Sigmoid()

    def forward(self, x, pos, batch):
        #print(x.size())
        batch_num = torch.max(batch).item() + 1
        #N = x.size(0)/batch_num
        #print(N)
        x1 = self.dg1(x, pos, batch)
        x2 = self.dg2(x1, pos, batch)
        #x3 = self.dg3(x2, pos, batch)
        #x3 = self.mlp(x2)
        global_fea = global_max_pool(x2, batch)
        #global_fea = global_fea.view(-1,1,256).repeat(1,N,1)
        #global_fea = global_fea.view(-1,256).contiguous()
        global_fea = torch.repeat_interleave(global_fea,1024,dim=0)
        #x = self.pool(F.relu(self.conv3(x)))
        #print(x.size())
        x4 = torch.cat([x1,x2,global_fea],dim=-1)
        fea = self.head(x4)
        score = self.activ(fea)
        return x4,score

class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=128):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_CONV(dim_feat + 128, [256, 128])
        self.mlp_3 = MLP_CONV(dim_feat + 128, [128, 128])
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat(1, 1,x1.size(2))], -2)) # (b, 256, 128)
        x3 = self.mlp_3(torch.cat([x1, feat.repeat(1, 1,x1.size(2))], -2))  # (b, 128, 256)
        #x3 = x3.permute(0, 2, 1).contiguous()
        completion = self.mlp_4(x3)  # (b, 3, 256)
        return completion  # (b, 256, 3)

class PointDecoder(torch.nn.Module):
    '''
    '''
    def __init__(self,upscale):
        super(PointDecoder,self).__init__()
        self.upscale = upscale
        self.dg1 = DGCNN(MLP([3*2, 32, 64]))
        self.ps = nn.ConvTranspose1d(128, 128, upscale,upscale, bias=False)   # point-wise splitting
        self.mlp_1 = MLP_CONV(512 + 64, [256, 128])
        self.mlp_2 = MLP_CONV(128 , [64 , 3])
        
    def forward(self,pt,shapecode):
        '''
        pt: points from previous layer
        fea: global_fea
        '''
        # local_fea: batch by 512 by 1
        B,C,N = pt.size()
        #print(patch_encoded[1].size())
        shapecode = shapecode.view(B,-1,1).repeat(1,1,pt.size(2))
        #print(pt.size())
        
        # pt: Batch by 3 by Number
        # flatten pt
        # pos: batch * number by 3
        pos = pt.permute(0,2,1).contiguous()
        pos = pos.view(-1,3)
        
        #build batch_index
        patch_vec = torch.arange(B,dtype=torch.int64).view(-1,1)
        patch_vec = patch_vec.repeat(1,N)
        batch = patch_vec.view(-1).to('cuda')
        # dgfea = Batch * Number by 64
        dgfea = self.dg1(pos, pos, batch)
        # rel_fea: Batch by 64 by Number
        rel_fea = dgfea.view(B,-1,64).permute(0,2,1)
        point_fea = torch.cat((shapecode, rel_fea), -2)
        
        x1 = self.mlp_1(point_fea) # B by 128 by N
        x_expand = self.ps(x1) # B by 128 by N*upscale 
        out = self.mlp_2(x_expand)  # B by 3 by N*upscale
        out = out+torch.repeat_interleave(pt, self.upscale, dim=2)
        # rescale xyz for each patch
        #torch.sigmoid(out)
        return out   
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.score_estimator = NoiseScoreNet()
        self.shape_extractor = MMGraphExtractor()
        self.seed_gen = SeedGenerator(dim_feat=512, num_pc=64)
        self.decoder2 = PointDecoder(upscale=4)
        self.decoder = PointDecoder(upscale=2)

    def forward(self,x_ini, x_pos, x_pi):
        '''
        x: the feature of original incomplete color pcd
            : conditional feature
        y: complete pcd without color
            : pos
        '''
        ## feature extractor
        batch_num = torch.max(x_pi) + 1
        #x_fea = x_fea.T.contiguous() 
        x_fea, score = self.score_estimator(x_pos.squeeze(),x_pos.squeeze(), x_pi)
        shape_fea = self.shape_extractor(x_fea, x_pos, score.squeeze(), x_pi)
        
        ## point cloud reconstruction
        shape_fea = shape_fea.view(batch_num,-1,1)
        init_point = self.seed_gen(shape_fea)
        pd_point = self.decoder(init_point,shape_fea)
        pd_point = self.decoder2(pd_point,shape_fea)
        pd_point = self.decoder2(pd_point,shape_fea)
        
        #x_ini_coarse = x_ini.view(-1,1,3).repeat(1,init_point.size(2),1)
        #ini_points = init_point.view(batch_num,-1,3)
        ini_points = init_point.permute(0, 2, 1).contiguous()
        pd_points = pd_point.permute(0, 2, 1).contiguous()
        return score,ini_points,pd_points



