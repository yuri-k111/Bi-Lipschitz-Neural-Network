import torch
import numpy as np
import time
from model import getModel
from dataset import getDataLoader
from utils import *
from matplotlib import pyplot as plt
import gc
import sys
import psutil
import os
import subprocess
import matplotlib.pyplot as plt
from matplotlib import animation
from argparse import ArgumentParser
class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, use_bias=False):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        # if use_bias:
        #     self.bias = nn.Parameter(torch.Tensor(out_features))
        # else:
        #     self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        #nn.init.xavier_uniform_(self.weight, 0.01)
        nn.init.xavier_normal_(self.weight)

    def forward(self, input):
        #return nn.functional.linear(input, torch.clamp(self.weight.exp(), min=0.0, max=1e10))
        return nn.functional.linear(input, torch.clamp(self.weight, min=0.0))


class BLNN2(nn.Module):
    """
    Bi-Lipschitz Neural network with alpha = 0
    """
    def __init__(self,config):
        super().__init__()

        self.contiz_dim = config.in_channels
        self.h_dim = config.h_dim
        self.out_dim = config.num_classes
        self.smooth=config.smooth
        self.convex=config.convex
        self.composite = config.composite
        self.brute_force = False
        self.max_iter = config.max_iter
        self.device = torch.device("cuda:0")#should i change this???
        self.px1_num_hidden_layers = config.num_hidden_layers
        self.seed = config.seed
        self.icnn1_Wy0 = nn.Linear(self.contiz_dim, self.h_dim).to(self.device)
        icnn1_Wy_layers = []
        icnn1_Wz_layers = []
        for i in range(self.px1_num_hidden_layers-1):
            icnn1_Wy_layers.append(nn.Linear(self.contiz_dim, self.h_dim).to(self.device))
            icnn1_Wz_layers.append(PositiveLinear(self.h_dim, self.h_dim).to(self.device))
        icnn1_Wy_layers.append(nn.Linear(self.contiz_dim, 1).to(self.device))
        icnn1_Wz_layers.append(PositiveLinear(self.h_dim, 1).to(self.device))

        self.icnn1_Wy_layers = nn.ModuleList(icnn1_Wy_layers)
        self.icnn1_Wz_layers = nn.ModuleList(icnn1_Wz_layers)

        if self.composite == True:
            self.px2_num_hidden_layers = config.num_hidden_layers

            self.icnn2_Wy0 = nn.Linear(self.out_dim, self.h_dim).to(self.device)
            icnn2_Wy_layers = []
            icnn2_Wz_layers = []
            for i in range(self.px2_num_hidden_layers-1):
                icnn2_Wy_layers.append(nn.Linear(self.out_dim, self.h_dim).to(self.device))
                icnn2_Wz_layers.append(PositiveLinear(self.h_dim, self.h_dim).to(self.device))
            icnn2_Wy_layers.append(nn.Linear(self.out_dim, 1).to(self.device))
            icnn2_Wz_layers.append(PositiveLinear(self.h_dim, 1).to(self.device))

            self.icnn2_Wy_layers = nn.ModuleList(icnn1_Wy_layers)
            self.icnn2_Wz_layers = nn.ModuleList(icnn1_Wz_layers)
    def f1(self, input):

        #h2[0] = torch.pow(nn.ReLU()(self.icnn1_Wy0(input)),4)
        h2 = nn.Softplus()(self.icnn1_Wy0(input))
        #h2[0] = nn.ELU()((self.icnn1_Wy0(input)))
        #h2[0] = torch.pow(nn.ReLU()(self.icnn1_Wy0(input)),2)
        #h2 = nn.ReLU()(self.icnn1_Wy0(input))

        #h1[0] = 4*torch.pow((nn.ReLU()(self.icnn1_Wy0(input))),3).view(-1,self.h_dim,1)*self.icnn1_Wy0.weight
        h1 = torch.sigmoid((self.icnn1_Wy0(input)).view(-1,self.h_dim,1))*self.icnn1_Wy0.weight
        #h1[0] = torch.minimum(torch.ones(h2[0].size()).to(self.device),torch.exp(self.icnn1_Wy0(input))).view(-1,self.h_dim,1)*self.icnn1_Wy0.weight
        #h1[0] = 2*(nn.ReLU()(self.icnn1_Wy0(input))).view(-1,self.h_dim,1)*self.icnn1_Wy0.weight
        #h1 = 0.5*(torch.sign(self.icnn1_Wy0(input))+1).view(-1,self.h_dim,1)*self.icnn1_Wy0.weight
        for i in range(self.px1_num_hidden_layers):
            #h2[i+1] = torch.pow(nn.ReLU()(self.icnn1_Wz_layers[i](h2[i]) + self.icnn1_Wy_layers[i](input)),4)
            h2_n = nn.Softplus()(self.icnn1_Wz_layers[i](h2) + self.icnn1_Wy_layers[i](input))
            #h2[i+1] = nn.ELU()(self.icnn1_Wz_layers[i](h2[i]) + self.icnn1_Wy_layers[i](input))
            #h2[i+1] = torch.pow(nn.ReLU()(self.icnn1_Wz_layers[i](h2[i]) + self.icnn1_Wy_layers[i](input)),2)
            #h2_n= nn.ReLU()(self.icnn1_Wz_layers[i](h2) + self.icnn1_Wy_layers[i](input))

            #h1[i+1] = 4*torch.pow((nn.ReLU()(self.icnn1_Wz_layers[i](h2[i]) + self.icnn1_Wy_layers[i](input))),3).view(-1,self.icnn1_Wy_layers[i].weight.size()[0],1)*(torch.clamp(self.icnn1_Wz_layers[i].weight, min=0.0, max=1e5)@h1[i] + self.icnn1_Wy_layers[i].weight)
            h1_n = torch.sigmoid(self.icnn1_Wz_layers[i](h2) + self.icnn1_Wy_layers[i](input)).view(-1,self.icnn1_Wy_layers[i].weight.size()[0],1)*(torch.clamp(self.icnn1_Wz_layers[i].weight, min=0.0)@h1 + self.icnn1_Wy_layers[i].weight)
            #h1[i+1] = torch.minimum(torch.ones(h2[i+1].size()).to(self.device),torch.exp(self.icnn1_Wz_layers[i](h2[i]) + self.icnn1_Wy_layers[i](input))).view(-1,self.icnn1_Wy_layers[i].weight.size()[0],1)*(torch.clamp(self.icnn1_Wz_layers[i].weight, min=0.0, max=1e5)@h1[i] + self.icnn1_Wy_layers[i].weight)
            #h1[i+1] = 2*(nn.ReLU()(self.icnn1_Wz_layers[i](h2[i]) + self.icnn1_Wy_layers[i](input))).view(-1,self.icnn1_Wy_layers[i].weight.size()[0],1)*(torch.clamp(self.icnn1_Wz_layers[i].weight, min=0.0, max=1e5)@h1[i] + self.icnn1_Wy_layers[i].weight)
            #h1_n = 0.5*(torch.sign(self.icnn1_Wz_layers[i](h2) + self.icnn1_Wy_layers[i](input))+1).view(-1,self.icnn1_Wy_layers[i].weight.size()[0],1)*(torch.clamp(self.icnn1_Wz_layers[i].weight, min=0.0, max=1e10)@h1 + self.icnn1_Wy_layers[i].weight)

            h2 = h2_n
            h1 = h1_n

        grad_icnn = h1.view(-1,self.contiz_dim) + 1/self.smooth*input
        icnn_output = h2 + 1/(2*self.smooth)*(torch.norm(input,dim=1)**2).view(-1,1)

        return grad_icnn#, icnn_output

    def f1_with_output(self, input):

        #h1 = [[None] for i in range(self.px1_num_hidden_layers + 1)]
        #h2 = [[None] for i in range(self.px1_num_hidden_layers + 1)]
        #h2[0] = torch.pow(nn.ReLU()(self.icnn1_Wy0(input)),4)
        h2 = nn.Softplus()(self.icnn1_Wy0(input))
        #h2[0] = nn.ELU()((self.icnn1_Wy0(input)))
        #h2[0] = torch.pow(nn.ReLU()(self.icnn1_Wy0(input)),2)
        #h2 = nn.ReLU()(self.icnn1_Wy0(input))

        #h1[0] = 4*torch.pow((nn.ReLU()(self.icnn1_Wy0(input))),3).view(-1,self.h_dim,1)*self.icnn1_Wy0.weight
        h1 = torch.sigmoid((self.icnn1_Wy0(input)).view(-1,self.h_dim,1))*self.icnn1_Wy0.weight
        #h1[0] = torch.minimum(torch.ones(h2[0].size()).to(self.device),torch.exp(self.icnn1_Wy0(input))).view(-1,self.h_dim,1)*self.icnn1_Wy0.weight
        #h1[0] = 2*(nn.ReLU()(self.icnn1_Wy0(input))).view(-1,self.h_dim,1)*self.icnn1_Wy0.weight
        #h1 = 0.5*torch.mul((torch.sign(self.icnn1_Wy0(input))+1).view(-1,self.h_dim,1),self.icnn1_Wy0.weight)
        for i in range(self.px1_num_hidden_layers):
            #h2[i+1] = torch.pow(nn.ReLU()(self.icnn1_Wz_layers[i](h2[i]) + self.icnn1_Wy_layers[i](input)),4)
            h2_n = nn.Softplus()(self.icnn1_Wz_layers[i](h2) + self.icnn1_Wy_layers[i](input))
            #h2[i+1] = nn.ELU()(self.icnn1_Wz_layers[i](h2[i]) + self.icnn1_Wy_layers[i](input))
            #h2[i+1] = torch.pow(nn.ReLU()(self.icnn1_Wz_layers[i](h2[i]) + self.icnn1_Wy_layers[i](input)),2)
            #h2_n= nn.ReLU()(self.icnn1_Wz_layers[i](h2) + self.icnn1_Wy_layers[i](input))

            #h1[i+1] = 4*torch.pow((nn.ReLU()(self.icnn1_Wz_layers[i](h2[i]) + self.icnn1_Wy_layers[i](input))),3).view(-1,self.icnn1_Wy_layers[i].weight.size()[0],1)*(torch.clamp(self.icnn1_Wz_layers[i].weight, min=0.0, max=1e10)@h1[i] + self.icnn1_Wy_layers[i].weight)
            #print("check",self.icnn1_Wz_layers[i](h2))
            h1_n = torch.sigmoid(self.icnn1_Wz_layers[i](h2) + self.icnn1_Wy_layers[i](input)).view(-1,self.icnn1_Wy_layers[i].weight.size()[0],1)*(torch.clamp(self.icnn1_Wz_layers[i].weight, min=0.0, max=1e10)@h1 + self.icnn1_Wy_layers[i].weight)
            #h1[i+1] = torch.minimum(torch.ones(h2[i+1].size()).to(self.device),torch.exp(self.icnn1_Wz_layers[i](h2[i]) + self.icnn1_Wy_layers[i](input))).view(-1,self.icnn1_Wy_layers[i].weight.size()[0],1)*(torch.clamp(self.icnn1_Wz_layers[i].weight, min=0.0, max=1e10)@h1[i] + self.icnn1_Wy_layers[i].weight)
            #h1[i+1] = 2*(nn.ReLU()(self.icnn1_Wz_layers[i](h2[i]) + self.icnn1_Wy_layers[i](input))).view(-1,self.icnn1_Wy_layers[i].weight.size()[0],1)*(torch.clamp(self.icnn1_Wz_layers[i].weight, min=0.0, max=1e10)@h1[i] + self.icnn1_Wy_layers[i].weight)
            #h1_n = 0.5*torch.mul((torch.sign(self.icnn1_Wz_layers[i](h2) + self.icnn1_Wy_layers[i](input))+1).view(-1,self.icnn1_Wy_layers[i].weight.size()[0],1),(torch.matmul(torch.clamp(self.icnn1_Wz_layers[i].weight, min=0.0, max=1e10),h1) + self.icnn1_Wy_layers[i].weight))

            h1 = h1_n
            h2 = h2_n
        grad_icnn = h1.view(-1,self.contiz_dim) + 1/self.smooth*input
        icnn_output = h2 + 1/(2*self.smooth)*(torch.norm(input,dim=1)**2).view(-1,1)

        return grad_icnn, icnn_output

    def f2(self,input):

        #h1_2 = [[None] for i in range(self.px2_num_hidden_layers + 1)]
        #h2_2 = [[None] for i in range(self.px2_num_hidden_layers + 1)]
        #h2[0] = torch.pow(nn.ReLU()(self.icnn2_Wy0(input)),4)
        #h2[0] = torch.log(1+torch.exp(self.icnn2_Wy0(input)))
        #h2[0] = nn.ELU()((self.icnn2_Wy0(input)))
        #h2[0] = torch.pow(nn.ReLU()(self.icnn2_Wy0(input)),2)
        h2 = nn.ReLU()(self.icnn2_Wy0(input))

        #h1[0] = 4*torch.pow((nn.ReLU()(self.icnn2_Wy0(input))),3).view(-1,self.h_dim,1)*self.icnn2_Wy0.weight
        #h1[0] = torch.sigmoid((self.icnn2_Wy0(input)).view(-1,self.h_dim,1))*self.icnn2_Wy0.weight
        #h1[0] = torch.minimum(torch.ones(h2[0].size()),torch.exp(self.icnn2_Wy0(input))).view(-1,self.h_dim,1)*self.icnn2_Wy0.weight
        #h1[0] = 2*(nn.ReLU()(self.icnn2_Wy0(input))).view(-1,self.h_dim,1)*self.icnn2_Wy0.weight
        h1 = 0.5*(torch.sign(self.icnn2_Wy0(input))+1).view(-1,self.h_dim,1)*self.icnn2_Wy0.weight
        for i in range(self.px2_num_hidden_layers):
            #h2_2[i+1] = torch.pow(nn.ReLU()(self.icnn2_Wz_layers[i](h2_2[i]) + self.icnn2_Wy_layers[i](input)),4)
            #h2_2[i+1] = torch.log(1+torch.exp(self.icnn2_Wz_layers[i](h2_2[i]) + self.icnn2_Wy_layers[i](input)))
            #h2_2[i+1] = nn.ELU()(self.icnn2_Wz_layers[i](h2_2[i]) + self.icnn2_Wy_layers[i](input))
            #h2_2[i+1] = torch.pow(nn.ReLU()(self.icnn2_Wz_layers[i](h2_2[i]) + self.icnn2_Wy_layers[i](input)),2)
            h2_n = nn.ReLU()(self.icnn2_Wz_layers[i](h2) + self.icnn2_Wy_layers[i](input))

            #h1_2[i+1] = 4*torch.pow((nn.ReLU()(self.icnn2_Wz_layers[i](h2_2[i]) + self.icnn2_Wy_layers[i](input))),3).view(-1,self.icnn2_Wy_layers[i].weight.size()[0],1)*(torch.clamp(self.icnn2_Wz_layers[i].weight, min=0.0, max=1e5)@h1_2[i] + self.icnn2_Wy_layers[i].weight)
            #h1_2[i+1] = torch.sigmoid(self.icnn2_Wz_layers[i](h2_2[i]) + self.icnn2_Wy_layers[i](input)).view(-1,self.icnn2_Wy_layers[i].weight.size()[0],1)*(torch.clamp(self.icnn2_Wz_layers[i].weight, min=0.0, max=1e5)@h1_2[i] + self.icnn2_Wy_layers[i].weight)
            #h1_2[i+1] = torch.minimum(torch.ones(h2_2[i+1].size()),torch.exp(self.icnn2_Wz_layers[i](h2[i]) + self.icnn2_Wy_layers[i](input))).view(-1,self.icnn2_Wy_layers[i].weight.size()[0],1)*(torch.clamp(self.icnn2_Wz_layers[i].weight, min=0.0, max=1e5)@h1_2[i] + self.icnn2_Wy_layers[i].weight)
            #h1_2[i+1] = 2*(nn.ReLU()(self.icnn2_Wz_layers[i](h2_2[i]) + self.icnn2_Wy_layers[i](input))).view(-1,self.icnn2_Wy_layers[i].weight.size()[0],1)*(torch.clamp(self.icnn2_Wz_layers[i].weight, min=0.0, max=1e5)@h1_2[i] + self.icnn2_Wy_layers[i].weight)
            h1_n = 0.5*(torch.sign(self.icnn2_Wz_layers[i](h2) + self.icnn2_Wy_layers[i](input))+1).view(-1,self.icnn2_Wy_layers[i].weight.size()[0],1)*(torch.clamp(self.icnn2_Wz_layers[i].weight, min=0.0)@h1 + self.icnn2_Wy_layers[i].weight)

            h1 = h1_n
            h2 = h2_n

        grad_icnn = h1.view(-1,self.out_dim) + 1/self.smooth*input
        icnn_output = h2 + 1/(2*self.smooth)*(torch.norm(input,dim=1)**2).view(-1,1)

        return grad_icnn#, icnn_output
    def eval_lip(self):
        x = (10*torch.rand(2000,self.contiz_dim) - 1)
        model_output = self.f1(x.cuda())
        L=[]
        for i in range (5000):
            train_evalset = np.random.choice(np.arange(x.shape[0]), 2, replace=False)
            train_data = x[train_evalset]
            fx = model_output[train_evalset]
            L.append(torch.norm(fx[0]-fx[1]).cpu().detach().numpy()/np.linalg.norm((train_data[0]-train_data[1]).cpu().numpy()))
        true_lip = 1/np.min(L)
        true_invlip = 1/np.max(L)
        return true_invlip

    def empirical_lipschitz(self,x, eps=0.05):

        norms = lambda X: X.view(X.shape[0], -1).norm(dim=1) ** 2
        lip = 0.0
        L = []
        for r in range(10):
            dx = torch.zeros_like(x).cuda()
            dx.uniform_(-eps,eps)
            x.requires_grad = True
            dx.requires_grad = True
            optimizer = torch.optim.Adam([x, dx], lr=1E-1)
            iter, j = 0, 0
            LipMax = 0.0
            while j < 100:
                LipMax_1 = LipMax
                optimizer.zero_grad()
                dy = model.f1(x + dx) - model.f1(x)
                Lip = norms(dy) / (norms(dx))
                Obj = -Lip.sum()
                Obj.backward()
                optimizer.step()
                LipMax = Lip.max().item()
                iter += 1
                j += 1
                if j >= 5:
                    if LipMax < LipMax_1 + 1E-3:
                        optimizer.param_groups[0]["lr"] /= 10.0
                        j = 0

                    if optimizer.param_groups[0]["lr"] <= 1E-5:
                        break

            lip = max(lip, LipMax)

        return lip**0.5

    def empirical_invlipschitz(self,x, eps=0.05):

        norms = lambda X: X.view(X.shape[0], -1).norm(dim=1) ** 2
        lip = 100000000.0
        L = []
        for r in range(10):
            dx = torch.zeros_like(x).cuda()
            dx.uniform_(-eps,eps)
            x.requires_grad = True
            dx.requires_grad = True
            optimizer = torch.optim.Adam([x, dx], lr=1E-1)
            iter, j = 0, 0
            LipMax = 100000000.0
            while j < 100:
                LipMax_1 = LipMax
                optimizer.zero_grad()
                dy = model.f1(x + dx) - model.f1(x)
                Lip = norms(dy) / (norms(dx))
                Obj = Lip.sum()
                Obj.backward()
                optimizer.step()
                LipMax = Lip.min().item()
                iter += 1
                j += 1
                if j >= 5:
                    if LipMax > LipMax_1 - 1E-3:
                        optimizer.param_groups[0]["lr"] /= 10.0
                        j = 0

                    if optimizer.param_groups[0]["lr"] <= 1E-5:
                        break

            lip = min(lip, LipMax)

        return lip**0.5


    def forward(self, z, optim, invlip, lip):
        true_invlip = invlip
        with torch.no_grad():
            if optim == "GD":
                L_list=[]
                invL_list=[]
                obj = []
                z_clone = z.clone().cpu()
                x = torch.ones(z.size()).cuda()
                step = 2*self.smooth
                for i in range (self.max_iter):
                    L=[]
                    for j in range (5000):
                        train_evalset = np.random.choice(np.arange(z.shape[0]), 2, replace=False)
                        train_data = z[train_evalset]
                        fx = x[train_evalset]
                        L.append(torch.norm(fx[0]-fx[1]).cpu().detach().numpy()/np.linalg.norm((train_data[0]-train_data[1]).cpu().numpy()))
                    L_list.append(np.max(L))
                    invL_list.append(np.min(L))
                    grad = self.f1(x)
                    x = x + step/(i+1) * (z-grad)
                    obj.append((torch.mean(torch.norm(z-grad,dim=1))).cpu().detach().numpy())
            else:
                L_list=[]
                invL_list=[]
                obj = []
                learning_rate = 2*self.smooth
                x = torch.ones(z.size()).cuda()
                tol = 1e-12
                def closure():
                    with torch.no_grad():
                        grad,F = self.f1_with_output(x)
                        loss = torch.sum(F) - torch.sum(x * z)
                        x.grad = grad - z
                    return loss

                if optim == "Adam":
                    optimizer = torch.optim.Adam([x], lr=learning_rate, eps=tol)
                elif optim == "Adagrad":
                    optimizer = torch.optim.Adagrad([x], lr=learning_rate, eps=tol)
                elif optim =="RMSprop":
                    optimizer = torch.optim.RMSprop([x], lr=learning_rate, eps=tol)
                elif optim == "SC-Adam":
                    m = torch.zeros_like(self.f1(x))
                    v = torch.zeros_like(self.f1(x))
                    beta_1 = 0.9
                    beta_2 = 0.999
                elif optim == "SC-Adagrad":
                    v = torch.zeros_like(self.f1(x))
                elif optim =="SC-RMSprop":
                    v = torch.zeros_like(self.f1(x))
                    alpha = 0.99
                elif optim ==  "AGD":
                    y = torch.ones(z.size()).cuda()

                for i in range (self.max_iter):
                    L=[]
                    for j in range (5000):
                        train_evalset = np.random.choice(np.arange(z.shape[0]), 2, replace=False)
                        train_data = z[train_evalset]
                        fx = x[train_evalset]
                        L.append(torch.norm(fx[0]-fx[1]).cpu().detach().numpy()/np.linalg.norm((train_data[0]-train_data[1]).cpu().numpy()))
                    L_list.append(np.max(L))
                    invL_list.append(np.min(L))
                    if optim == "Newton":
                        hess = torch.linalg.inv(torch.func.vmap(torch.func.jacrev(self.f1))(x.view(-1,1,self.contiz_dim)).view(-1,self.contiz_dim,self.contiz_dim))
                        grad = self.f1(x)

                        x -= torch.bmm(hess,(grad-z).view(-1,self.contiz_dim,1)).view(-1,self.contiz_dim)
                        obj.append((torch.mean(torch.norm(z-grad,dim=1))).cpu().detach().numpy())

                    elif optim =="SC-Adagrad":
                        grad = self.f1(x)-z
                        v += grad**2
                        gm = learning_rate/(i+1)
                        x -= gm*torch.mul((v+tol)**(-1),grad)

                        with torch.no_grad():
                            grad = self.f1(x)
                            obj.append((torch.mean(torch.norm(z-grad,dim=1))).cpu().detach().numpy())

                    elif optim =="SC-Adam":

                        grad = self.f1(x)-z
                        beta_1 = 0.9*0.99**i
                        beta_2 = 1-1/(i+1)

                        m = beta_1*m + (1-beta_1)*grad
                        v = beta_2*v + (1-beta_2)*grad**2

                        gm = 2*self.smooth/(i+1)

                        hat_m = m/(1-beta_1**(i+1))
                        hat_v = v/(1-beta_2**(i+1))
                        x -= gm*torch.mul(((hat_v)**(1)+tol)**(-1),hat_m)


                        with torch.no_grad():
                            grad = self.f1(x)
                            obj.append((torch.mean(torch.norm(z-grad,dim=1))).cpu().detach().numpy())

                    elif optim == "SC-RMSprop":
                        grad = self.f1(x)-z
                        alpha = 1-1/(i+1)
                        v = alpha*v + (1-alpha)*grad**2

                        gm = 2*self.smooth/(i+1)

                        x -= gm*torch.mul(((v)**(1)+tol/(i+1))**(-1),grad)

                        with torch.no_grad():
                            grad = self.f1(x)
                            obj.append((torch.mean(torch.norm(z-grad,dim=1))).cpu().detach().numpy())
                        #print(optimizer.state_dict())
                    elif optim == "AGD":

                        step = 2*self.smooth/(i+1)
                        #step = invlip
                        Q = invlip*self.smooth

                        grad = self.f1(x)-z

                        y_bef = y
                        y = x-step*grad
                        x = (1-(Q**0.5-1)/(Q**0.5+1))*y+(Q**0.5-1)/(Q**0.5+1)*y_bef

                        with torch.no_grad():
                            grad = self.f1(x)
                            obj.append((torch.mean(torch.norm(z-grad,dim=1))).cpu().detach().numpy())

                    else:
                        if optim != "Adagrad":
                            for g in optimizer.param_groups:
                                #print("lr",g['lr'])
                                g['lr'] = 2*self.smooth/(i+1)

                        optimizer.step(closure)
                        with torch.no_grad():
                            grad = self.f1(x)
                            obj.append((torch.mean(torch.norm(z-grad,dim=1))).cpu().detach().numpy())
                        #print(optimizer.state_dict())

            #print(L_list)]
        np.save("evolution_Lipschitz"+optim+"_"+str(self.max_iter)+"_"+str(self.seed),L_list)
        np.save("evolution_invLipschitz"+optim+"_"+str(self.max_iter)+"_"+str(self.seed),invL_list)
        np.save("evolution_objective"+optim+"_"+str(self.max_iter)+"_"+str(self.seed),invL_list)

        plt.plot(L_list, "o",label="Estimated Lipschitz")
        plt.plot(np.ones_like(L_list)*self.smooth,"r",label="True Lipschitz")
        plt.legend()
        plt.xlabel("Iteration",fontsize=15)
        plt.yscale("log")
        plt.savefig("plot_evolution_Lipschitz"+optim+"_"+str(self.max_iter)+"_"+str(self.seed)+".png")
        plt.cla()
        plt.plot(invL_list,"o",label="Estimated inverse Lipschitz")
        plt.plot(np.ones_like(L_list)*true_invlip,"r",label="True inverse Lipschitz")
        plt.legend()
        plt.xlabel("Iteration",fontsize=15)
        plt.yscale("log")
        plt.savefig("plot_evolution_invLipschitz"+optim+"_"+str(self.max_iter)+"_"+str(self.seed)+".png")
        plt.cla()
        plt.plot(obj)
        #plt.legend()
        plt.xlabel("Iteration",fontsize=15)
        plt.yscale("log")
        plt.savefig("plot_evolution_objective"+optim+"_"+str(self.max_iter)+"_"+str(self.seed)+".png")
        plt.cla()
        torch.cuda.empty_cache()
        gc.collect()

        return x+self.convex*z



parser = ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('-m', '--model', type=str, default='Resnet',
                        help="[DNN, KWL, Resnet, Toy]")
parser.add_argument('-d', '--dataset', type=str, default='tiny_imagenet',
                        help="dataset [cifar10, cifar100, tiny_imagenet, square_wave, linear1, linear50]")
parser.add_argument('-g', '--gamma', type=float, default=10.0,
                    help="Network Lipschitz bound")
parser.add_argument('-s', '--seed', type=int, default=127)
parser.add_argument('-e','--epochs', type=int, default=100)

parser.add_argument('--layer', type=str, default='SLL')
parser.add_argument('--scale', type=str, default='xlarge')
parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate")
parser.add_argument('--loss', type=str, default='xent')
parser.add_argument('--root_dir', type=str, default='./saved_models')
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--LLN', action='store_true')
parser.add_argument('--normalized', action='store_true')
parser.add_argument('--cert_acc', action='store_true')

parser.add_argument('-smooth', '--smooth', \
            type=float, default=10.0)
parser.add_argument('-convex', '--convex', \
            type=float, default=1.0)
parser.add_argument('-hd', '--h_dim', \
            type=int, default=10)
parser.add_argument('-od', '--out_dim', \
            type=int, default=1)
parser.add_argument('-cd', '--contiz_dim', \
            type=int, default=2)
parser.add_argument('-nhl', '--num_hidden_layers', \
            type=int, default=2)
parser.add_argument('-ep', '--max_iter', \
            type=int, default=2500)
parser.add_argument('-comp', '--composite', \
            type=bool, default=False)
parser.add_argument('-eval', '--eval', \
            type=bool, default=False)
parser.add_argument('-resume', '--resume', \
            type=bool, default=False)
parser.add_argument('-brute', '--brute_force', \
            type=bool, default=False)

config = parser.parse_args()

seed_everything(config.seed)


config.in_channels = 2
config.img_size = 0
config.num_classes = 1
config.train_batch_size = 50
        #config.train_batch_size = 1
config.test_batch_size = 200
config.num_workers = 0

optimizers = ["GD","Adam", "Adagrad", "RMSprop", "Newton"]#,"SC-Adam", "SC-Adagrad", "SC-RMSprop"]
model = BLNN2(config).cuda()
size = 5000
x = (10*torch.rand(size,config.in_channels) - 1)
lip = model.empirical_lipschitz(x.cuda())
invlip = model.empirical_invlipschitz(x.cuda())


np.save("truelip"+"_"+str(config.max_iter)+"_"+str(config.seed),[lip])
np.save("trueinvlip"+"_"+str(config.max_iter)+"_"+str(config.seed),[invlip])

for optim in optimizers:
    with torch.no_grad():
        config.optim = optim
        print(optim)
        model(x.cuda(), config.optim, 1/lip, 1/invlip)
