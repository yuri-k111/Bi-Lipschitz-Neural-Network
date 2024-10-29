import torch
import torch.utils.data
from torch.nn import functional as F
from torch import nn, Tensor
import time
import torch.nn.init as init

#from https://github.com/yixinwang/lidvae-public
# To implement Theorem 3.7 for more efficient backpropagation, we recommend to incorporate this skeleton into the DEQ framework https://github.com/locuslab/deq
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
        return nn.functional.linear(input, torch.clamp(self.weight, min=0.0, max=1e10))

class BLNN(nn.Module):
    """
    Bi-Lipschitz Neural network
    """
    def __init__(self,args):

        super(BLNN, self).__init__()
        self.contiz_dim = args.contiz_dim #input dim
        self.h_dim = args.h_dim #hidden dim

        self.qy_layers = []
        self.device = torch.device("cuda")
        self.px1_num_hidden_layers = args.num_hidden_layers1
        self.px2_num_hidden_layers = args.num_hidden_layers2
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


        self.smooth= args.smooth
        self.convex= args.convex

    def f1(self, input, with_output=False, create_graph = True):

        # The core ICNN with regularization term at the output


        with torch.enable_grad():
            #h2[0] = torch.pow(nn.ReLU()(self.icnn1_Wy0(input)),4)
            input.requires_grad_(True)
            h2 = nn.Softplus()(self.icnn1_Wy0(input))
            #h2 = nn.ELU()((self.icnn1_Wy0(input))) #ELU
            #h2 = torch.pow(nn.ReLU()(self.icnn1_Wy0(input)),2) #ReLU squared
            #h2 = nn.ReLU()(self.icnn1_Wy0(input)) #ReLU

            for i in range(self.px1_num_hidden_layers):
                #h2[i+1] = torch.pow(nn.ReLU()(self.icnn1_Wz_layers[i](h2[i]) + self.icnn1_Wy_layers[i](input)),4)
                h2_n = nn.Softplus()(self.icnn1_Wz_layers[i](h2) + self.icnn1_Wy_layers[i](input))
                #h2_n = nn.ELU()(self.icnn1_Wz_layers[i](h2) + self.icnn1_Wy_layers[i](input))
                #h2_n = torch.pow(nn.ReLU()(self.icnn1_Wz_layers[i](h2) + self.icnn1_Wy_layers[i](input)),2)
                #h2_n= nn.ReLU()(self.icnn1_Wz_layers[i](h2) + self.icnn1_Wy_layers[i](input))


                h2 = h2_n

            icnn_output = h2 + 1/(2*self.smooth)*(torch.norm(input,dim=1)**2).view(-1,1)
            grad_icnn = torch.autograd.grad(icnn_output, [input], torch.ones_like(icnn_output), create_graph=create_graph)[0]

            if with_output:
                return grad_icnn, icnn_output
            else:
                return grad_icnn



    def legendre(self, z, id = None, eval=False):

        # Legendre-Fenchel transformation

        x1 = torch.ones(z.size()).cuda()
        step = 2*self.smooth
        if eval == True:
            max_it = 5000
        else:
            max_it = 500
        for i in range (max_it):
            grad = self.f1(x1)
            x1 = x1 + step/(i+1) * (z-grad)
            if torch.mean(torch.norm(z-grad,dim=1))<0.001:
                break
        return x1+self.convex*z

    def forward(self, x, id=None, extended=False):
        output = self.legendre(x, id)

        return output
class PBLNN(nn.Module):
    def __init__(self, dimx, dimy, dimh, num_hidden_layers, smooth, convex):
        super(PBLNN, self).__init__()

        self.dimx = dimx
        self.dimy = dimy
        self.dimh = dimh

        self.out_dim = 101
        self.composite = True
        self.device = torch.device("cuda")
        self.px1_num_hidden_layers = num_hidden_layers
        self.px2_num_hidden_layers = num_hidden_layers

        self.smooth = smooth
        self.convex = convex

        self.act = nn.Softplus()

        Wzs = list()
        for _ in range(num_hidden_layers - 1):
            Wzs.append(PositiveLinear(dimh, dimh))
        Wzs.append(PositiveLinear(dimh, 1))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wzus = list()
        for _ in range(num_hidden_layers - 1):
            Wzus.append(nn.Linear(dimh, dimh, bias=True))
        Wzus.append(nn.Linear(dimh, 1, bias=True))
        self.Wzus = torch.nn.ModuleList(Wzus)

        Wys = list()
        Wys.append(nn.Linear(dimy, dimh, bias=False))
        for _ in range(num_hidden_layers - 1):
            Wys.append(nn.Linear(dimy, dimh, bias=False))
        Wys.append(nn.Linear(dimy, 1, bias=False))
        self.Wys = torch.nn.ModuleList(Wys)

        Wyus = list()
        Wyus.append(nn.Linear(dimx, dimy, bias=True))
        for _ in range(num_hidden_layers - 1):
            Wyus.append(nn.Linear(dimh, dimy, bias=True))
        Wyus.append(nn.Linear(dimh, dimy, bias=True))
        self.Wyus = torch.nn.ModuleList(Wyus)

        Wus = list()
        Wus.append(nn.Linear(dimx, dimh, bias=True))
        for _ in range(num_hidden_layers - 1):
            Wus.append(nn.Linear(dimh, dimh, bias=True))
        Wus.append(nn.Linear(dimh, 1, bias=True))
        self.Wus = torch.nn.ModuleList(Wus)

        Wuus = list()
        Wuus.append(nn.Linear(dimx, dimh, bias=True))
        for _ in range(num_hidden_layers - 1):
            Wuus.append(nn.Linear(dimh, dimh, bias=True))
        self.Wuus = torch.nn.ModuleList(Wuus)

        self.convert = nn.Linear(self.dimy, self.out_dim).requires_grad_(False)

        self.init_points1 = torch.ones((600000,dimy)).to(self.device)
        self.init_points2 = torch.ones((600000,1)).to(self.device)

    def f(self, x, y):
        if pr == True:
            print("x",x)
            print("y",y)
        with torch.enable_grad():

            y.requires_grad_(True)

            prevZ, prevU = None, x
            u = self.act(self.Wuus[0](prevU))
            yu_u = self.Wyus[0](prevU)
            z_yu = self.Wys[0](y * yu_u)
            z_u = self.Wus[0](prevU)
            z = self.act(z_yu+z_u)

            prevZ = z
            prevU = u
            for Wz, Wzu, Wy, Wyu, Wu, Wuu in zip(
                    self.Wzs[:-1], self.Wzus[:-1],
                    self.Wys[1:-1], self.Wyus[1:-1], self.Wus[1:-1],
                    self.Wuus[1:-1]):
                u = self.act(Wuu(prevU))

                zu_u = self.act(Wzu(prevU))
                z_zu = Wz(prevZ * zu_u)

                yu_u = Wyu(prevU)
                z_yu = Wy(y * yu_u)

                z_u = Wu(prevU)

                z = self.act(z_zu+z_yu+z_u)

                prevU = u
                prevZ = z

            zu_u = self.act(self.Wzus[-1](prevU))
            z_zu = self.Wzs[-1](prevZ * zu_u)

            yu_u = self.Wyus[-1](prevU)
            z_yu = self.Wys[-1](y * yu_u)

            z_u = self.Wus[-1](prevU)

            z = (z_zu+z_yu+z_u)


            icnn_output = z + 1/(2*self.smooth)*(torch.norm(y,dim=1)**2).view(-1,1)
            grad_icnn = torch.autograd.grad(icnn_output, [y], torch.ones_like(icnn_output), create_graph=True)[0]
        return grad_icnn

    def legendre(self, x, y, id = None, eval=False):
        if id == None:
            y1 = torch.ones(y.size()).cuda()
        else:
            y1 = self.init_points1[id]
        step = 2*self.smooth
        if eval == True:
            max_it = 5000
        else:
            max_it = 500
        for i in range (max_it):
            grad = self.f(x,y1, pr=pr)

            y1 = y1 + step/(i+1) * (y-grad)
            if torch.mean(torch.norm(y-grad,dim=1))<0.001:
                break
        if id != None:
            self.init_points1[id] = y1.clone().detach()

        if self.composite == True:

            y2 = torch.mean(y1+self.convex*y, dim = 1)

        return y1+self.convex*y, y2.view(-1,1)

    def forward(self, x, y, id=None, pr=False ):
        f_ast1, f_ast2 = self.legendre(x, y, id, pr)

        return f_ast2
