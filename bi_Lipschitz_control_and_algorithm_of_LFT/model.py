import torch
import torch.nn as nn
from layer import *
import gc
def getModel(config):
    models = {
        'DNN': DNN,
        'KWL': KWL,
        'Resnet': Resnet,
        'Toy': Toy
    }[config.model]
    if config.layer == "legendre":
        return BLNN(config)
    else:
        return models(config)

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


class DNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma
        self.LLN = config.LLN
        self.layer = config.layer
        self.scale = config.scale
        g = self.gamma ** 0.5
        if self.img_size == 32:
            n = 32 // 8
            layer = [
                SandwichConv(self.in_channels, 64, 3, scale=g),
                SandwichConv(64, 64, 3, stride=2),
                SandwichConv(64, 128, 3),
                SandwichConv(128, 128, 3, stride=2),
                SandwichConv(128, 256, 3),
                SandwichConv(256, 256, 3, stride=2),
                nn.Flatten(),
                SandwichFc(256 * n * n, 2048),
                SandwichFc(2048, 2048)
            ]
        elif self.img_size == 64:
            n = 64 // 16
            layer = [
                SandwichConv(self.in_channels, 64, 3, scale=g),
                SandwichConv(64, 64, 3, stride=2),
                SandwichConv(64, 128, 3),
                SandwichConv(128, 128, 3, stride=2),
                SandwichConv(128, 256, 3),
                SandwichConv(256, 256, 3, stride=2),
                SandwichConv(256, 512, 3),
                SandwichConv(512, 512, 3, stride=2),
                nn.Flatten(),
                SandwichFc(512 * n * n, 2048),
                SandwichFc(2048, 2048)
            ]
        if self.LLN:
            layer.append(SandwichFc(2048, 1024, scale=g))
            layer.append(LinearNormalized(1024, self.num_classes))
        else:
            layer.append(SandwichFc(2048, 1024))
            layer.append(SandwichLin(1024, self.num_classes, scale=g))

        self.model = nn.Sequential(*layer)

    def forward(self, x):
        return self.model(x)


#------------------------------------------------------------
class KWL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma
        self.LLN = config.LLN
        self.width = config.width
        self.layer = config.layer

        w = self.width
        if self.gamma is not None:
            g = self.gamma ** (1.0 / 2)
        n = self.img_size // 4

        if self.layer == 'Plain':
            self.model = nn.Sequential(
                PlainConv(self.in_channels, 32 * w, 3), nn.ReLU(),
                PlainConv(32 * w, 32 * w, 3, stride=2), nn.ReLU(),
                PlainConv(32 * w, 64 * w, 3), nn.ReLU(),
                PlainConv(64 * w, 64 * w, 3, stride=2), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * n * n * w, 640 * w), nn.ReLU(),
                nn.Linear(640 * w, 512 * w), nn.ReLU(),
                nn.Linear(512 * w, self.num_classes)
            )
        elif self.layer == 'Sandwich':
            self.model = nn.Sequential(
                SandwichConv(self.in_channels, 32 * w, 3, scale=g),
                SandwichConv(32 * w, 32 * w, 3, stride=2),
                SandwichConv(32 * w, 64 * w, 3),
                SandwichConv(64 * w, 64 * w, 3, stride=2),
                nn.Flatten(),
                SandwichFc(64 * n * n * w, 512 * w),
                SandwichFc(512 * w, 512 * w),
                SandwichLin(512 * w, self.num_classes, scale=g)
            )
        elif self.layer == 'Orthogon':
            self.model = nn.Sequential(
                    OrthogonConv(self.in_channels, 32 * w, 3, scale=g),
                    OrthogonConv(32 * w, 32 * w, 3, stride=2),
                    OrthogonConv(32 * w, 64 * w, 3),
                    OrthogonConv(64 * w, 64 * w, 3, stride=2),
                    nn.Flatten(),
                    OrthogonFc(64 * n * n * w, 640 * w),
                    OrthogonFc(640 * w, 512 * w),
                    OrthogonLin(512 * w, self.num_classes, scale=g)
                )
        elif self.layer == 'Aol':
            self.model = nn.Sequential(
                AolConv(self.in_channels, 32 * w, 3, scale=g),
                AolConv(32 * w, 32 * w, 3),
                AolConv(32 * w, 64 * w, 3),
                AolConv(64 * w, 64 * w, 3),
                nn.AvgPool2d(4, divisor_override=4),
                nn.Flatten(),
                AolFc(64 * n * n * w, 640 * w),
                AolFc(640 * w, 512 * w),
                AolLin(512 * w, self.num_classes, scale=g)
            )

    def forward(self, x):
        return self.model(x)

# #---------------------------------------------------------------------------
class Resnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gamma = config.gamma
        self.depth_conv = config.depth_conv
        self.n_channels = config.n_channels
        self.depth_linear = config.depth_linear
        self.n_features = config.n_features
        self.conv_size = config.conv_size
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.LLN = config.LLN

        if self.gamma is None:
            g = 1.0
        else:
            g = self.gamma ** (1.0/2)

        layers = []
        self.conv1 = PaddingChannels(self.in_channels, self.n_channels, scale=g)
        for _ in range(self.depth_conv):
            layers.append(SLLBlockConv(self.n_channels, self.conv_size, 3))
        layers.append(nn.AvgPool2d(4, divisor_override=4))

        self.stable_block = nn.Sequential(*layers)

        layers_linear = [nn.Flatten()]

        in_features = self.n_channels * ((self.img_size // 4) ** 2)

        for _ in range(self.depth_linear):
            layers_linear.append(SLLBlockFc(in_features, self.n_features))
        self.layers_linear = nn.Sequential(*layers_linear)

        if self.LLN:
            self.last_last = LinearNormalized(in_features, self.num_classes, scale=g)
        else:
            self.last_last = FirstChannel(self.num_classes, scale=g)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stable_block(x)
        x = self.layers_linear(x)
        x = self.last_last(x)
        return x

class Toy(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.gamma = config.gamma
        self.layer = config.layer
        self.g = (self.gamma ** 0.5)
        w = config.h_dim
        if self.layer == 'Plain':
            #self.gamma = 50
            #w = 128
            self.model = nn.Sequential(
                torch.nn.utils.parametrizations.spectral_norm(nn.Linear(self.in_channels, w)), nn.ReLU(),
                torch.nn.utils.parametrizations.spectral_norm(nn.Linear(w, w)), nn.ReLU(),
                torch.nn.utils.parametrizations.spectral_norm(nn.Linear(w, w)), nn.ReLU(),
                torch.nn.utils.parametrizations.spectral_norm(nn.Linear(w, self.num_classes))
            )
        elif self.layer == 'Sandwich':
            #w = 86
            #w = 50
            #print(w)
            self.model = nn.Sequential(
                SandwichFc(self.in_channels, w, scale=self.g),
                SandwichLin(w, self.num_classes, scale=self.g)
            )
        elif self.layer == 'Orthogon':
            #w = 128
            #w = 32
            self.model = nn.Sequential(
                OrthogonFc(self.in_channels, w, scale=self.g),
                OrthogonFc(w, w),
                OrthogonFc(w, w),
                OrthogonLin(w, self.num_classes, scale=self.g)
            )
        elif self.layer == 'Aol':
            #w = 128
            #w = 32
            self.model = nn.Sequential(
                AolFc(self.in_channels, w, scale=self.g),
                AolFc(w, w),
                AolFc(w, w),
                AolLin(w, self.num_classes, scale=self.g)
            )
        elif self.layer == 'SLL':
            #w = 128
            #w = 7
            self.model = nn.Sequential(
                PaddingFeatures(self.in_channels, w, scale=self.g),
                SLLBlockFc(w, w),
                SLLBlockFc(w, w),
                FirstChannel(self.num_classes, scale=self.g)
            )
        elif self.layer == 'bilipnet':
            self.model = nn.Sequential(
            MonLipLayer(self.in_channels,[w,w],0,self.g**2)
            )

    def forward(self, x):
        return self.model(x)

class BLNN(nn.Module):
    """
    Bi-Lipschitz Neural network with alpha = 0
    """
    def __init__(self,config):
        super().__init__()

        self.contiz_dim = config.in_channels
        self.h_dim = config.h_dim
        self.out_dim = config.num_classes
        self.smooth=config.gamma-config.convex
        self.gamma=config.gamma
        self.convex=config.convex
        self.composite = config.composite
        self.brute_force = config.brute_force
        self.device = torch.device("cuda")#should i change this ????
        self.px1_num_hidden_layers = config.num_hidden_layers
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

        #self.icnn1_Wy_layers = icnn1_Wy_layers
        #self.icnn1_Wz_layers = icnn1_Wz_layers

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

            h1 = h1_n
            h2 = h2_n
        grad_icnn = h1.view(-1,self.contiz_dim) + 1/self.smooth*input
        icnn_output = h2 + 1/(2*self.smooth)*(torch.norm(input,dim=1)**2).view(-1,1)

        return grad_icnn, icnn_output

    def f2(self,input):

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

        #print(grad_icnn)
        return grad_icnn#, icnn_output


    def forward(self, z, eval=False):
        #with torch.no_grad():
        if self.brute_force==False:
            x = torch.ones(z.size()).cuda()
            step = 2*self.smooth
            def closure():
                grad,F = self.f1_with_output(x)
                loss = torch.sum(F) - torch.sum(x * z)
                x.grad = grad - z
                return loss
            #optimizer = torch.optim.Adagrad([x],lr=learning_rate,eps=tol)


            if eval == True:
                max_it = 5000#this seems to change memory usage why?
            else:
                max_it = 1000
            for i in range (max_it):
                #grad, fx =self.f1(x)
                grad =self.f1(x)
                x = x + step/(i+1) * (z-grad)
                #optimizer.step(closure)
                if torch.mean(torch.norm(z-grad,dim=1))<0.001:
                    break
                #gc.collect()
                #print(torch.cuda.memory_summary(device=None, abbreviated=False))

            if self.composite == True:
                z = (x+self.convex*z)@(torch.eye(self.contiz_dim,self.out_dim).cuda())
                x = torch.ones(z.size()).cuda()

                for i in range (max_it):
                    #grad, fx =self.f2(x)
                    grad =self.f2(x)
                    x = x + step/(i+1) * (z-grad)
                    if torch.mean(torch.norm(z-grad,dim=1))<0.001:
                        break
        else:
            #print("brute force False")
            learning_rate = 2*self.smooth
            x = z.clone().detach()
            if eval == True:
                max_iter = 1000000
            else:
                max_iter = 2000
            tol = 1e-12
            #tol = 1e-3
            def closure():
                with torch.no_grad():
                    grad,F = self.f1_with_output(x)
                    loss = torch.sum(F) - torch.sum(x * z)
                    x.grad = grad - z
                return loss

            optimizer = torch.optim.LBFGS([x], lr=learning_rate, line_search_fn="strong_wolfe", max_iter=max_iter, tolerance_grad=tol, tolerance_change=tol)


            optimizer.step(closure)
            gc.collect()
            #print(torch.cuda.memory_summary(device=None, abbreviated=False))


        return x+self.convex*z
