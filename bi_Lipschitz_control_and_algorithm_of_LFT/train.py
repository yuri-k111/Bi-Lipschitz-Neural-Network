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
#print(torch.cuda.is_available())
def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())


def get_gpu_utilization():
    cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader"
    utilization = subprocess.check_output(cmd, shell=True)
    utilization = utilization.decode("utf-8").strip().split("\n")
    utilization = [int(x.replace(" %", "")) for x in utilization]
    return utilization

def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)


def train_toy(config):
    print("layer:",config.layer)
    print("gamma:",config.gamma)
    seed_everything(config.seed)
    trainLoader, testLoader = getDataLoader(config)
    model = getModel(config).cuda()
    criterion = getLoss(config)
    #for name,param in model.named_parameters():
    #for param in model.parameters():
        #print(name, param.data)
        #print(param.requires_grad)
        #print(param)
    txtlog = TxtLogger(config)
    # wanlog = WandbLogger(config)

    txtlog(f"Set global seed to {config.seed:d}")

    nparams = np.sum([p.numel() for p in model.parameters() if p.requires_grad])

    if nparams >= 1000000:
        txtlog(f"name: {config.model}-{config.layer}-{config.scale}, num_params: {1e-6*nparams:.1f}M")
    else:
        txtlog(f"name: {config.model}-{config.layer}-{config.scale}, num_params: {1e-3*nparams:.1f}K")

    Epochs = config.epochs
    Lr = config.lr
    steps_per_epoch = len(trainLoader)
    gamma = config.gamma

    opt = torch.optim.Adam(model.parameters(), lr=Lr, weight_decay=0)
    lr_schedule = lambda t: np.interp([t], [0, Epochs*2//5, Epochs*4//5, Epochs], [0, Lr, Lr/20.0, 0])[0]
    data = torch.load("./data/"+config.dataset+".pt")
    xt = data["xt"]
    yt = data["yt"]
    scale = 1
    if config.resume == False:
        L_list=[]
        gam_list = []
        for epoch in range(Epochs):
            ## IF you want to use annealing, turn on the following part
            '''with torch.no_grad():
                gam = empirical_lipschitz(model, xt.cuda(), config, scale)
                print(gam)
                L_list.append(gam)
                gam_list.append(model.gamma*scale)
                print(abs(gam -model.gamma*scale))
                if abs(gam -model.gamma*scale)<=5e-2 and epoch%5==0:
                    print("changing Lip")
                    if config.layer == "legendre":
                        model.gamma=model.gamma*1.5
                        model.smooth=model.gamma-model.convex
                    else:
                        scale = scale*1.5'''

            n, Loss = 0, 0.0
            model.train()
            start0 = time.time()
            for batch_idx, batch in enumerate(trainLoader):
                opt.zero_grad()
                x, y = batch[0].cuda(), batch[1].cuda()
                lr = lr_schedule(epoch + (batch_idx+1)/steps_per_epoch)
                opt.param_groups[0].update(lr=lr)
                if config.layer == "Plain":
                    x = x*config.gamma
                yh = model(scale*x)
                J = criterion(yh, y)
                J.backward()

                opt.step()

                loss = J.item()
                n += y.size(0)
                Loss += loss * y.size(0)
            train_loss = Loss/n
            end = time.time()
            model(torch.rand((1,x.shape[1])).to(x.device))

            n, Loss = 0, 0.0,
            model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(testLoader):
                    x, y = batch[0].cuda(), batch[1].cuda()
                    if config.layer == "Plain":
                        x = x*config.gamma
                    yh = model(scale*x)
                    Loss += criterion(yh, y).item() * y.size(0)
                    n += y.size(0)

            test_loss = Loss/n

            txtlog(f"Epoch: {epoch+1:3d} | loss: {train_loss:.2f}/{test_loss:.2f}, 100lr: {100*lr:.3f}")

            if epoch % config.save_freq == 0 or epoch + 1 == Epochs:
                torch.save(model.state_dict(), f"{config.train_dir}/model_4.ckpt")
    else:
    # after training
        chpt = torch.load(f"{config.train_dir}/model_4.ckpt")
        model.load_state_dict(chpt)

    xshape = (config.lip_batch_size, config.in_channels)
    x = (torch.rand(xshape) + 0.3*torch.randn(xshape)).cuda()

    data = torch.load("./data/"+config.dataset+".pt")

    xt = data["xt"]
    yt = data["yt"]
    if config.layer == "legendre":
        y_pred = model(xt.cuda(),eval=True)
    if config.layer == "Plain":
        y_pred = model(config.gamma*xt.cuda())
    else:
        y_pred = model(scale*xt.cuda())
    print("plotting...")
    plt.plot(xt.cpu().detach().numpy(),yt,label="True")
    plt.scatter(xt.cpu().detach().numpy(),y_pred.cpu().detach().numpy(), label="Learned")
    plt.legend()
    plt.plot()
    print("result4"+str(config.seed)+"_"+str(config.layer)+"_"+config.dataset+"_"+str(config.gamma)+"_"+str(config.h_dim)+".png")
    plt.savefig("result4"+str(config.seed)+"_"+str(config.layer)+"_"+config.dataset+"_"+str(config.gamma)+"_"+str(config.h_dim)+".png")
    print("calculating Lipschitz")
    gam = empirical_lipschitz(model, xt.cuda(), config)
    #print(gam)
    if config.layer == "Plain":
        gam = empirical_lipschitz(model, xt.cuda(), config)*config.gamma
    with open("result4"+str(config.seed)+"_"+str(config.layer)+"_"+config.dataset+"_"+str(config.gamma)+"_"+str(config.h_dim)+'.log', 'w') as f:
        f.write(f"Lipschitz capcity: {gam:.4f}/{gamma:.2f}, {100*gam/gamma:.2f}")
        f.write("\n")

    if model.gamma is None:
        txtlog(f"Lipschitz: {gam:.2f}/--")
    else:
        txtlog(f"Lipschitz capcity: {gam:.4f}/{gamma:.2f}, {100*gam/gamma:.2f}")
