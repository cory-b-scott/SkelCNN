import torch
from constants import *
from skel_models import SkelConvModel
import sys

def build_skel_head():
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        SkelConvModel()
    )

def build_cnn_head():
    layerlist = []
    for i in range(len(LAYER_SIZES)-1):
        layerlist += [
            torch.nn.BatchNorm2d(LAYER_SIZES[i]),
            torch.nn.Conv2d(LAYER_SIZES[i],LAYER_SIZES[i+1],(3,3),stride=1,padding=1, bias=False),
            #torch.nn.Conv2d(LAYER_SIZES[i],LAYER_SIZES[i+1],(2,2),stride=2,padding=0, bias=False),
            torch.nn.SiLU(),
            torch.nn.AvgPool2d(2,stride=2)
        ]
    layerlist.append(torch.nn.Flatten())
    return torch.nn.Sequential(
        *layerlist
    )

def build_classifier():
    return torch.nn.Sequential(
        torch.nn.BatchNorm1d(LAYER_SIZES[-1]),
        torch.nn.Linear(LAYER_SIZES[-1],256),
        torch.nn.SiLU(),
        torch.nn.Dropout(.25),
        torch.nn.Linear(256,128),
        torch.nn.SiLU(),
        torch.nn.Linear(128,10)
    )

def build_prefix():
    return torch.nn.Sequential(
        torch.nn.Unflatten(1, (1,28,28)),
        torch.nn.BatchNorm2d(1),
        torch.nn.ZeroPad2d(2),
        #torch.nn.AvgPool2d(2)
    )

def model_param_size(mod):
    total = 0
    for param in mod.parameters():
        p = 1
        size = (tuple(param.shape))
        for v in size:
            p*= v
        total += p
    return total
    
if __name__ == '__main__':
    seed = int(sys.argv[1])
    torch.manual_seed(seed)

    mod = torch.nn.Sequential(
        build_prefix(),
        build_cnn_head(),
        build_classifier()
    ).to(dev)
    #print(mod(torch.rand((10,28*28),device=dev)).shape)
    print("Conv:", model_param_size(mod))

    mod = torch.nn.Sequential(
        build_prefix(),
        build_skel_head(),
        build_classifier()
    ).to(dev)
    print("SkelConv:", model_param_size(mod))
    #print(mod(torch.rand((10,28*28),device=dev)).shape)
    #print([item.shape for item in mod.parameters()])
    #print(mod)
    #print(mod[1])
