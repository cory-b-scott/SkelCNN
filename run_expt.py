from src.skel.glineage import *#GridGraphLineage, ParametrizedGraphLineage, CompleteGraphLineage, PathGraphLineage, CycleGraphLineage
from src.skel.skel import *
from src.skel.kron import *

from torch.nn import Linear
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.preprocessing import StandardScaler

from skimage.transform import resize

from time import time

import sys

from load_mnist import mnist

from constants import *
from build_models import *

from copy import deepcopy
from tqdm import tqdm

import random

def evaluate(mod, loader):
    total_loss = 0
    all_preds = []
    all_labels = []
    for b,(bX,bY) in enumerate(loader):
        with torch.no_grad():
            bX = bX.to(dev)
            bY = bY.to(dev)
            pred = mod(bX)
            loss = lossfunc(pred, bY)
            all_preds.append(torch.argmax(pred,1).cpu().numpy())
            all_labels.append(bY.cpu().numpy())
            total_loss += loss.detach()*bX.shape[0]

    loss = (total_loss.cpu().numpy()/np.concatenate(all_preds).shape[0])
    acc = (np.concatenate(all_preds) == np.concatenate(all_labels)).mean()
    return loss, acc

def train_model(mod, trX, trY, tsX, tsY):

    trainData = TensorDataset(torch.tensor(trX).float(), torch.tensor(trY))
    trainLoader = DataLoader(trainData, batch_size = BATCH_SIZE, shuffle=True, drop_last=False, num_workers=4)

    testData = TensorDataset(torch.tensor(tsX).float(), torch.tensor(tsY))
    testLoader = DataLoader(testData, batch_size = BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

    best_acc_so_far = 0.0
    best_model = None


    params = [{"params":mod[0].parameters(),"lr":LEARNING_RATE}, {"params":mod[-1].parameters(),"lr":LEARNING_RATE}]
    
    if MODEL_TYPE == "CNN":
        params.append({"params":mod[1].parameters(),"lr":LEARNING_RATE})
    else:
        params.append({"params":mod[1].parameters(),"lr":SKEL_LEARNING_RATE})
        
    opt = torch.optim.Adam(params, LEARNING_RATE)

    ep_loss, ep_acc = evaluate(mod, testLoader)
    reps = 0
    print(0.0, reps,  "%10f"%100.0, "%10f"% 0.0, "%10f"%ep_loss, "%10f"% ep_acc)
    t0 = time()

    for i in range(EPOCHS):
        mod.train()
        for b,(bX,bY) in enumerate(trainLoader):

            opt.zero_grad()
            bX = bX.to(dev)
            bY = bY.to(dev)
            pred = mod(bX)
            #print(all_preds)
            #print(bX.shape, bY.shape, pred.shape)
            loss = lossfunc(pred, bY)
            #param = list(mod[1][1].parameters())[5]

            loss.backward()
            opt.step()
            if b > BATCH_PER_EP:
                break
            #print(loss)
        reps += 1
        t1 = time()
        mod.eval()
        ep_loss, ep_acc = evaluate(mod, testLoader)
        if ep_acc > best_acc_so_far:
            best_acc_so_far = ep_acc
            best_model = deepcopy(mod)

        tr_loss, tr_acc = evaluate(mod, trainLoader)




        print(t1 - t0, reps,  "%10f"%tr_loss, "%10f"% tr_acc, "%10f"%ep_loss, "%10f"% ep_acc)
    return best_model

if __name__ == '__main__':
    # python run_expt.py 11111 CNN > results/MNIST/CNN_11111.txt
    # parallel -j 2 "python run_expt.py 5232025{1} {2} > results/MNIST/{2}_5232025{1}.txt" ::: {1..3} ::: {SkelCNN,CNN}
    # parallel -j 8 "python -u run_expt.py 1{}1{}1 CNN > results/MNIST/061825_CNN_{1}.txt" ::: {1..32}
    # parallel -j 8 "python -u run_expt.py 1{1}1{1}1 {2} {3} > results/{3}/061825_{2}_{1}.txt" ::: {1..4} ::: {SkelCNN,CNN} ::: MNIST FMNIST
    seed = int(sys.argv[1])

    if "cuda" in dev:
        dev = "cuda:%d" % (seed % 2)
        #torch.set_default_device('cuda')
        torch.cuda.set_device(dev) 
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    MODEL_TYPE = sys.argv[2]
    DATASET = sys.argv[3]

    trainvalidX, trainvalidY, testX, testY = mnist(path="/home/cory/"+DATASET)

    idx = np.arange(trainvalidX.shape[0])
    np.random.shuffle(idx)
    trainX = trainvalidX[idx[:50000]]
    trainY = trainvalidY[idx[:50000]]
    validX = trainvalidX[idx[50000:]]
    validY = trainvalidY[idx[50000:]]
    testX = testX


    #tdat = torch.tensor(trainX[:10])
    #pref = build_prefix()
    #test = build_model()

    #output = test(tdat)
    #print(output.shape)
    #quit()

    prefix = build_prefix()
    if MODEL_TYPE == 'CNN':
        head = build_cnn_head()
    elif MODEL_TYPE == 'SkelCNN':
        head = build_skel_head()
    else:
        raise Exception("Bad Model Name")

    classifier = build_classifier()
    model = torch.nn.Sequential(prefix,head,classifier).to(dev)
    #print(model[1][0].weight)
    #quit()
    #print(torch.seed())
    #quit()

    best = train_model(model, trainX, trainY, validX, validY)


    testData = TensorDataset(torch.tensor(testX).float(), torch.tensor(testY))
    testLoader = DataLoader(testData, batch_size = BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)

    best.eval()
    final_loss, final_acc = evaluate(model, testLoader)
    print("X", "TEST", "X", "X", final_loss, final_acc)
    #output = test(tdat)
    #print(output.shape)

    #print([item.shape for item in [trainX, trainY, testX, testY]])
