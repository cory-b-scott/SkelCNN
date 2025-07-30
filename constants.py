import torch

LAYER_SIZES = [1,2,4,8,16,32]
GRID_SIZES = [1,2,4,8,16,32]
LEARNING_RATE = 1e-4
SKEL_LEARNING_RATE = 1e-2
BATCH_SIZE = 2048
BATCH_PER_EP = 10
EPOCHS = 150
lossfunc = torch.nn.CrossEntropyLoss()
dev = 'cuda'
