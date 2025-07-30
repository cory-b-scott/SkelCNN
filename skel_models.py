from src.skel.glineage import *
from src.skel.skel import *
from src.skel.kron import *

import torch
from constants import *

from time import time

import networkx as nx

import matplotlib.pyplot as plt

def gen_grid_gmat(size):
    if size==1:
        return torch.ones((1,1)).to_sparse()
    #print(size)
    #return torch.eye(size**2).to_sparse()
    gr = nx.grid_graph((size, size))
    for i in range(1,size-1):
        for j in range(1,size-1):
            gr.add_edge((i,j),(i+1,j+1))
            gr.add_edge((i,j),(i+1,j-1))
            gr.add_edge((i,j),(i-1,j+1))
            gr.add_edge((i,j),(i-1,j-1))

    #if size > 1:
    #    gr.add_edge((1,0),(0,1))
    #for i in range(2,size-2):
    #    for j in range(2,size-2):
    #        gr.add_edge((i-1,j-1),(i+1,j+1))
    #        gr.add_edge((i-1,j+1),(i+1,j-1))
    #nx.draw_networkx(gr)
    #plt.show()

    lapl = -nx.laplacian_matrix(gr)
    
    #lapl *= (.25 - .5*np.random.random(lapl.shape))
    im = scipy_spr_to_torch_sparse(lapl.tocoo())
    if True:#size == 1:
        im = im + 1.5*torch.eye(size**2).to_sparse()
    #print(im)
    #im = im * .185
    return im.coalesce().float().to_sparse()

def gen_grid_pmat(sizeA, sizeB):
    #print(sizeB, sizeA)
    #quit()
    pA = torch_sparse_kron(
        torch.eye(sizeA).to_sparse(),
        (1/np.sqrt(2) * torch.ones((sizeB//sizeA),1)).to_sparse()
    )
    pB = torch_sparse_kron(
        torch.eye(sizeA).to_sparse(),
        (1/np.sqrt(2) * torch.ones((sizeB//sizeA),1)).to_sparse()
    )
    #print(pA.shape)
    #orig = (.05 - .1*torch.rand((sizeA*sizeA, sizeB * sizeB)))
    #orig[torch.abs(orig) > .03] = 0
    orig = torch_sparse_kron(pA, pB).T
    #plt.matshow(orig.to_dense().detach().cpu().numpy())
    #plt.show()
    #orig = orig / orig.sum(0, keepdims=True)
    orig = orig.coalesce()
    #print(orig.shape)
    return orig.to_sparse()

def gen_path_pmat(size):
    vector = np.zeros(size+2)
    vector[:4] = 1
    mat = np.stack([np.roll(vector, 2*i) for i in range(size//2)]).T
    mat *= 1-2*np.random.random(mat.shape)
    #mat *= 1/np.sqrt(3)
    return torch.tensor(mat[1:-1,:]).float().to_sparse()

def gen_path_gmat(size):    
    vector = np.zeros(size+2)
    vector[:4] = 1
    vector *= np.random.random(vector.shape)
    #print(vector)
    #quit()
    mat = np.stack([np.roll(vector, i) for i in range(size+2)]).T
    #mat *= 1-2*np.random.random(mat.shape)
    #print(mat)
    #mat *= 1/np.sqrt(3)
    return torch.tensor(mat[1:-1,:][:,1:-1]).float().to_sparse()
    

def gen_feat_gmat(size):
    if size==1:
        return torch.ones((1,1)).to_sparse()
    #gr = nx.erdos_renyi_graph(size, .5)
    #print(old, orig)
    #lapl = nx.to_scipy_sparse_array(gr)
    #print(lapl)
    lapl = np.ones((size,size))
    #lapl = lapl*(1/size)*(1 - 2*np.random.random(lapl.shape))
    lapl = lapl*(1 - 2*np.random.random(lapl.shape))
    #print(lapl)
    #im = torch.tensor(.125 - .25*np.random.random((size, size))).to_sparse()
    #im = scipy_spr_to_torch_sparse(lapl.tocoo())
    #im = im + (torch.eye(size)).to_sparse()
    #np.fill_diagonal(lapl, 1.0)
    im = torch.tensor(lapl).to_sparse()
    #plt.matshow(im.to_dense().detach().cpu().numpy())
    #plt.show()
    return im.coalesce().float().to_sparse()

def gen_feat_pmat(sA, sB):
    if False:#sA > 1:
        #orig = torch_sparse_kron(
        #    torch.eye(sA).to_sparse(),
        #    (1/np.sqrt(sB//sA) * torch.ones((sB//sA),1)).to_sparse()
        #).T.coalesce()
        #orig = (.5 - torch.rand((sB, sB)))
        orig = (.5 - torch.rand((sA, sB)))
        #orig[torch.rand((sB, sB)) > .15] = 0
        #orig = torch.qr(orig)[0].t()[:sA]
        #plt.matshow(orig.to_dense().numpy())
        #plt.show()
    else:
        #orig = 1/np.sqrt(sB*sA)*(1-2*torch.rand((sA, sB)))
        orig = (1-2*torch.rand((sA, sB)))
    #orig *= .1
    #print(orig)
    return orig.to_sparse()

class SkelConvModel(torch.nn.Module):

    def __init__(self):
        super(SkelConvModel, self).__init__()

        
        self.grid_gl = [gen_grid_gmat(GRID_SIZES[i]) for i in range(len(GRID_SIZES))]
        self.grid_gr_idxs = [item.indices() for item in self.grid_gl]
        self.grid_gr_params = torch.nn.ParameterList([torch.nn.Parameter(item.values()) for item in self.grid_gl])
        #self.grid_gr_params = [item.values() for item in self.grid_gl]
        """
        self.grid_pl = [gen_grid_pmat(GRID_SIZES[i],GRID_SIZES[i+1]) for i in range(len(GRID_SIZES)-1)]
        self.grid_pm_idxs = [item.indices() for item in self.grid_pl]
        self.grid_pm_params = torch.nn.ParameterList([torch.nn.Parameter(item.values()) for item in self.grid_pl])
        #self.grid_pm_params = [item.values() for item in self.grid_pl]
        

        self.path_gl_A = [gen_path_gmat(GRID_SIZES[i]) for i in range(len(GRID_SIZES))]
        self.path_gl_A_idxs = [item.indices() for item in self.path_gl_A]
        self.path_gl_A_params = torch.nn.ParameterList([torch.nn.Parameter(item.values()) for item in self.path_gl_A])

        self.path_gl_B = [gen_path_gmat(GRID_SIZES[i]) for i in range(len(GRID_SIZES))]
        self.path_gl_B_idxs = [item.indices() for item in self.path_gl_B]
        self.path_gl_B_params = torch.nn.ParameterList([torch.nn.Parameter(item.values()) for item in self.path_gl_B])           """
        
        self.path_pl_A = [gen_path_pmat(GRID_SIZES[i+1]) for i in range(len(GRID_SIZES)-1)]
        self.path_pl_A_idxs = [item.indices() for item in self.path_pl_A]
        self.path_pl_A_params = torch.nn.ParameterList([torch.nn.Parameter(item.values()) for item in self.path_pl_A])

        self.path_pl_B = [gen_path_pmat(GRID_SIZES[i+1]) for i in range(len(GRID_SIZES)-1)]
        self.path_pl_B_idxs = [item.indices() for item in self.path_pl_B]
        self.path_pl_B_params = torch.nn.ParameterList([torch.nn.Parameter(item.values()) for item in self.path_pl_B])

        #print([item.shape for item in self.path_pl_B])
        #quit()
        #self.grid_pm_params = [item.values() for item in self.grid_pl]
        
        self.feat_gl = [gen_feat_gmat(LAYER_SIZES[i]) for i in range(len(LAYER_SIZES))]
        self.feat_gr_idxs = [item.indices() for item in self.feat_gl]
        self.feat_gr_params = torch.nn.ParameterList([torch.nn.Parameter(item.values()) for item in self.feat_gl])

        self.feat_pl = [gen_feat_pmat(LAYER_SIZES[i],LAYER_SIZES[i+1]) for i in range(len(LAYER_SIZES)-1)]
        self.feat_pm_idxs = [item.indices() for item in self.feat_pl]
        self.feat_pm_params = torch.nn.ParameterList([torch.nn.Parameter(item.values()) for item in self.feat_pl])

        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm2d(ft) for ft in LAYER_SIZES[::-1]])
        #print(self.)
        #print([item.shape for item in self.feat_pl])
        #quit()

    def forward(self, x):
        t0 = time()
        #self.feat_lin.rebuild()
        #self.grid_lin.rebuild()
        #print(self.grid_lin.gl)
        #print([item.shape for item in self.grid_gl])
        #print(self.feat_pm_params)
        grids = [
            torch.sparse_coo_tensor(
                self.grid_gr_idxs[i],
                self.grid_gr_params[i],
                device=dev,
                requires_grad=True
            ).to_dense()
            for i in range(len(self.grid_gr_idxs))
        ]

        """
        grid_pm = [
            torch.sparse_coo_tensor(
                self.grid_pm_idxs[i],
                self.grid_pm_params[i],
                device=dev,
                requires_grad=True
            ).to_dense()
            for i in range(len(self.grid_pm_idxs))
        ]
        print([item.shape for item in grid_pm])"""
        grid_pm = [torch_sparse_kron(
                torch.sparse_coo_tensor(
                        self.path_pl_A_idxs[i],
                        self.path_pl_A_params[i],
                        device=dev,
                        requires_grad=True
                ).coalesce(),
                torch.sparse_coo_tensor(
                            self.path_pl_B_idxs[i],
                            self.path_pl_B_params[i],
                            device=dev,
                            requires_grad=True
                    ).coalesce()
            ).to_dense().T
            for i in range(len(self.path_pl_A))]
        
        #quit()
        feats = [
            torch.sparse_coo_tensor(
                self.feat_gr_idxs[i],
                self.feat_gr_params[i],
                device=dev,
                requires_grad=True
            ).to_dense()
            for i in range(len(self.feat_gr_idxs))
        ]

        feat_pm = [
            torch.sparse_coo_tensor(
                self.feat_pm_idxs[i],
                self.feat_pm_params[i],
                device=dev,
                requires_grad=True
            ).to_dense()
            for i in range(len(self.feat_pm_idxs))
        ]

        #print(grids)
        t1 = time()
        skel_blocks = torch_dense_skeletal_box_cross_product_blocks(
            #[(item + torch_sparse_eye(*item.shape).to(dev)).coalesce() for item in grids[::-1]],
            #[(item + torch_sparse_eye(*item.shape).to(dev)).coalesce() for item in feats],
            grids[::-1],
            feats,
            grid_pm[::-1],
            feat_pm
        )
        #    [(item - .1*torch_sparse_eye(*item.shape).coalesce().to(dev)).coalesce() for item in self.grid_lin.gl],
        #    [(item -.1*torch_sparse_eye(*item.shape).coalesce().to(dev)).coalesce() for item in self.feat_lin.gl[::-1]],
        #    self.grid_lin.pl,
        #    self.feat_lin.pl[::-1]
        #)
        #print("**")
        #print([item.flatten()[:10] for item in grids])
        #print([item.flatten()[:10] for item in feats])
        #print([item.shape for item in grid_pm])
        #print([item.shape for item in feats])
        #print([item.shape for item in feat_pm])

        t2 = time()
        block_idx_i = 0

        current = x#.float()
        #print(current.shape)
        #quit()
        #print("$$$")
        #plt.matshow(skel_blocks[2][2].to_dense().detach().cpu().numpy())
        #plt.show()
        while block_idx_i < len(skel_blocks)-1:
            #print(block_idx_i)
            #print(current.shape,skel_blocks[block_idx_i][block_idx_i].shape)
            #print(current.shape, skel_blocks[block_idx_i][block_idx_j].shape)
            #print("%%%")
            #print(current.max(), skel_blocks[block_idx_i][block_idx_i].values().max())

            #print("***")
            #print(current)
            #print(skel_blocks[block_idx_i][block_idx_i].T)
            #print(current.abs().max())
            #oc = current
            gsize = GRID_SIZES[::-1][block_idx_i]
            fsize = LAYER_SIZES[block_idx_i]
            current = current.reshape(-1, gsize, gsize, fsize)
            current = self.bns[block_idx_i](current)
            current = torch.flatten(current, start_dim=1)
            current = torch.matmul(skel_blocks[block_idx_i][block_idx_i].T, current.T).T
            #print(current.shape, .shape)
            #print(self.bns[block_idx_i])
            #current = torch.nn.functional.silu(current)
            #current = current + oc
            #print(current.shape,skel_blocks[block_idx_i][block_idx_i+1].shape)
            #print(current.abs().max())
            current = torch.nn.functional.silu(current)
            current = torch.matmul(skel_blocks[block_idx_i][block_idx_i+1].T, current.T).T
            #print(current.abs().max())
            #print(current.shape)
            #print(current.abs().max())
            block_idx_i += 1
        #print(skel_blocks[-1][-1].shape, test_B.gl[-1].shape, current.shape)
        #current = torch.spmm(self.grid_lin.gl[-1].T.float(), current.T).T
        #current = torch.sigmoid(current)
        #current = self.bns[block_idx_i](current)
        current = torch.nn.functional.silu(current)
        current = torch.matmul(skel_blocks[block_idx_i][block_idx_i].T, current.T).T
        #print(current)
        #print(current)
        #current = torch.nn.functional.silu(current)
        #current = torch.spmm(skel_blocks[-1][-1].T, current.T).T
        #current = torch.nn.functional.silu(current)
        t3 = time()
        #print(t1 - t0, t2 - t1, t3 - t2)
        #print(current.shape)

        return current



if __name__ == '__main__':
    sk = SkelConvModel()
    x = torch.rand((20,1024)).to(dev)
    test = sk(x)
    print(test)
    quit()
    targ = torch.rand(test.shape,device=dev)
    loss = torch.abs(targ - test).mean()
    #print("grids")
    #for par in list(sk.grid_gr_params):
        #print(par)
    #    print(torch.autograd.grad(test.sum(), par,retain_graph=True))
    #print("grid pms")
    #for par in list(sk.grid_pm_params):
        #print(par)
    #    print(torch.autograd.grad(test.sum(), par,retain_graph=True))
    #print("feats")
    for par in list(sk.feat_gr_params):
        #print(par)
        print(torch.autograd.grad(test.sum(), par,retain_graph=True))
    print("feat pms")
    for par in list(sk.feat_pm_params):
        #print(par)
        print(torch.autograd.grad(test.sum(), par,retain_graph=True))
