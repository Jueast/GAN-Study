# py2/3 compatibility

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn

class MLP_G(nn.Module):
    def __init__(self, isize, nc, nz, ngh, ngpu):
        super(MLP_G, self).__init__()
        self.ngpu = ngpu
        
        main = nn.Sequential(
            # Z goes into a linear of size: ngh
            nn.Linear(nz, ngh),
            nn.ReLU(True),
            nn.Linear(ngh, ngh),
            nn.ReLU(True),
            nn.Linear(ngh, ngh),
            nn.ReLU(True),
            nn.Linear(ngh, nc * isize * isize),
        )
        
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz
    def forward(self, input_z):
        input_z = input_z.view(input_z.size(0), self.nz)
        gpu_ids = None
        if isinstance(input_z.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        out = nn.parallel.data_parallel(self.main, input_z, gpu_ids)
        return out.view(input_z.size(0), self.nc, self.isize, self.isize)

        
        
    
class MLP_D(nn.Module):
    def __init__(self, isize, nc, ndh, ngpu):
        super(MLP_D, self).__init__()
        self.ngpu = ngpu
        main = nn.Sequential(
            # Z goes into a linear of size: ngh
            nn.Linear(nc * isize * isize, ndh),
            nn.ReLU(True),
            nn.Linear(ndh, ndh),
            nn.ReLU(True),
            nn.Linear(ndh, ndh),
            nn.ReLU(True),
            nn.Linear(ndh, 1),
        )
        
        self.main = main
        self.nc = nc
        self.isize = isize

    def forward(self, input_x):
        input_x = input_x.view(input_x.size(0), self.nc * self.isize * self.isize)
        gpu_ids = None
        if isinstance(input_x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        out = nn.parallel.data_parallel(self.main, input_x, gpu_ids)

        # minibatch critics here to avoid generating mode problem
        out = out.mean(0)
        return out.view(1)
    