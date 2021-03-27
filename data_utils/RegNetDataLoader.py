import numpy as np
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

MAX_POINTS=500

def sample_points(points):
    if points.shape[0] < MAX_POINTS:
        return np.concatenate([points, np.zeros((MAX_POINTS-points.shape[0], 3), dtype=np.float32)], axis=0)
    else:
        idx = np.arange(points.shape[0])
        np.random.shuffle(idx)
        return points[idx[0:MAX_POINTS]]

class RegNetDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', cache_size=15000):
        self.root = root
        self.npoints = npoint

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'test.txt'))]


        assert (split == 'train' or split == 'test')
        
        self.datapath = [[os.path.join(root,file),os.path.join(root,file.replace("txt","para"))] for file in shape_ids[split]]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, para = self.cache[index]
        else:
            fn = self.datapath[index]
            point_set = np.loadtxt(fn[0], delimiter=' ').astype(np.float32)
            point_set=sample_points(point_set)
            para = np.loadtxt(fn[1], delimiter=' ').astype(np.float32)
            point_set = point_set[0:self.npoints,:]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, para)

        return point_set, para

    def __getitem__(self, index):
        return self._get_item(index)



if __name__ == '__main__':
    import torch

    data = RegNetDataLoader('data/minibus/')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=False)
    for point, para in DataLoader:
        print(point.shape)
        print(para.shape)
        