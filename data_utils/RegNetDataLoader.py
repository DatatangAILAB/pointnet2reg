import numpy as np
import warnings
import os
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def sample_points(points,npoint):
    if points.shape[0] < npoint:
        return np.concatenate([points, np.zeros((npoint-points.shape[0], 3), dtype=np.float32)], axis=0)
    else:
        idx = np.arange(points.shape[0])
        np.random.shuffle(idx)
        return points[idx[0:npoint]]

class RegNetDataLoader(Dataset):
    def __init__(self, root,  npoint=500, split='train',uniform=False, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform

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

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            point_set=sample_points(point_set,self.npoints)

            para = np.loadtxt(fn[1], delimiter=' ').astype(np.float32)
            para=para[6:7]

            ##不考虑旋转的相对方向因素
            dd= lambda t: t+np.pi if t<0 else t
            para=dd(para)
            

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
        # print(point)
        print(para.shape)
        print(para)
        