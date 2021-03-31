import argparse
import numpy as np
import os
import torch
import sys
import importlib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def sample_points(points,npoint):
    if points.shape[0] < npoint:
        return np.concatenate([points, np.zeros((npoint-points.shape[0], 3), dtype=np.float32)], axis=0)
    else:
        idx = np.arange(points.shape[0])
        np.random.shuffle(idx)
        return points[idx[0:npoint]]

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--log_dir', type=str, default='pointnet2_ssg_normal', help='Experiment root')
    parser.add_argument('--data', type=str, help='data dir need infer')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = '/content/drive/MyDrive/log/' + args.log_dir

    '''arg'''
    args = parse_args()
    print(args)

    '''DATA LOADING'''
    print('Load data ...',args.data)
    DATA_PATH = 'data/minibus/'
    
    '''MODEL LOADING'''
    num_reg = 1
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    
    reg = MODEL.get_model(num_reg).cuda()

    checkpoint = torch.load(str(experiment_dir) + '/cps/best_model.pth')
    reg.load_state_dict(checkpoint['model_state_dict'])

    points = np.loadtxt(args.data, delimiter=' ').astype(np.float32)
    points[:, 0:3] = pc_normalize(points[:, 0:3])
    points=sample_points(points,args.num_point)
    points_tensor=torch.Tensor(points)
    points_tensor = points_tensor.transpose(2, 1)
    points_tensor = points_tensor.cuda()
    pred, _ = reg(points_tensor)
    print(pred)



if __name__ == '__main__':
    args = parse_args()
    main(args)
