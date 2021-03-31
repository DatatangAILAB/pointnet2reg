"""
Author: Benny
Date: Nov 2019
"""
from data_utils.RegNetDataLoader import RegNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_ssg_normal', help='Experiment root')
    return parser.parse_args()

def test(model, loader):
    mean_err = []
    for j, data in tqdm(enumerate(loader), total=len(loader)):
        points, target = data
        points = points.data.numpy()
        points = torch.Tensor(points)
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        reg = model.eval()
        pred, _ = reg(points)
        mean_err.append(F.l1_loss(pred,target).item())
        print("\n\nbatch",j)
        print("test_err",F.l1_loss(pred,target).item())
        print(pred-target)
    

    test_err = np.mean(np.asarray(mean_err))
    return test_err


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = '/content/drive/MyDrive/log/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = 'data/minibus/'
    TEST_DATASET = RegNetDataLoader(root=DATA_PATH, npoint=args.num_point, uniform=False, split='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_reg = 1
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    
    reg = MODEL.get_model(num_reg).cuda()

    checkpoint = torch.load(str(experiment_dir) + '/cps/best_model.pth', map_location=lambda storage, loc: storage)
    reg.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        test_loss = test(reg.eval(), testDataLoader)
        log_string('Test loss: %f' % (test_loss))



if __name__ == '__main__':
    args = parse_args()
    main(args)
