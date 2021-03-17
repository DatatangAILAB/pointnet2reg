"""
Author: Benny
Date: Nov 2019
"""
from data_utils.RegNetDataLoader import RegNetInferDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
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
    experiment_dir = './log/' + args.log_dir

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
    DATA_PATH = 'data/treenet/'
    INFER_DATASET = RegNetInferDataLoader(root=DATA_PATH, npoint=args.num_point)
    inferDataLoader = torch.utils.data.DataLoader(INFER_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_reg = 7
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    
    reg = MODEL.get_model(num_reg).cuda()

    checkpoint = torch.load(str(experiment_dir) + '/cps/best_model.pth')
    reg.load_state_dict(checkpoint['model_state_dict'])

    for j, points in tqdm(enumerate(inferDataLoader), total=len(inferDataLoader)):
        points = points.transpose(2, 1)
        points = points.cuda()
        reg = model.eval()
        pred, _ = reg(points)
        print(pred)



if __name__ == '__main__':
    args = parse_args()
    main(args)
