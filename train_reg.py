"""
Author: Benny
Date: Nov 2019
"""
from data_utils.RegNetDataLoader import RegNetDataLoader
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
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

    test_err = np.mean(np.asarray(mean_err))
    return test_err

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('/content/drive/MyDrive/log')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('cps/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = 'data/minibus/'

    TRAIN_DATASET = RegNetDataLoader(root=DATA_PATH, uniform=False,npoint=args.num_point, split='train')
    TEST_DATASET = RegNetDataLoader(root=DATA_PATH, uniform=False, npoint=args.num_point, split='test')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    num_reg = 1
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))

    reg = MODEL.get_model(num_reg).cuda()
    criterion = MODEL.get_loss().cuda()

    try:
        checkpoint = torch.load(str(checkpoints_dir)+'/best_model.pth')
        start_epoch = checkpoint['epoch']
        reg.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
        log_string('start at epoch:%d' % (start_epoch))
        min_loss=checkpoint['test_err']
        log_string('start at test_err:%.4f' % (min_loss))
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        min_loss=1000


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            reg.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(reg.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    '''TRANING'''
    
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        mean_err = []
        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            # points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            # points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)

            points = points.transpose(2, 1)
            
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            reg = reg.train()
            pred, trans_feat = reg(points)

            loss = criterion(pred, target, trans_feat)
            mean_err.append(F.l1_loss(pred,target).item())

            loss.backward()
            optimizer.step()
            
        train_err = np.mean(np.asarray(mean_err))
        log_string('Epoch (%d/%s) Train_Error (%.4f):' % (epoch + 1, args.epoch,train_err))

        with torch.no_grad():
            test_err=test(reg.eval(), testDataLoader)
            log_string('Test_Error (%.4f):' % (test_err))

            state = {
                'epoch': epoch,
                'train_err': train_err,
                'test_err': test_err,
                'model_state_dict': reg.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }

            if test_err < min_loss:
                min_loss=test_err
                logger.info('Save model...')
                bestsavepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s'% bestsavepath)     
                torch.save(state, bestsavepath)


    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
