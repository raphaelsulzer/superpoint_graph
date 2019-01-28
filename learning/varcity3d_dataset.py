"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import os
import functools
import torch
import torchnet as tnt
import h5py
import spg

# def get_datasets(args, test_seed_offset=0):
#     """ Gets training and test datasets. """
#
#     # Load superpoints graphs
#     testlist, trainlist = [], []
#     for n in ['train', 'test']:
#         if n != args.cvfold:
#             path = '{}/superpoint_graphs/'.format(args.CUSTOM_SET_PATH, n)
#             for fname in sorted(os.listdir(path)):
#                 if fname.endswith(".h5"):
#                     print("h5 file found")
#                     trainlist.append(spg.spg_reader(args, path + fname, True))
#     path = '{}/superpoint_graphs/'.format(args.CUSTOM_SET_PATH, args.cvfold)
#     for fname in sorted(os.listdir(path)):
#         if fname.endswith(".h5"):
#             testlist.append(spg.spg_reader(args, path + fname, True))
#
#     # Normalize edge features
#     if args.spg_attribs01:
#         trainlist, testlist = spg.scaler01(trainlist, testlist)
#
#     return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
#                                     functools.partial(spg.loader, train=True, args=args, db_path=args.S3DIS_PATH)), \
#            tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
#                                     functools.partial(spg.loader, train=False, args=args, db_path=args.S3DIS_PATH, test_seed_offset=test_seed_offset))



def get_datasets(args, test_seed_offset=0):

    # train_names = ['bildstein_station1', 'bildstein_station5', 'domfountain_station1', 'domfountain_station3', 'neugasse_station1', 'sg27_station1', 'sg27_station2', 'sg27_station5', 'sg27_station9', 'sg28_station4', 'untermaederbrunnen_station1']
    # valid_names = ['bildstein_station3', 'domfountain_station2', 'sg27_station4', 'untermaederbrunnen_station3']
    #
    if args.db_train_name == 'train':
        trainset = ['train/']
    elif args.db_train_name == 'test':
        testset = ['test/']

    # if args.db_test_name == 'val':
    #     testset = ['train/' + f for f in valid_names]
    # elif args.db_test_name == 'testred':
    #     testset = ['test_reduced/' + os.path.splitext(f)[0] for f in os.listdir(args.SEMA3D_PATH + '/superpoint_graphs/test_reduced')]
    # elif args.db_test_name == 'testfull':
    #     testset = ['test_full/' + os.path.splitext(f)[0] for f in os.listdir(args.SEMA3D_PATH + '/superpoint_graphs/test_full')]

    # Load superpoints graphs
    testlist = [spg.spg_reader(args, args.CUSTOM_SET_PATH + '/superpoint_graphs/test/pcl_gt_test_withColor' + '.h5', True)]
    trainlist = [spg.spg_reader(args, args.CUSTOM_SET_PATH + '/superpoint_graphs/train/pcl_gt_train_withColor' + '.h5', True)]

    # Normalize edge features
    if args.spg_attribs01:
        trainlist, testlist = spg.scaler01(trainlist, testlist)

    # aaa = tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
    #                         functools.partial(spg.loader, train=True, args=args, db_path=args.CUSTOM_SET_PATH))


    return tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
                                    functools.partial(spg.loader, train=True, args=args, db_path=args.CUSTOM_SET_PATH)), \
           tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
                                    functools.partial(spg.loader, train=False, args=args, db_path=args.CUSTOM_SET_PATH, test_seed_offset=test_seed_offset))



def get_info(args):
    edge_feats = 0
    for attrib in args.edge_attribs.split(','):
        a = attrib.split('/')[0]
        if a in ['delta_avg', 'delta_std', 'xyz']:
            edge_feats += 3
        else:
            edge_feats += 1

    return {
        'node_feats': 14 if args.pc_attribs=='' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'classes': 7,
        'inv_class_map': {0:'facade', 1:'door', 2:'sky', 3:'balcony', 4:'window', 5:'shop', 6:'roof'},
    }



def preprocess_pointclouds(CUSTOM_SET_PATH):
    """ Preprocesses data by splitting them by components and normalizing."""

    for n in ['train', 'test']:
        pathP = '{}/parsed/{}/'.format(CUSTOM_SET_PATH, n)
        pathD = '{}/features/{}/'.format(CUSTOM_SET_PATH, n)
        pathC = '{}/superpoint_graphs/{}/'.format(CUSTOM_SET_PATH, n)
        if not os.path.exists(pathP):
            os.makedirs(pathP)
        random.seed(n)

        for file in os.listdir(pathC):
            print(file)
            if file.endswith(".h5"):
                f = h5py.File(pathD + file, 'r')
                xyz = f['xyz'][:]
                rgb = f['rgb'][:].astype(np.float)
                elpsv = np.stack([ f['xyz'][:,2][:], f['linearity'][:], f['planarity'][:], f['scattering'][:], f['verticality'][:] ], axis=1)

                # rescale to [-0.5,0.5]; keep xyz
                elpsv[:,0] = elpsv[:,0] / 4 - 0.5 # (4m rough guess)
                elpsv[:,1:] -= 0.5
                rgb = rgb/255.0 - 0.5

                ma, mi = np.max(xyz,axis=0,keepdims=True), np.min(xyz,axis=0,keepdims=True)
                xyzn = (xyz - mi) / (ma - mi + 1e-8)   # as in PointNet ("normalized location as to the room (from 0 to 1)")

                print(xyz.shape)
                print(xyzn.shape)
                print(rgb.shape)
                print(elpsv.shape)


                P = np.concatenate([xyz, rgb, elpsv, xyzn], axis=1)

                f = h5py.File(pathC + file, 'r')
                numc = len(f['components'].keys())

                with h5py.File(pathP + file, 'w') as hf:
                    for c in range(numc):
                        idx = f['components/{:d}'.format(c)][:].flatten()
                        if idx.size > 10000: # trim extra large segments, just for speed-up of loading time
                            ii = random.sample(range(idx.size), k=10000)
                            idx = idx[ii]

                        hf.create_dataset(name='{:d}'.format(c), data=P[idx,...])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    parser.add_argument('--CUSTOM_SET_PATH', default='/home/raphael/PhD/data/hayko-varcity3dchallenge-3cb58e583578/data/ruemonge428')
    args = parser.parse_args()
    preprocess_pointclouds(args.CUSTOM_SET_PATH)
