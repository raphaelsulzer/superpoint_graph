"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

from sklearn import preprocessing

import random
import numpy as np
import os
import functools
import torch
import torchnet as tnt
import h5py
import spg

from sklearn.model_selection import train_test_split



def get_datasets(args, test_seed_offset=0):
    """ Gets training and test datasets. """

    # Load superpoints graphs
    split = []
    # for n in range(1,14):
    #     if n != args.cvfold:

    path = '{}/superpoint_graphs/cut/'.format(args.CUSTOM_SET_PATH)
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".h5"):
            split.append(spg.spg_reader(args, path + fname, True))

    trainlist = [split[0],split[1],split[2],split[4],split[5],split[6],split[8],split[9],split[10]]
    testlist = [split[3],split[7],split[11],split[12]]



    # path = '{}/superpoint_graphs/cut/'.format(args.CUSTOM_SET_PATH)
    # for fname in sorted(os.listdir(path)):
    #     if fname.endswith(".h5"):
    #         testlist.append(spg.spg_reader(args, path + fname, True))

    # Normalize edge features
    if args.spg_attribs01:
        trainlist, testlist = spg.scaler01(trainlist, testlist)

    # train_folds = tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in trainlist],
    #                         functools.partial(spg.loader, train=True, args=args, db_path=args.CUSTOM_SET_PATH))
    #
    # test_folds = tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in testlist],
    #                         functools.partial(spg.loader, train=False, args=args, db_path=args.CUSTOM_SET_PATH,
    #                                           test_seed_offset=test_seed_offset))



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

    for n in ['cut']:
        pathP = '{}/parsed/{}/'.format(CUSTOM_SET_PATH, n)
        pathD = '{}/features/{}/'.format(CUSTOM_SET_PATH, n)
        pathC = '{}/superpoint_graphs/{}/'.format(CUSTOM_SET_PATH, n)
        if not os.path.exists(pathP):
            os.makedirs(pathP)
        #random.seed(n)

        for file in os.listdir(pathC):
            print(file)
            if file.endswith(".h5"):
                f = h5py.File(pathD + file, 'r')
                xyz = f['xyz'][:]
                rgb = f['rgb'][:].astype(np.float)
                elpsv = np.stack([ f['xyz'][:,2][:], f['linearity'][:], f['planarity'][:], f['scattering'][:], f['verticality'][:] ], axis=1)
                #lpsv = np.stack([ f['linearity'][:], f['planarity'][:], f['scattering'][:], f['verticality'][:] ], axis=1)

                ma, mi = np.max(xyz,axis=0,keepdims=True), np.min(xyz,axis=0,keepdims=True)
                xyzn = (xyz - mi) / (ma - mi + 1e-8)   # as in PointNet ("normalized location as to the room (from 0 to 1)")
                #xyzn = preprocessing.normalize(xyz)



                # rescale to [-0.5,0.5]; keep x
                elpsv[:,1:] -= 0.5
                elpsv[:, 0] = (elpsv[:, 0] - mi[0][2]) / 4 - 1  # (4m rough guess)
                #elpsv[:, 0] = preprocessing.normalize()

                #lpsv = preprocessing.normalize(lpsv)

                #rgb = preprocessing.normalize(rgb)
                rgb = rgb/255.0 - 0.5


                P = np.concatenate([xyz, rgb, elpsv, xyzn], axis=1)


                # P = np.concatenate([xyz, rgb, elpsv, xyz], axis=1)
                # P = preprocessing.normalize(P)


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
    parser.add_argument('--CUSTOM_SET_PATH', default='/home/raphael/PhD/data/varcity3d/data/ruemonge428')
    args = parser.parse_args()
    preprocess_pointclouds(args.CUSTOM_SET_PATH)
