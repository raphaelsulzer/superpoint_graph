from plyfile import PlyData, PlyElement
import numpy as np



full = '/home/raphael/PhD/data/hayko-varcity3dchallenge-3cb58e583578/data/ruemonge428/pcl_short.ply'
train = '/home/raphael/PhD/data/hayko-varcity3dchallenge-3cb58e583578/data/ruemonge428/pcl_train_short.ply'
test = '/home/raphael/PhD/data/hayko-varcity3dchallenge-3cb58e583578/data/ruemonge428/pcl_test_short.ply'

plydata_f = PlyData.read(filename)
plydata_tr = PlyData.read(filename)
plydata_te = PlyData.read(filename)

xyz = np.stack([plydata['vertex'][n] for n in['x', 'y', 'z']], axis=1)



b=5

b=4
