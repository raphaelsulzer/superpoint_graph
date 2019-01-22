from plyfile import PlyData, PlyElement
import numpy as np



filename = '/home/raphael/PhD/data/hayko-varcity3dchallenge-3cb58e583578/data/ruemonge428/pcl.ply'


plydata = PlyData.read(filename)

xyz = np.stack([plydata['vertex'][n] for n in['x', 'y', 'z']], axis=1)


b=5

try:
    rgb = np.stack([plydata['vertex'][n]
                    for n in ['red', 'green', 'blue']]
                   , axis=1).astype(np.uint8)
except ValueError:
    rgb = np.stack([plydata['vertex'][n]
                    for n in ['r', 'g', 'b']]
                   , axis=1).astype(np.float32)
if np.max(rgb) > 1:
    rgb = rgb
try:
    object_indices = plydata['vertex']['object_index']
    labels = plydata['vertex']['label']
    #return xyz, rgb, labels, object_indices
except ValueError:
    try:
        labels = plydata['vertex']['label']
        #return xyz, rgb, labels
    except ValueError:
        #return xyz, rgb
        pass


a=5
