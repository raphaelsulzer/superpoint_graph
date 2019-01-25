from plyfile import PlyData, PlyElement
import numpy as np


full = '/home/raphael/PhD/data/hayko-varcity3dchallenge-3cb58e583578/data/ruemonge428/original_data/pcl.ply'
train = '/home/raphael/PhD/data/hayko-varcity3dchallenge-3cb58e583578/data/ruemonge428/original_data/pcl_gt_test.ply'

plydata_f = PlyData.read(full)
plydata_tr = PlyData.read(train)

xyz = np.stack([plydata_f['vertex'][n] for n in['x', 'y', 'z']], axis=1)

rgb = np.stack([plydata_f['vertex'][n] for n in['diffuse_red', 'diffuse_green', 'diffuse_blue']], axis=1)

label_tr = np.stack([plydata_tr['vertex'][n] for n in['red', 'green', 'blue']], axis=1)
#label_te = np.stack([plydata_te['vertex'][n] for n in['red', 'green', 'blue']], axis=1)


file=open('/home/raphael/PhD/data/hayko-varcity3dchallenge-3cb58e583578/data/ruemonge428/pcl_gt_test_withColor.ply','w+')

# write ply file header
file.write(
'ply \n'
'format ascii 1.0 \n'
'element vertex 897324 \n'   # this has to be manually adjusted in the resulting file
'property float x \n'
'property float y \n'
'property float z \n'
'property uchar r \n'
'property uchar g \n'
'property uchar b \n'
'property uchar label \n'
'end_header \n'
)



for i, row in enumerate(xyz):


    #label
    red=label_tr[i][0]
    green=label_tr[i][1]
    blue=label_tr[i][2]

    # if no label present, continue
    if (red == 0) & (green == 0) & (blue == 0):
        continue


    #xyz
    file.write('{:0.6f} '.format(row[0]))
    file.write('{:0.6f} '.format(row[1]))
    file.write('{:0.6f} '.format(row[2]))
    #color
    file.write('{:d} '.format(rgb[i][0]))
    file.write('{:d} '.format(rgb[i][1]))
    file.write('{:d} '.format(rgb[i][2]))


    #write label
    if (red == 255) & (green == 255) & (blue == 0):
        file.write(str(1)+'\n') # yellow = facade
    elif (red == 255) & (green == 128) & (blue == 0):
        file.write(str(2)+'\n') # orange = door
    elif (red == 128) & (green == 255) & (blue == 255):
        file.write(str(3)+'\n') # lightblue = sky
    elif (red == 128) & (green == 0) & (blue == 255):
        file.write(str(4) + '\n')  # purple = balcony
    elif (red == 255) & (green == 0) & (blue == 0):
        file.write(str(5) + '\n')  # red = window
    elif (red == 0) & (green == 255) & (blue == 0):
        file.write(str(6) + '\n')  # green = shop
    elif (red == 0) & (green == 0) & (blue == 255):
        file.write(str(7) + '\n')  # blue = roof
    else:
        file.write(str(0)+'\n') # black = no class / should not happen
        print('invalid class: ', red, green, blue)



file.close()




