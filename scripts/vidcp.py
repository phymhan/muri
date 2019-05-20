import os
import glob
from datetime import datetime
import skvideo.io
import shutil


def read_xy(filename):
    with open(filename, 'r') as f:
        l = f.readline()
    l = l.split()
    return int(l[0]), int(l[1])


# src = '/dresden/users/lh599/Research/Jean/data_1/round1/videos_crop'
# dst = '/dresden/users/lh599/Research/Jean/data_1/round1/videos_renamed'
src = '/media/ligong/Picasso/Datasets/Jean/round1/videos_crop'
dst = '/media/ligong/Picasso/Datasets/Jean/round1/videos_renamed'


if not os.path.exists(dst):
    os.makedirs(dst)

dids = [os.path.basename(s) for s in glob.glob(os.path.join(src, 'Dyad*'))]

for did in dids:
    vidfiles = [os.path.basename(s) for s in glob.glob(os.path.join(src, did, '*.mp4'))]
    for filename in vidfiles:
        file_in = os.path.join(src, did, filename)
        newname = did+'_'+filename
        file_out = os.path.join(dst, newname)
        shutil.copyfile(file_in, file_out)
    print('-> did: %s' % did)
