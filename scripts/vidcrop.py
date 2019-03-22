import os
import glob
from datetime import datetime
import skvideo.io


def read_xy(filename):
    with open(filename, 'r') as f:
        l = f.readline()
    l = l.split()
    return int(l[0]), int(l[1])


src = '/dresden/users/lh599/Research/Jean/data_1/round1/videos_mp4'
dst = '/dresden/users/lh599/Research/Jean/data_1/round1/videos_crop'
W = 200

if not os.path.exists(dst):
    os.makedirs(dst)

dids = [os.path.basename(s) for s in glob.glob(os.path.join(src, 'Dyad*'))]

for did in dids:
    vidfiles = [os.path.basename(s) for s in glob.glob(os.path.join(src, did, '*.mp4'))]
    if not os.path.exists(os.path.join(dst, did)):
        os.makedirs(os.path.join(dst, did))
    for filename in vidfiles:
        file_in = os.path.join(src, did, filename)
        file_out = os.path.join(dst, did, filename)
        x, y = read_xy(os.path.join(src, did, filename+'.txt'))
        print('ffmpeg -i %s -filter:v "crop=%d:%d:%d:%d" %s' % (file_in, W, W, x, y, file_out))
    print('-> did: %s' % did)
