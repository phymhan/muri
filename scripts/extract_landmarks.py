import os
import sys
import glob
from datetime import datetime
import skvideo.io
import shutil
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from facial_landmarks import get_heatmap_from_image
# from .. import facial_landmarks
import face_alignment
from PIL import Image
import torch
import numpy
import pdb


def read_xy(filename):
    with open(filename, 'r') as f:
        l = f.readline()
    l = l.split()
    return int(l[0]), int(l[1])


def write_xy(pts, filename):
    with open(filename, 'w') as f:
        for i in range(pts.size(0)):
            f.write('')


def upsample_image(tensor, size):
    tensor = torch.nn.functional.interpolate(input=tensor.unsqueeze_(0), size=size, mode='bilinear', align_corners=True)
    return tensor[0, ...]


# src = '/dresden/users/lh599/Research/Jean/data_1/round1/videos_crop'
# dst = '/dresden/users/lh599/Research/Jean/data_1/round1/videos_renamed'
src = '/media/ligong/Picasso/Datasets/Jean/round1/videos_crop'
dst = '/media/ligong/Picasso/Datasets/Jean/round1/videos_renamed'


if not os.path.exists(dst):
    os.makedirs(dst)

dids = [os.path.basename(s) for s in glob.glob(os.path.join(src, 'Dyad*'))]
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, face_detector='sfd', flip_input=True, device='cuda')

for did in dids:
    vidfiles = [os.path.basename(s) for s in glob.glob(os.path.join(src, did, '*.mp4'))]
    for filename in vidfiles:
        file_in = os.path.join(src, did, filename)
        file_out = os.path.join(src, did, filename.replace('.mp4', '_landmark.mp4'))
        txt_out = os.path.join(src, did, filename.replace('.mp4', '_landmark.txt'))
        video_data = skvideo.io.vread(file_in)
        # hms = []
        # pts = []
        writer = skvideo.io.FFmpegWriter(file_out)
        with open(txt_out, 'w') as f:
            for i, image in enumerate(video_data):
                lm, pt = get_heatmap_from_image(image, fa, ret_pts=True)
                if lm is None:
                    lm = torch.zeros([1, image.shape[0], image.shape[1]])
                    pt = torch.zeros([68, 2])
                else:
                    lm = torch.clamp(upsample_image(lm[0], (image.shape[0], image.shape[1])), 0, 1)
                    pt = pt[0]
                # hms.append(lm)
                # pts.append(pt)
                frame = (lm.cpu().numpy() * 255).transpose((1, 2, 0)).astype(numpy.uint8)
                writer.writeFrame(frame)
                f.write(f'{i} ' + ' '.join(map(str, pt.view(-1).numpy())) + '\n')
                print(f'--> frame {i}')
        writer.close()
        print('-> %s' % file_out)
