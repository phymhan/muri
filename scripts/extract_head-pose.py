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
import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import datasets, hopenet, utils

from skimage import io
import dlib

# src = '/dresden/users/lh599/Research/Jean/data_1/round1/videos_crop'
# dst = '/dresden/users/lh599/Research/Jean/data_1/round1/videos_renamed'
src = '/media/ligong/Picasso/Datasets/Jean/round1/videos_crop'
dst = '/media/ligong/Picasso/Datasets/Jean/round1/videos_renamed'


if not os.path.exists(dst):
    os.makedirs(dst)

dids = [os.path.basename(s) for s in glob.glob(os.path.join(src, 'Dyad*'))]
cudnn.enabled = True
batch_size = 1
gpu = 0
snapshot_path = '/media/ligong/Picasso/Active/deep-head-pose/hopenet_robust_alpha1.pkl'

# ResNet50 structure
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
print('Loading snapshot.')
# Load snapshot
saved_state_dict = torch.load(snapshot_path)
model.load_state_dict(saved_state_dict)
print('Loading data.')
transformations = transforms.Compose([transforms.Scale(224),
                                      transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])
model.cuda(gpu)
print('Ready to test network.')

# Test the Model
model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
total = 0
idx_tensor = [idx for idx in range(66)]
idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

for did in dids:
    vidfiles = [os.path.basename(s) for s in glob.glob(os.path.join(src, did, '*driver.mp4'))] + \
               [os.path.basename(s) for s in glob.glob(os.path.join(src, did, '*passenger.mp4'))]
    for filename in vidfiles:
        file_in = os.path.join(src, did, filename)
        file_out = os.path.join(src, did, filename.replace('.mp4', '_headpose.avi'))
        txt_out = os.path.join(src, did, filename.replace('.mp4', '_headpose.txt'))
        landmark_path = os.path.join(src, did, filename.replace('.mp4', '_landmark.txt'))
        video = cv2.VideoCapture(file_in)

        with open(landmark_path, 'r') as f:
            landmark = f.readlines()

        # New cv2
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(file_out, fourcc, video.get(cv2.CAP_PROP_FPS), (width, height))

        with open(txt_out, 'w') as f:
            i = 0
            while True:
                ret, frame = video.read()
                if ret == False:
                    break
                cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                lm = np.asarray(landmark[i].split()[1:], dtype=np.float32).astype(int)
                conf = lm.sum()
                x_min = lm[0::2].min()
                x_max = lm[0::2].max()
                y_min = lm[1::2].min()
                y_max = lm[1::2].max()
                # print(y_min, y_max, x_min, x_max)

                if conf > 0.0:
                    bbox_width = abs(x_max - x_min)
                    bbox_height = abs(y_max - y_min)
                    x_min -= 2 * bbox_width / 4
                    x_max += 2 * bbox_width / 4
                    y_min -= 3 * bbox_height / 4
                    y_max += bbox_height / 4
                    x_min = int(round(max(x_min, 0)))
                    y_min = int(round(max(y_min, 0)))
                    x_max = int(round(min(frame.shape[1], x_max)))
                    y_max = int(round(min(frame.shape[0], y_max)))
                    # print(y_min, y_max, x_min, x_max)
                    # Crop image
                    img = cv2_frame[y_min:y_max, x_min:x_max]
                    img = Image.fromarray(img)

                    # Transform
                    img = transformations(img)
                    img_shape = img.size()
                    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                    img = Variable(img).cuda(gpu)

                    yaw, pitch, roll = model(img)

                    yaw_predicted = F.softmax(yaw)
                    pitch_predicted = F.softmax(pitch)
                    roll_predicted = F.softmax(roll)
                    # Get continuous predictions in degrees.
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

                    # Print new frame with cube and axis
                    f.write(f'{i} {yaw_predicted} {pitch_predicted} {roll_predicted}\n')
                    # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                    utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx=(x_min + x_max) / 2,
                                    tdy=(y_min + y_max) / 2, size=bbox_height / 2)
                    # Plot expanded bounding box
                    # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
                else:
                    f.write(f'{i} {-1} {-1} {-1}\n')

                writer.write(frame)
                if i % 1000 == 0:
                    print(f'--> frame {i}')
                i += 1
        writer.release()
        video.release()
        print('-> %s' % file_out)
