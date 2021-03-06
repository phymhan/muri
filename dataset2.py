import os
import glob
import torch
import skvideo.io
import skvideo.datasets
import torch.utils.data as data
import numpy
import numpy as np
import face_alignment
from facial_landmarks import get_heatmap_from_image
import cv2
from copy import deepcopy
EPS_FRAME = 3


def upsample_image(tensor, size):
    tensor = torch.nn.functional.interpolate(input=tensor.unsqueeze_(0), size=size, mode='bilinear', align_corners=True)
    return tensor[0, ...]


def sample_video_clip2(file1_path, file2_path, clip_step=1, clip_length=16):
    video1_data = skvideo.io.vread(file1_path)
    video2_data = skvideo.io.vread(file2_path)
    video_length = min(video1_data.shape[0], video2_data.shape[0])
    clip_start = numpy.random.randint(0, video_length - (clip_step * (clip_length - 1) + 1))
    if abs(video1_data.shape[0]-video2_data.shape[0]) <= EPS_FRAME:
        clip_step1 = clip_step
        clip_step2 = clip_step
        clip_start1 = clip_start
        clip_start2 = clip_start
    elif abs(video1_data.shape[0]*2-video2_data.shape[0]) <= EPS_FRAME:
        clip_step1 = clip_step
        clip_step2 = 2 * clip_step
        clip_start1 = clip_start
        clip_start2 = 2 * clip_start
    else:
        clip_step1 = 2 * clip_step
        clip_step2 = clip_step
        clip_start1 = 2 * clip_start
        clip_start2 = clip_start
    clip_start1 = min(clip_start1, video1_data.shape[0] - (clip_step1 * (clip_length - 1) + 1))
    clip_start2 = min(clip_start2, video2_data.shape[0] - (clip_step2 * (clip_length - 1) + 1))
    # print(file1_path)
    # print(file2_path)
    # print(clip_start)
    clip1_data = video1_data[clip_start1:clip_start1+clip_step1*clip_length:clip_step1, :, :, :]
    clip2_data = video2_data[clip_start2:clip_start2+clip_step2*clip_length:clip_step2, :, :, :]
    return clip1_data, clip2_data


def make_dataset2(dataroot, datafile, binarize=False):
    video_files = []

    for line in datafile:
        # file_name label
        line_ = line.split()
        file1_name = line_[0]
        file2_name = line_[1]
        label = int(line_[2])
        if binarize:
            label = 0 if label < 5 else 1
        video_files.append((os.path.join(dataroot, file1_name), os.path.join(dataroot, file2_name), label))

    return video_files


class VideoFolder2(data.Dataset):

    def __init__(self, dataroot='', datafile='', transform=None, clip_step=1, clip_length=16,
                 binarize=False, landmark=False, fa=None):
        with open(datafile, 'r') as f:
            self.datafile = [l.rstrip('\n') for l in f.readlines()]
        self.videos = make_dataset2(dataroot, self.datafile, binarize)
        self.clip_step = clip_step
        self.clip_length = clip_length
        self.transform = transform
        self._landmark = landmark
        self.fa = fa
        print('Dataset size %d.' % len(self.videos))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video1, video2, video_label = self.videos[index]
        images1, images2 = sample_video_clip2(video1, video2, self.clip_step, self.clip_length)
        video1, video2 = [], []
        if self.transform is not None:
            for image1_, image2_ in zip(images1, images2):
                image1 = self.transform(image1_)
                image2 = self.transform(image2_)
                if self._landmark:
                    lm1 = get_heatmap_from_image(image1_, self.fa)
                    if lm1 is None:
                        lm1 = torch.zeros([1, image1.size(1), image1.size(2)])
                    else:
                        lm1 = upsample_image(lm1[0], (image1.size(1), image1.size(2)))
                    lm2 = get_heatmap_from_image(image2_, self.fa)
                    # print(lm2)
                    if lm2 is None:
                        lm2 = torch.zeros([1, image2.size(1), image2.size(2)])
                    else:
                        lm2 = upsample_image(lm2[0], (image2.size(1), image2.size(2)))
                    image1 = torch.cat((image1, lm1), dim=0)
                    image2 = torch.cat((image2, lm2), dim=0)
                video1.append(image1)
                video2.append(image2)

                # fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=True)
                # pred = fa.get_landmarks_from_image(image1)
                # # print(pred)
                #
                # hm = get_heatmap_from_image(image1, fa)[0]
                # image1 = self.transform(image1)
                # print(image1.size())
                #
                # gcam = hm.squeeze().cpu().data.numpy()
                # gcam = cv2.resize(gcam, (200, 200))
                # gcam = cv2.applyColorMap(
                #     np.uint8(255 * gcam), cv2.COLORMAP_JET)
                # cv2.imwrite('att_res/heatmap.jpg', gcam)
                # cv2.imwrite('att_res/image.jpg', image1)
                # exit(0)

        video1 = torch.stack(video1).transpose(0, 1)
        video2 = torch.stack(video2).transpose(0, 1)
        return video1, video2, video_label


class VideoFile2(data.Dataset):

    def __init__(self, datapath1='', datapath2='', label=0, transform=None, clip_step=1, clip_length=16):
        print(datapath1, datapath2)
        self.meta1 = skvideo.io.ffprobe(datapath1)
        self.meta2 = skvideo.io.ffprobe(datapath2)
        print(self.meta1)
        print(self.meta2)
        self.datapath1 = datapath1
        self.datapath2 = datapath2
        # self.chunk1 = skvideo.io.vread(datapath1)
        # self.chunk2 = skvideo.io.vread(datapath2)
        # try:
        #     self.chunk1 = skvideo.io.vread(datapath1)
        # except:
        #     raise RuntimeError
        # try:
        #     self.chunk2 = skvideo.io.vread(datapath2)
        # except:
        #     raise RuntimeError
        # print(self.chunk1.shape)
        # print(self.chunk2.shape)
        self.nb_frames1 = int(self.meta1['video']['@nb_frames'])
        self.nb_frames2 = int(self.meta2['video']['@nb_frames'])
        if abs(self.nb_frames1 - self.nb_frames2) <= EPS_FRAME:
            self._clip_step1 = clip_step
            self._clip_step2 = clip_step
        elif abs(self.nb_frames1 * 2 - self.nb_frames2) <= EPS_FRAME:
            self._clip_step1 = clip_step
            self._clip_step2 = 2 * clip_step
        else:
            self._clip_step1 = 2 * clip_step
            self._clip_step2 = clip_step
        self._clip_length = clip_length
        self._label = int(label)
        self.transform = transform
        print(datapath1, datapath2)

    def __len__(self):
        return int(self.nb_frames1/(self._clip_step1*self._clip_length))

    def __getitem__(self, index):
        chunk1 = skvideo.io.vread(self.datapath1)
        chunk2 = skvideo.io.vread(self.datapath2)
        images1 = chunk1[index*self._clip_step1*self._clip_length:(index+1)*self._clip_step1*self._clip_length:self._clip_step1, :, :, :]
        images2 = chunk2[index * self._clip_step2 * self._clip_length:(index + 1) * self._clip_step2 * self._clip_length:self._clip_step2, :, :, :]
        video1, video2 = [], []
        if self.transform is not None:
            for image1, image2 in zip(images1, images2):
                video1.append(self.transform(image1))
                video2.append(self.transform(image2))
        video1 = torch.stack(video1).transpose(0, 1)
        video2 = torch.stack(video2).transpose(0, 1)
        return video1, video2, self._label