import os
import glob
import torch
import skvideo.io
import skvideo.datasets
import torch.utils.data as data
from random import randint


def sample_video_clip2(file1_path, file2_path, clip_step=1, clip_length=16, verbose=False):
    video1_data = skvideo.io.vread(file1_path)
    video2_data = skvideo.io.vread(file2_path)
    video_length = video1_data.shape[0]

    clip_start = randint(0, video_length - (clip_step * (clip_length - 1) + 1))
    clip1_data = video1_data[clip_start::clip_step, :, :, :][:clip_length, :, :, :]
    clip2_data = video2_data[clip_start::clip_step, :, :, :][:clip_length, :, :, :]

    if verbose:
        print(video1_data)
        # print(video2_data)
        print(clip1_data)
        # print(clip2_data)

    return clip1_data[::1, :, :, :], clip2_data[::1, :, :, :]


def make_dataset2(dataroot, datafile):
    video_files = []

    for line in datafile:
        # file_name label
        line_ = line.split()
        file1_name = line_[0]
        file2_name = line_[1]
        label = int(line_[2])
        video_files.append((os.path.join(dataroot, file1_name), os.path.join(dataroot, file2_name), label))

    return video_files


class VideoFolder2(data.Dataset):

    def __init__(self, dataroot='', datafile='', transform=None, clip_step=1, clip_length=16):
        with open(datafile, 'r') as f:
            self.datafile = [l.rstrip('\n') for l in f.readlines()]
        self.videos = make_dataset2(dataroot, self.datafile)
        self.clip_step = clip_step
        self.clip_length = clip_length
        self.transform = transform
        print('Dataset size %d.' % len(self.videos))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        print('index: %d' % index)
        video1, video2, video_label = self.videos[index]
        print(video1, video2)
        # if index == 2:
        #     images1, images2 = sample_video_clip2(video1, video2, self.clip_step, self.clip_length, True)
        # else:
        #     images1, images2 = sample_video_clip2(video1, video2, self.clip_step, self.clip_length)
        images1, images2 = sample_video_clip2(video1, video2, self.clip_step, self.clip_length)
        video1, video2 = [], []
        if self.transform is not None:
            for image1, image2 in zip(images1, images2):
                video1.append(self.transform(image1))
                video2.append(self.transform(image2))
        # if index == 2:
        #     print(video1)
        #     print(images1)
        #     print(video1[0].size())
        video1 = torch.stack(video1).transpose(0, 1)
        video2 = torch.stack(video2).transpose(0, 1)
        return video1, video2, video_label
