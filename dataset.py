import os
import glob
import torch
import skvideo.io
import skvideo.datasets
import torch.utils.data as data
from random import randint


def sample_video_clip(file_path, clip_step=1, clip_length=16):
    video_data = skvideo.io.vread(file_path)
    video_length = video_data.shape[0]

    clip_start = randint(0, video_length - (clip_step * (clip_length - 1) + 1))
    clip_data = video_data[clip_start::clip_step, :, :, :][:clip_length, :, :, :]
    return clip_data[::1, :, :, :]


def make_dataset(dataroot, datafile):
    video_files = []

    for line in datafile:
        # file_name label
        line_ = line.split()
        file_name = line_[0]
        label = int(line_[1])
        video_files.append((os.path.join(dataroot, file_name), label))

    return video_files


class VideoFolder(data.Dataset):

    def __init__(self, dataroot='', datafile='', transform=None, clip_step=1, clip_length=16):
        with open(datafile, 'r') as f:
            self.datafile = [l.rstrip('\n') for l in f.readlines()]
        self.videos = make_dataset(dataroot, self.datafile)
        self.clip_step = clip_step
        self.clip_length = clip_length
        self.transform = transform
        print('Dataset size %d.' % len(self.videos))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video_path, video_label = self.videos[index]
        print(video_path)
        images = sample_video_clip(video_path, self.clip_step, self.clip_length)
        video = []
        if self.transform is not None:
            for image in images:
                video.append(self.transform(image))

        video = torch.stack(video).transpose(0, 1)
        # print('done.')
        return video, video_label
