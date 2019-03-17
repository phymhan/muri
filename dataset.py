import os
import glob
import torch
import skvideo.io
import skvideo.datasets
import torch.utils.data as data
from random import randint


test_set = ["006AZ_Round3_p2_clip5_0",
            "006AZ_Round3_p6_clip1_0",
            "009SB_Round3_p1_clip5_1",
            "009SB_Round3_p3_clip3_1",
            "009SB_Round3_p5_clip2_1",
            "009SB_Round3_p6_clip6_0",
            "009SB_Round3_p7_clip4_0"]


def sample_video_clip(file_path, clip_step=1, clip_length=16):
    video_data = skvideo.io.vread(file_path)
    video_length = video_data.shape[0]

    clip_start = randint(0, video_length - (clip_step * (clip_length - 1) + 1))
    clip_data = video_data[clip_start::clip_step, :, :, :][:clip_length, :, :, :]
    return clip_data[::1, :, :, :]


def make_dataset(dir_path, train=True):
    videos_train = []
    videos_test = []

    for file_name in sorted(glob.glob(os.path.join(dir_path, '*.avi'))):
        label = file_name[:-4].split('_')[-1]
        to_train = True
        for clip_name in test_set:
            if clip_name in file_name:
                to_train = False
                break
        if to_train:
            videos_train.append((file_name, int(label)))
        else:
            videos_test.append((file_name, int(label)))

    if train:
        return videos_train
    else:
        return videos_test



class VideoFolder(data.Dataset):

    def __init__(self, dir_path, transform=None, clip_step=1, clip_length=16, train=True):
        videos = make_dataset(dir_path, train)
        if len(videos) == 0:
            raise (RuntimeError("Found 0 videos in sub folders of: " + dir_path + ".\n"))

        self.videos = videos
        self.clip_step = clip_step
        self.clip_length = clip_length
        self.transform = transform
        print('Found %d videos in %s.' % (len(self.videos), dir_path))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video_path, video_label = self.videos[index]
        images = sample_video_clip(video_path, self.clip_step, self.clip_length)
        video = []
        if self.transform is not None:
            for image in images:
                video.append(self.transform(image))

        video = torch.stack(video).transpose(0, 1)
        return video, video_label


if __name__ == '__main__':
    from torchvision import transforms

    # sample_video_clip("/home/garyzhao/Workspaces/MURI/video/006AZ_Round3_p1_clip1_0.avi")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop((270, 270)),
        transforms.Scale(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    dataset = VideoFolder("./video", transform=transform, train=True)
    for i in range(5):
        print(dataset[i][1])
