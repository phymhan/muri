# @ This code is for a demo to generate the grad-cam map

from __future__ import print_function

import argparse

import cv2
import numpy as np
from gradcam_lkp_net_video import GradCAM

import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn
import torch
import os

from model2 import C3D2
from dataset import VideoFolder
import dataset

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def preprocess_image(image, transform):

    image_ = transform(image).unsqueeze(0)
    if args.cuda:
        image_ = image_.cuda()
    image_ = Variable(image_, volatile=False, requires_grad=True)
    return image_


def view_cam(images, gcam):
    gcam = gcam - np.min(gcam)
    gcam = gcam / np.max(gcam)

    repeat = np.full(gcam.shape[1], (video_clip_length / gcam.shape[1]), dtype=np.int64)
    print(gcam.shape)
    gcam_repeat = np.repeat(gcam, repeat, axis=1)
    cams = np.zeros((224 * 2, 224 * gcam_repeat.shape[1], 3), np.uint8)
    for ind, image in enumerate(images):
        gcam = gcam_repeat[0][ind]
        b, g, r = cv2.split(image)  # get b,g,r
        image = cv2.merge([r, g, b])
        h, w, d = image.shape
        gcam = cv2.resize(gcam, (w, h))
        gcam = cv2.applyColorMap(
            np.uint8(255 * gcam), cv2.COLORMAP_JET)
        gcam = np.asarray(gcam, dtype=np.float) #+ \
               #np.asarray(image, dtype=np.float)
        gcam = 255 * gcam / np.max(gcam)
        gcam = np.uint8(gcam)
        cams[0:224, ind*224:ind*224 + 224, :3] = image
        cams[224:224*2, ind*224:ind*224 + 224, :3] = gcam
    cv2.imshow('v_attention', cams)
    cv2.waitKey(0)


def save_cam(filename, images, gcam):
    # zero out boarders
    # print(gcam.shape)
    # print(gcam[0,2,...])

    gcam = gcam - np.min(gcam)
    gcam = gcam / np.max(gcam)

    repeat = np.full(gcam.shape[1], (video_clip_length / gcam.shape[1]), dtype=np.int64)
    gcam_repeat = np.repeat(gcam, repeat, axis=1)
    cams = np.zeros((224 * 2, 224 * gcam_repeat.shape[1], 3), np.uint8)
    for ind, image in enumerate(images):
        gcam = gcam_repeat[0][ind]
        b, g, r = cv2.split(image)  # get b,g,r
        image = cv2.merge([r, g, b])
        h, w, d = image.shape
        gcam = cv2.resize(gcam, (w, h))
        gcam = cv2.applyColorMap(
            np.uint8(255 * gcam), cv2.COLORMAP_JET)
        gcam = np.asarray(gcam, dtype=np.float)  # + \
        # np.asarray(image, dtype=np.float)
        gcam = 255 * gcam / np.max(gcam)
        gcam = np.uint8(gcam)
        cams[0:224, ind * 224:ind * 224 + 224, :3] = image
        cams[224:224 * 2, ind * 224:ind * 224 + 224, :3] = gcam
    cv2.imwrite(filename, cams)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Grad-CAM visualization')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--layer', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--name', type=str, default='1')
    parser.add_argument('--output_dir', type=str, default='att_res')
    parser.add_argument('--arch', type=int, default=1)
    parser.add_argument('--comb', type=int, default=1)
    parser.add_argument('--fc_dim', type=int, default=1024)
    global args
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    #################################################
    # load a trained model peremeter
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    print('Loading a model...')
    #model = torchvision.models.vgg19(pretrained=True)
    # model = C3D(num_classes=args.num_classes).cuda()
    model = C3D2(num_classes=args.num_classes, arch=args.arch, comb=args.comb, fc_dim=args.fc_dim).cuda()

    # load model parameters
    # load model perematers for ordinary classification model
    #model.load_state_dict(torch.load("grad_cam/classifier_vgg.pt"))
    # load model perematers for our classification model based on two losses
    checkpoint = torch.load(args.resume)
    # best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint)

    #video_dir_path = "./face_video"
    video_path = args.video
    # label = float(video_path[:-4].split('_')[-1])
    label = 8
    video_frame_size = 112
    video_clip_step = 5
    video_clip_length = 16
    num_workers = 0

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([video_frame_size, video_frame_size]),
        transforms.ToTensor()])
    transform_ = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        transforms.ToTensor()])
    images = dataset.sample_video_clip(video_path, video_clip_step, video_clip_length)
    image_list = []
    video = []
    for image in images:
        video.append(transform(image))
        image_ = transform_(image)
        image_ = image_.data.numpy().transpose(1, 2, 0) * 255
        image_list.append(image_)
    video = torch.stack(video).transpose(0, 1)
    video = video[None, :, :, :, :]
    #
    print('\nGrad-CAM')
    gcam = GradCAM(model=model, target_layers=[args.layer],
                   n_class=args.num_classes, cuda=args.cuda)
    #for i, (input, target) in enumerate(video):
    input_var = torch.autograd.Variable(video, volatile=True).cuda()

    gcam.forward(input_var)
    gcam.prob, gcam.idx = gcam.probs.data.squeeze().sort(0, True)
    spy_score = "{:.4f}".format(float(gcam.prob[1]) + label)
    #spy_score = spy_score + label
    #pred_clas = classes[gcam.idx[0]]
    # print

    # print("Predication class label: ")
    # print(gcam.idx[0])

    gcam.backward(idx=[8])  ##HACK: hhh
    gcam_map = gcam.generate(args.layer, 'raw')
    gcam_map = gcam_map.cpu()

    #gcam = torch.div(gcam, 255)
    #print (gcam)

    # from torch tensor to numpy
    gcam_data = gcam_map.data.numpy()
    # save grad-cam + ori_img

    # view_cam(image_list, gcam_data)
    # file_name = 'att_res/' + str(spy_score) + '.png'
    file_name = os.path.join(args.output_dir, args.name + '.png')
    save_cam(file_name, image_list, gcam_data)
