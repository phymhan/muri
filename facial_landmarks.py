from __future__ import print_function
import os
import torch
from enum import Enum
from skimage import io
from skimage import color
import numpy as np
import cv2
from face_alignment.utils import *
from copy import deepcopy


def get_heatmap_from_image(image_or_path, fa, detected_faces=None, ret_pts=False):
    """Predict the landmarks for each face present in the image.

    This function predicts a set of 68 2D or 3D images, one for each image present.
    If detect_faces is None the method will also run a face detector.

     Arguments:
        image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

    Keyword Arguments:
        detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
        in the image (default: {None})
    """
    if isinstance(image_or_path, str):
        try:
            image = io.imread(image_or_path)
        except IOError:
            print("error opening file :: ", image_or_path)
            return None
    else:
        # image = deepcopy(image_or_path)
        image = image_or_path

    if image.ndim == 2:
        image = color.gray2rgb(image)
    elif image.ndim == 4:
        image = image[..., :3]

    if detected_faces is None:
        detected_faces = fa.face_detector.detect_from_image(image[..., ::-1].copy())

    if len(detected_faces) == 0:
        print("Warning: No faces were detected.")
        if ret_pts:
            return None, None
        return None

    with torch.no_grad():
        heatmaps_all = []
        points_all = []
        for i, d in enumerate(detected_faces):
            center = torch.FloatTensor(
                [d[2] - (d[2] - d[0]) / 2.0, d[3] -
                 (d[3] - d[1]) / 2.0])
            center[1] = center[1] - (d[3] - d[1]) * 0.12
            scale = (d[2] - d[0] +
                     d[3] - d[1]) / fa.face_detector.reference_scale

            inp = crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose(
                (2, 0, 1))).float()
            inp = inp.to(fa.device)
            # inp = inp.cuda()
            inp.div_(255.0).unsqueeze_(0)

            out = fa.face_alignment_net(inp)[-1].detach()
            if fa.flip_input:
                out += flip(fa.face_alignment_net(flip(inp))
                            [-1].detach(), is_label=True)
            out = out.cpu()

            pts, pts_img = get_preds_fromhm(out, center, scale)
            pts_img = pts_img.view(68, 2)
            # pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

            heatmaps = np.zeros((68, image.shape[0], image.shape[1]), dtype=np.float32)
            for j in range(68):
                if pts_img[j, 0] > 0:
                    heatmaps[j] = draw_gaussian(heatmaps[j], pts_img[j], 2)
            heatmaps = torch.from_numpy(heatmaps).unsqueeze_(0)
            heatmaps = torch.sum(heatmaps, dim=1)
            # heatmaps.requires_grad_(False)
            heatmaps_all.append(heatmaps)
            points_all.append(pts_img)
    if ret_pts:
        return heatmaps_all, points_all
    return heatmaps_all
