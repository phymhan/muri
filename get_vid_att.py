import os
import sys
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_path', type=str, default='./face_video')
    parser.add_argument('--output_dir', type=str, default='att_res')
    parser.add_argument('--times', type=int, default=1)
    parser.add_argument('--layer', type=str, default='conv3b')
    parser.add_argument('--name', type=str, default='experiment')
    parser.add_argument('--which_epoch', type=str, default='100')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--arch', type=int, default=1)
    parser.add_argument('--comb', type=int, default=2)
    parser.add_argument('--fc_dim', type=int, default=4096)
    args = parser.parse_args()

    vids = [f for f in os.listdir(args.vid_path)]
    model_path = os.path.join(args.checkpoint_dir, args.name, '%s_net.pth' % args.which_epoch)

    for i in range(args.times):
        for vid in vids:
            video = os.path.join(args.vid_path, vid)
            cmd = 'python demo_generate_grad_cam_video.py --resume ' + model_path + ' --video ' + video + \
                  ' --layer ' + args.layer + ' --name ' + str(vid) + ' --output_dir ' + args.output_dir + \
                  ' --arch ' + str(args.arch) + ' --comb ' + str(args.comb) + ' --fc_dim ' + str(args.fc_dim)
            print(cmd)
            os.system(cmd)
