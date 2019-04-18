import os
import glob
import skvideo.io
import skvideo.datasets
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default='/media/ligong/Picasso/Datasets/Jean/round1/videos_crop')
parser.add_argument('--dst', type=str, default='sourcefiles/_checklist.txt')
args = parser.parse_args()


def check(video_path):
    meta = skvideo.io.ffprobe(video_path)
    print(video_path)
    print(meta['video']['@duration'])
    # return True
    video_data = skvideo.io.vread(video_path)
    print(video_data.shape)
    return True


dids = [os.path.basename(s) for s in glob.glob(os.path.join(args.src, 'Dyad*'))]
with open(args.dst, 'w') as f:
    for did in dids:
        print('-> did: %s' % did)
        vidfiles = [os.path.basename(s) for s in glob.glob(os.path.join(args.src, did, '*.mp4'))]
        for filename in vidfiles:
            filename = os.path.join(args.src, did, filename)
            if not check(os.path.join(args.src, did, filename)):
                print(f'file {did}/{filename} damaged.')
                f.write(f'file {did}/{filename} damaged.\n')
