import os
import sys

vid_path = './face_video'
print(os.listdir(vid_path))
vids = [f for f in os.listdir(vid_path)]
times = 1
layer = 'conv3b'
model_path = 'checkpoints/0325_passenger_nonhumor/100_net.pth'
print(vids)

for i in range(times):
    for vid in vids:
        video = os.path.join(vid_path, vid)
        cmd = 'python demo_generate_grad_cam_video.py --resume ' + model_path + ' --video ' + video + ' --layer ' + layer + ' --name ' + str(vid)
        print(cmd)
        os.system(cmd)
