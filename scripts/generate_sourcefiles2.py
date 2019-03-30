import argparse
import os

# round 1
parser = argparse.ArgumentParser()
parser.add_argument('--delta', type=float, default=0.125, help='step size to discretize GT scores')
parser.add_argument('--input', type=str, default='sourcefiles/NewList1.txt')
parser.add_argument('--output', type=str, default='sourcefiles/driver_frame1.txt')
parser.add_argument('--driver', type=str, default='driver')
parser.add_argument('--humor', type=str, default='humor')
parser.add_argument('--frame', type=str, default='frame1')
args = parser.parse_args()

with open(args.input, 'r') as f:
    lines = [l.rstrip('\n') for l in f.readlines()]

if args.frame == 'frame1':
    frame = 'frame1'
else:
    frame = 'frame2'

if args.driver == 'driver' and args.humor == 'humor':
    # driver humor
    with open(args.output, 'w') as f:
        for line in lines:
            line_ = line.split(',')
            did = line_[0].strip(' ')
            label = int(float(line_[3]) / args.delta)
            f.write('%s/%s_driver.mp4 %s/%s_passenger.mp4 %d\n' % (did, frame, did, frame, label))
elif args.driver == 'driver' and args.humor == 'nonhumor':
    # driver non-humor
    with open(args.output, 'w') as f:
        for line in lines:
            line_ = line.split(',')
            did = line_[0].strip(' ')
            label = int(float(line_[4]) / args.delta)
            f.write('%s/%s_driver.mp4 %s/%s_passenger.mp4 %d\n' % (did, frame, did, frame, label))
elif args.driver == 'passenger' and args.humor == 'humor':
    # passenger humor
    with open(args.output, 'w') as f:
        for line in lines:
            line_ = line.split(',')
            did = line_[0].strip(' ')
            label = int(float(line_[1]) / args.delta)
            f.write('%s/%s_driver.mp4 %s/%s_passenger.mp4 %d\n' % (did, frame, did, frame, label))
else:
    # passenger non-humor
    with open(args.output, 'w') as f:
        for line in lines:
            line_ = line.split(',')
            did = line_[0].strip(' ')
            label = int(float(line_[2]) / args.delta)
            f.write('%s/%s_driver.mp4 %s/%s_passenger.mp4 %d\n' % (did, frame, did, frame, label))
