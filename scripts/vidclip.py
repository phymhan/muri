import os
import glob
from datetime import datetime
import skvideo.io

rootdir = '/dresden/users/lh599/Research/Jean/data/RWNVDE_Data/FormalDataCollection/synced/'
outdir = '/dresden/users/lh599/Research/Jean/data_1/round1/videos_mp4/'
csvfile = 'EventList_RecMem.csv'
newfile1 = 'NewList1.txt'
newfile2 = 'NewList2.txt'
badfile = 'BadList.txt'
FMT = '%H:%M:%S'


def time2sec(s):
    s_ = s.split(':')
    return float(s_[0])*3600 + float(s_[1])*60 + float(s_[2])


if not os.path.exists(outdir):
    os.makedirs(outdir)

folders = [os.path.basename(s) for s in glob.glob(os.path.join(rootdir, 'Dyad*'))]
# print(folders)

with open(csvfile, 'r') as f:
    thelist = [l.strip('\n') for l in f.readlines()]

newlines1 = []
newlines2 = []
badlines = []

for line in thelist[1:]:
    line_ = line.split(',')
    did = line_[0].strip('"')
    print(did)
    if line_[1].strip('"') == 'NA':
        continue
    frame1_start = ':'.join(line_[1].strip('"').split('_')[1:])
    frame1_stop  = ':'.join(line_[2].strip('"').split('_')[1:])
    frame2_start = ':'.join(line_[3].strip('"').split('_')[1:])
    frame2_stop  = ':'.join(line_[4].strip('"').split('_')[1:])
    duration1 = str(datetime.strptime(frame1_stop, FMT)-datetime.strptime(frame1_start, FMT))
    duration2 = str(datetime.strptime(frame2_stop, FMT)-datetime.strptime(frame2_start, FMT))
    folder0 = list(filter(lambda x: x.startswith(did), folders))[0]
    if len(glob.glob(os.path.join(rootdir, folder0, 'Vehicle', '*.mp4'))) > 0:
        folder1 = '.'
    else:
        nextlevelfolders = [os.path.basename(s) for s in glob.glob(os.path.join(rootdir, folder0, 'Vehicle', 'Dyad*'))]
        folder1 = list(filter(lambda x: x.startswith(did), nextlevelfolders))[0]
    vidfiles = [os.path.basename(s) for s in glob.glob(os.path.join(rootdir, folder0, 'Vehicle', folder1, '*.mp4'))]

    vidfile_driver = list(filter(lambda x: x.split('_')[-1].startswith('Driver'), vidfiles))
    if len(vidfile_driver) > 0:
        vidfile_driver = vidfile_driver[0]
        vidfile_passenger = 'Passenger'
    else:
        vidfile_driver = list(filter(lambda x: x.split('_')[-1].startswith('View 3'), vidfiles))
        if len(vidfile_driver) > 0:
            vidfile_driver = vidfile_driver[0]
            vidfile_passenger = 'View 4'
        else:
            continue
    vidfile_passenger = list(filter(lambda x: x.split('_')[-1].startswith(vidfile_passenger), vidfiles))[0]
    outdir_did = os.path.join(outdir, did)
    print(outdir_did)
    if not os.path.exists(outdir_did):
        os.makedirs(outdir_did)

    fine1 = True
    fine2 = True

    # driver
    filepath_in = os.path.join(rootdir, folder0, 'Vehicle', folder1, vidfile_driver)
    meta = skvideo.io.ffprobe(filepath_in)
    if 'video' in meta:
        if float(meta['video']['@duration']) >= time2sec(frame1_stop):
            # driver frame 1
            filepath_out = os.path.join(outdir_did, 'frame1_driver.mp4')
            os.system('ffmpeg -ss %s -t %s -i "%s" -c copy "%s"' % (frame1_start, duration1, filepath_in, filepath_out))
        else:
            badlines.append('%s, frame1 exceeds duration of driver video' % did)
            fine1 = False
        if float(meta['video']['@duration']) >= time2sec(frame2_stop):
            # driver frame 2
            filepath_out = os.path.join(outdir_did, 'frame2_driver.mp4')
            os.system('ffmpeg -ss %s -t %s -i "%s" -c copy "%s"' % (frame2_start, duration2, filepath_in, filepath_out))
        else:
            badlines.append('%s, frame2 exceeds duration of driver video' % did)
            fine2 = False
    else:
        fine1 = False
        fine2 = False
        badlines.append('%s, no meta info for driver video' % did)

    # passenger
    filepath_in = os.path.join(rootdir, folder0, 'Vehicle', folder1, vidfile_passenger)
    meta = skvideo.io.ffprobe(filepath_in)
    if 'video' in meta:
        if float(meta['video']['@duration']) >= time2sec(frame1_stop):
            # passenger frame 1
            filepath_out = os.path.join(outdir_did, 'frame1_passenger.mp4')
            os.system('ffmpeg -ss %s -t %s -i "%s" -c copy "%s"' % (frame1_start, duration1, filepath_in, filepath_out))
        else:
            badlines.append('%s, frame1 exceeds duration of passenger video' % did)
            fine1 = False
        if float(meta['video']['@duration']) >= time2sec(frame2_stop):
            # passenger frame 2
            filepath_out = os.path.join(outdir_did, 'frame2_passenger.mp4')
            os.system('ffmpeg -ss %s -t %s -i "%s" -c copy "%s"' % (frame2_start, duration2, filepath_in, filepath_out))
        else:
            badlines.append('%s, frame2 exceeds duration of passenger video' % did)
            fine2 = False
    else:
        fine1 = False
        fine2 = False
        badlines.append('%s, no meta info for passenger video' % did)

    if fine1:
        newlines1.append('%s, %s, %s, %s, %s' % (did, line_[5], line_[6], line_[7], line_[8].strip('\n')))
    if fine2:
        newlines2.append('%s, %s, %s, %s, %s' % (did, line_[5], line_[6], line_[7], line_[8].strip('\n')))

with open(newfile1, 'w') as f:
    for l in newlines1:
        f.write(l+'\n')

with open(newfile2, 'w') as f:
    for l in newlines2:
        f.write(l+'\n')

with open(badfile, 'w') as f:
    for l in badlines:
        f.write(l+'\n')
