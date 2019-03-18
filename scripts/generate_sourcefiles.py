## round 1
delta = 0.125
mode = '_val'
mode_ = mode[1:]+'_'
csvfile = 'sourcefiles/NewList%s.txt' % mode
with open(csvfile, 'r') as f:
    lines = [l.rstrip('\n') for l in f.readlines()]

# driver -> driver humor
with open('sourcefiles/%sd_dh_f1.txt' % mode_, 'w') as f:
    for line in lines:
        line_ = line.split(',')
        did = line_[0].strip(' ')
        label = int(float(line_[3])/delta)
        f.write('%s/frame1_driver.avi %d\n' % (did, label))

with open('sourcefiles/%sd_dh_f2.txt' % mode_, 'w') as f:
    for line in lines:
        line_ = line.split(',')
        did = line_[0].strip(' ')
        label = int(float(line_[3])/delta)
        f.write('%s/frame2_driver.avi %d\n' % (did, label))

# driver -> driver non-humor
with open('sourcefiles/%sd_dn_f1.txt' % mode_, 'w') as f:
    for line in lines:
        line_ = line.split(',')
        did = line_[0].strip(' ')
        label = int(float(line_[4])/delta)
        f.write('%s/frame1_driver.avi %d\n' % (did, label))

with open('sourcefiles/%sd_dn_f2.txt' % mode_, 'w') as f:
    for line in lines:
        line_ = line.split(',')
        did = line_[0].strip(' ')
        label = int(float(line_[4])/delta)
        f.write('%s/frame2_driver.avi %d\n' % (did, label))

# pass -> pass humor
with open('sourcefiles/%sp_ph_f1.txt' % mode_, 'w') as f:
    for line in lines:
        line_ = line.split(',')
        did = line_[0].strip(' ')
        label = int(float(line_[1])/delta)
        f.write('%s/frame1_passenger.avi %d\n' % (did, label))

with open('sourcefiles/%sp_ph_f2.txt' % mode_, 'w') as f:
    for line in lines:
        line_ = line.split(',')
        did = line_[0].strip(' ')
        label = int(float(line_[1])/delta)
        f.write('%s/frame2_passenger.avi %d\n' % (did, label))

# pass -> pass non-humor
with open('sourcefiles/%sp_pn_f1.txt' % mode_, 'w') as f:
    for line in lines:
        line_ = line.split(',')
        did = line_[0].strip(' ')
        label = int(float(line_[2])/delta)
        f.write('%s/frame1_passenger.avi %d\n' % (did, label))

with open('sourcefiles/%sp_pn_f2.txt' % mode_, 'w') as f:
    for line in lines:
        line_ = line.split(',')
        did = line_[0].strip(' ')
        label = int(float(line_[2])/delta)
        f.write('%s/frame2_passenger.avi %d\n' % (did, label))