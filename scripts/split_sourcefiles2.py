import argparse
import os
import random

# round 1
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='sourcefiles/NewList1.txt')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--test_ratio', type=float, default=0.3)
args = parser.parse_args()

with open(args.input, 'r') as f:
    lines = [l.rstrip('\n') for l in f.readlines()]

partial_did = set()
for line in lines:
    partial_did.add(line[:line.find('Drive')])

random.seed(args.seed)
num_test = int(len(partial_did) * args.test_ratio)
test_pdids = random.sample(partial_did, num_test)

train_list = []
test_list = []
for line in lines:
    if line[:line.find('Drive')] in test_pdids:
        test_list.append(line)
    else:
        train_list.append(line)

with open(args.input.replace('.txt', f'_train_rng{args.seed}.txt'), 'w') as f:
    for line in train_list:
        f.write(line+'\n')

with open(args.input.replace('.txt', f'_test_rng{args.seed}.txt'), 'w') as f:
    for line in test_list:
        f.write(line+'\n')
