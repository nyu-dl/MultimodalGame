import numpy as np
import math
import argparse
import functools
import pickle
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from agents import reset_parameters_util
from misc import recursively_set_device

SHAPES = [None, 'circle', 'cross', 'ellipse', 'pentagon', 'rectangle', 'semicircle', 'square', 'triangle']
COLORS = [None, 'blue', 'cyan', 'gray', 'green', 'magenta', 'red', 'yellow']
SHAPES_DICT = {}
COLORS_DICT = {}
for i, s in enumerate(SHAPES):
    SHAPES_DICT[s] = i
for i, c in enumerate(COLORS):
    COLORS_DICT[c] = i


def load_classifier_dataset(data_path, batch_size, random_seed, shuffle, cuda, num_examples=1, binary=1, truncate_final_batch=False):
    data = pickle.load(open(data_path, 'rb'))
    x = []
    y = []
    s = []
    c = []
    for _, d in enumerate(data):
        if binary:
            for m1, m2 in zip(d["msg_1"], d["msg_2"]):
                x.append(m1)
                y.append(1)
                x.append(m2)
                y.append(0)
                es = torch.zeros(len(SHAPES))
                es[SHAPES_DICT[d['shape']]] = 1
                s.append(es)
                s.append(es)
                ec = torch.zeros(len(COLORS))
                ec[COLORS_DICT[d['color']]] = 1
                c.append(ec)
                c.append(ec)
        else:
            for m1, m2 in zip(d["probs_1"], d["probs_2"]):
                x.append(m1)
                y.append(1)
                x.append(m2)
                y.append(0)
                es = torch.zeros(len(SHAPES))
                es[SHAPES_DICT[d['shape']]] = 1
                s.append(es)
                s.append(es)
                ec = torch.zeros(len(COLORS))
                ec[COLORS_DICT[d['color']]] = 1
                c.append(ec)
                c.append(ec)
    assert len(x) == len(y)
    size = len(x)
    order = list(range(size - num_examples + 1))
    # print(num_examples, order[-1])
    # print(f"Dataset size: {size}")
    # Convert to numpy
    x = np.stack(x)
    y = np.stack(y)
    s = np.stack(s)
    c = np.stack(c)
    # print(f'x: {x.shape}, y: {y.shape}, s: {s.shape}, c: {c.shape}')
    # Shuffle
    if shuffle:
        random.seed(11 + random_seed)
        random.shuffle(order)
    # Generate batches
    num_batches = size // batch_size
    if truncate_final_batch:
        if size - (num_batches * batch_size) > 0:
            num_batches = num_batches + 1
    for i in range(num_batches):
        batch_indices = sorted(order[i * batch_size:(i + 1) * batch_size])
        batch = dict()
        # print(f'batch idxs: {batch_indices}')
        bx = None
        bs = None
        bc = None
        for j in range(num_examples):
            current_batch_indices = [b + j for b in batch_indices]
            # print(f'batch idxs: {batch_indices}')
            if j == 0:
                bx = Variable(torch.from_numpy(x[current_batch_indices]).float())
                bs = Variable(torch.from_numpy(s[current_batch_indices]).float())
                bc = Variable(torch.from_numpy(c[current_batch_indices]).float())
            else:
                _x = Variable(torch.from_numpy(x[current_batch_indices]).float())
                _s = Variable(torch.from_numpy(s[current_batch_indices]).float())
                _c = Variable(torch.from_numpy(c[current_batch_indices]).float())
                bx = torch.cat([bx, _x], dim=1)
                bs = torch.cat([bs, _x], dim=1)
                bc = torch.cat([bc, _x], dim=1)

        by = Variable(torch.from_numpy(y[batch_indices]).float())
        by = torch.unsqueeze(by, dim=1)
        if cuda:
            bx = bx.cuda()
            by = by.cuda()
            bs = bs.cuda()
            bc = bc.cuda()
        batch['x'] = bx
        batch['y'] = by
        batch['s'] = bs
        batch['c'] = bc
        # print(batch)
        yield batch


class Classifier(nn.Module):
    '''Processes sentence representations to the correct hidden dimension'''

    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.input_dim = input_dim
        self.layers = []
        self.layers += [nn.Linear(input_dim, math.floor(input_dim / 2))]
        self.layers += [nn.ReLU(inplace=True)]
        self.layers += [nn.Linear(math.floor(input_dim / 2), 1)]
        self.model = nn.Sequential(*self.layers)

    def reset_parameters(self):
        reset_parameters_util(self)

    def forward(self, x):
        x = self.model(x)
        return F.sigmoid(x)


def eval_dev(dataloader, net, epoch, i, moreinfo):
    net.eval()
    total = 0
    total_correct = 0
    mean_out = []
    for i_batch, batch in enumerate(dataloader):
        if moreinfo:
            x = torch.cat((batch['x'], batch['s'], batch['c']), dim=1)
        else:
            x = batch['x']
        # print(f'x: {x.size()}')
        out = net(x)
        target = torch.round(out)
        mean_out.append(out.data.sum() / out.data.size(0))
        correct = target == batch['y']
        correct = correct.sum()
        if i_batch == 0:
            print(x[:5])
            print(out[:25])
            # print(target)
            # print(batch['y'])
            print(correct)
        total += out.data.size(0)
        total_correct += correct.data[0]
        # print(f'total: {total}, correct: {total_correct}, correct: {correct}')
    print(f'Epoch: {epoch}, Batch: {i}, Total: {total}, Correct: {total_correct}, Accuracy: {total_correct/total}, Mean pred: {sum(mean_out) / len(mean_out)}')


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser(description='Analyze messages')
    parser.add_argument('--datapath', type=str, default="./logs/experiments_030718/big_valid_msg_eval_only_A_1_2_message_stats.pkl", help='Path to messages')
    parser.add_argument('--inputdim', type=int, default=8, help='input dim')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--cuda', type=int, default=0, help='whether to use cuda')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--seed', type=int, default=7, help='random see')
    parser.add_argument('--shuffle', type=int, default=1, help='whether to shuffle the dataset')
    parser.add_argument('--log', type=int, default=100, help='how often to log output')
    parser.add_argument('--epoch', type=int, default=20, help='how many epochs')
    parser.add_argument('--moreinfo', type=int, default=1, help='whether to use shape and color info')
    parser.add_argument('--num_examples', type=int, default=1, help='number of examples to use per datapoint')
    parser.add_argument('--binary', type=int, default=1, help='whether to use binary or probs')
    args = parser.parse_args()
    print(args)
    print(SHAPES_DICT)
    print(COLORS_DICT)
    learning_rate = args.lr
    cuda = args.cuda
    data_path = args.datapath
    batch_size = args.batch_size
    random_seed = args.seed
    shuffle = args.shuffle
    report_step = args.log
    max_epoch = args.epoch
    more_info = args.moreinfo
    num_examples = args.num_examples
    binary = args.binary
    if more_info:
        input_dim = (args.inputdim + len(SHAPES) + len(COLORS)) * num_examples
    else:
        input_dim = args.inputdim * num_examples

    # Net
    net = Classifier(input_dim)
    print("Net Architecture: {}".format(net))
    total_params = sum([functools.reduce(lambda x, y: x * y, p.size(), 1.0)
                        for p in net.parameters()])
    print("Total Parameters: {}".format(total_params))

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Loss
    loss = nn.BCELoss()

    # GPU support
    if cuda:
        net.cuda()
        loss = loss.cuda()

    # Training loop
    step = 0
    epoch = 0
    while epoch < max_epoch:
        # Iterate through batches
        dataloader = load_classifier_dataset(data_path, batch_size, random_seed, shuffle, cuda, num_examples=num_examples, binary=binary)
        for i_batch, batch in enumerate(dataloader):
            net.train()
            optimizer.zero_grad()
            if more_info:
                x = torch.cat((batch['x'], batch['s'], batch['c']), dim=1)
            else:
                x = batch['x']
            # print(f'x: {x.size()}')
            out = net(x)
            output = loss(out, batch['y'])
            output.backward()
            optimizer.step()
            step += 1
            # break
        evalloader = load_classifier_dataset(data_path, batch_size, random_seed, 0, cuda, num_examples=num_examples, binary=binary)
        eval_dev(evalloader, net, epoch, i_batch, args.moreinfo)
        epoch += 1
        # break
