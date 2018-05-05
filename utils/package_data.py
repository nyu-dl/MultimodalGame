"""
Given a directory of images with following hierarchy:
    ./imgs/
        class-0/
            img-0-01
            img-0-02
            ...
        class-1/
            img-1-01
            ...
        ...

Create hdf5 representing each of these images preprocessed.

Layer Names
===========

ResNet-34

bn1           - (4L, 64L, 114L, 114L)
relu          - (4L, 64L, 114L, 114L)
maxpool       - (4L, 64L, 57L, 57L)
layer1        - (4L, 64L, 57L, 57L)
layer2        - (4L, 128L, 29L, 29L)
layer3        - (4L, 256L, 15L, 15L)
layer4_0_relu - (4L, 512L, 8L, 8L)
layer4_1_relu - (4L, 512L, 8L, 8L)
layer4_2      - (4L, 512L, 8L, 8L) #32768
layer4_2_relu - (4L, 512L, 8L, 8L)
avgpool       - (4L, 512L, 1L, 1L)
avgpool_512   - (4L, 512L)
fc            - (4L, 1000L)

"""


import sys
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.nn import AvgPool2d
import numpy as np
import h5py
from tqdm import tqdm
import gflags

FLAGS = gflags.FLAGS


def get_model():
    return eval("models.resnet{}".format(FLAGS.resnet))


def basic_block(layer, relu=False):
    def forward(x):
        residual = x

        out = layer.conv1(x)
        out = layer.bn1(out)
        out = layer.relu(out)

        out = layer.conv2(out)
        out = layer.bn2(out)

        if layer.downsample is not None:
            residual = layer.downsample(x)

        out += residual
        if relu:
            out = layer.relu(out)

        return out
    return forward


class FeatureModel(nn.Module):
    def __init__(self):
        super(FeatureModel, self).__init__()
        self.fn = get_model()(pretrained=True)

        # Turn off inplace
        for p in self.fn.modules():
            if "ReLU" in p.__repr__():
                p.inplace = False

    def forward(self, x, request=["layer4_2", "fc"]):
        model = self.fn

        ret = []

        layers = [
            (model.conv1, 'conv1'),
            (model.bn1, 'bn1'),
            (model.relu, 'relu'),
            (model.maxpool, 'maxpool'),
            (model.layer1, 'layer1'),
            (model.layer2, 'layer2'),
            (model.layer3, 'layer3'),
        ]

        if FLAGS.resnet == "34":
            layers += [
                (model.layer4[0], 'layer4_0_relu'),
                (model.layer4[1], 'layer4_1_relu'),
                (basic_block(model.layer4[2]), 'layer4_2'),
                (lambda x: model.layer4[2].relu(x), 'layer4_2_relu'),
            ]
        else:
            raise NotImplementedError()

        layers += [
            #(model.avgpool, 'avgpool'),
            # avgpool fix to get to original specifid dimensions - perhaps ResNet spec changed?
            (AvgPool2d(kernel_size=8, stride=1, padding=0, ceil_mode=False, count_include_pad=True), 'avgpool_4D'),
            (lambda x: x.view(x.size(0), -1), 'avgpool_512'),
            (model.fc, 'fc'),
        ]

        for module, name in layers:
            #print(f'name: {name}\n module: {module}')
            #print(f'x shape before: {x.size()}')
            x = module(x)
            #print(f'x shape after: {x.size()}')
            # print(" N", x.data.numel())
            # print("<0", (x.data < 0.).sum())
            # print("=0", (x.data == 0.).sum())
            # print("{} - {}".format(name, tuple(x.size())))
            if name in request:
                # print(f'{name} in request, appending...')
                ret.append(x)

        return ret


def label_mapping(desc_path):
    label_to_id = dict()
    with open(desc_path) as f:
        for line in f:
            line = line.strip()
            label_id, label, desc = line.split(',')
            label_to_id[label] = int(label_id)
    return label_to_id


def custom_dtype(outp, request):
    schema = [('Location', np.str_, 50),
             ('Target', 'i')]
    for o, r in zip(outp, request):
        size = tuple([1] + list(o.shape)[1:])
        schema.append((r, np.float32, size))
    dtype = np.dtype(schema)
    return dtype


def multi_split(outp, batch_size):
    return [np.split(o, batch_size) for o in outp]


def run():
    dtype = None

    # Model Initialization
    model = FeatureModel()
    model.fn.eval()
    model.eval()

    if FLAGS.cuda:
        model.fn.cuda()
        model.cuda()

    # Load dataset and transform
    dataset = dset.ImageFolder(root=FLAGS.load_imgs,
                               transform=transforms.Compose([
                               transforms.Resize(227),
                               transforms.CenterCrop(227),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])
                              )

    # Read images
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=FLAGS.batch_size,
                                             shuffle=False)

    # Used to get label ids
    label_to_id = label_mapping(FLAGS.load_desc)

    # Only keep relevant output
    request = FLAGS.request.split(',')

    # Preprocess images to new vector representation
    data = []

    targets = []
    locations = []
    other = dict()

    def data_it(dataloader):
        _it = iter(dataloader)

        while True:
            try:
                ret = next(_it)
            except StopIteration:
                break
            except BaseException as e:
                continue
            yield ret

    for i, img in tqdm(enumerate(data_it(dataloader))):
        tensor, target = img
        if i == 0:
            print('')
            print(tensor.size())
            print(request)
            print(target)
        if FLAGS.cuda:
            tensor = tensor.cuda()
        outp = model(Variable(tensor), request)

        np_outp = [o.data.cpu().numpy() for o in outp]

        batch_size = np_outp[0].shape[0]
        offset = i * batch_size

        for j, o in enumerate(zip(*multi_split(np_outp, batch_size))):
            filename = dataset.imgs[offset + j][0]
            parts = filename.split(os.sep)
            label = parts[-2] # the label as a string
            loc = parts[-1] # something like '1-100-251756690_e68ac649e3_z.jpg'
            label_id = label_to_id[label] # use the label id specified by the desc file

            # Save Image
            row = tuple([loc, label_id] + list(o))
            data.append(row)
            targets.append(label_id)
            locations.append(loc)
            for r, oo in zip(request, list(o)):
                other.setdefault(r, []).append(oo)

    # Save hdf5 file
    hdf5_f = h5py.File(FLAGS.save_hdf5, 'w')
    hdf5_f.create_dataset("Target", data=np.array(targets))
    # Location format not saveable to hdf5
    #dt = np.dtype(str)
    #locations = np.array(locations).astype(dt)
    #hdf5_f.create_dataset("Location", (len(locations),), dtype=str)
    #hdf5_f.create_dataset("Location", data=locations))
    for r in request:
        hdf5_f.create_dataset(r, data=np.array(other[r]))
    hdf5_f.close()


if __name__ == "__main__":
    # Settings
    gflags.DEFINE_string("load_desc", "descriptions.csv", "Path to description file.")
    gflags.DEFINE_string("load_imgs", "./imgs/train", "Path to input data.")
    gflags.DEFINE_string("save_hdf5", "train.hdf5", "Path to store new dataset.")
    gflags.DEFINE_integer("batch_size", 4, "Minibatch size.")
    gflags.DEFINE_enum("resnet", "34", ["18", "34", "50", "101", "152"], "Specify Resnet variant.")
    gflags.DEFINE_string("request", "layer4_2,avgpool_512,fc", "Run feature model. Save specified layer output.")
    gflags.DEFINE_boolean("cuda", False, "")

    FLAGS(sys.argv)

    run()
