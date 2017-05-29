#!/bin/bash

# This is an example of how one might build train/dev/test splits for MultimodalGame.

# Download imagenet urls.
wget http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz
tar -xvzf imagenet_fall11_urls.tgz

# Download images and build descriptions.
python download_data.py \
    -cmd_urls \
    -cmd_split \
    -cmd_desc \
    -cmd_download

# Build hdf5 files.
python package_data.py -cuda -load_desc descriptions.csv -load_imgs ./imgs/train -save_hdf5 train.hdf5
python package_data.py -cuda -load_desc descriptions.csv -load_imgs ./imgs/dev -save_hdf5 dev.hdf5
python package_data.py -cuda -load_desc descriptions.csv -load_imgs ./imgs/test -save_hdf5 test.hdf5

# Finished!
