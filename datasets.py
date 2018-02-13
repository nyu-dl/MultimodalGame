from shapeworld import dataset
import logging
import sys
import random
import torch
from torchvision.utils import save_image
import numpy as np
from skimage.transform import resize
import math


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
debuglogger = logging.getLogger('main_logger')
debuglogger.setLevel(10)


def upscale(ims):
    (bs, width, height, ch) = ims.shape
    new_ims = np.zeros((bs, 227, 227, 3))
    for i in range(bs):
        new_ims[i] = resize(ims[i], (227, 227))
    return new_ims


def load_shapeworld_dataset(data_path, mode, size, ds_type, name, batch_size, random_seed, shuffle, img_feats, truncate_final_batch=False):
    """
    Reads ShapeWorld dataset into random num_batches
    Args:
        - data_path: path to folder containing the shapeworld data
        - mode: 'train', 'eval', or 'test'
        - size: size of dataset
        - ds_type: problem type e.g. 'agreement'
        - name: name of dataset, e.g. 'oneshape_simple_textselect'
        - batch_size: size of each batch
        - random_seed: int to use to set random seed
        - shuffle: whether to shuffle the dataset
        - img_feats: what type of image features to use e.g. 'avgpool_512', 'layer4_2'
        - truncate_final_batch: whether to use a smaller final batch or not

    Each batch is a dict consisting of:
        batch = { "im_feats_1": im_feats_1,
                  "im_feats_2": im_feats_2,
                  "im_1": masked_im_1,
                  "im_2": masked_im_2,
                  "p": p,
                  "texts_str": natural_lang_desc_texts,
                  "texts_vec": texts_vec,
                  "target": targets,
        }

    im_feats_1: image features for agent 1
    im_feats_1: image features for agent 2
    masked_im_1: masked input image received by agent 1
    masked_im_2: masked input image received by agent 2
    p: percentage of the input image received by agent 1. Agent 2 received (1 - p)
    natural_lang_desc_texts: set of natural language descriptions of the image (only one is correct)
    texts_vec: vector representation of the set of natural language image descriptions
    targets: index of correct textual description
    """
    # Read data
    load_cmd = 'load(' + data_path + ')'
    data = dataset(dtype=ds_type, name=name, config=load_cmd)
    generated = data.generate(n=size, mode=mode)
    order = list(range(size))
    assert len(generated['texts_str']) == size

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
        debuglogger.info(f'batch idxs: {batch_indices}')

        # Upscale images and convert to tensors
        ims = generated['world'][batch_indices]
        ims = upscale(ims)
        batch['images'] = torch.from_numpy(ims).float().permute(0, 3, 1, 2)

        # Extract target and texts
        batch['target'] = torch.from_numpy(generated['target'][batch_indices]).int()
        batch["texts_str"] = [generated['texts_str'][j] for j in batch_indices]

        # Generate p
        batch['p'] = torch.from_numpy(np.random.rand(batch_size))
        debuglogger.debug(f'p: {batch["p"]}')

        # Mask images
        (bs, ch, width, height) = batch['images'].shape
        mask = torch.ones(bs, ch, width, height)
        mask_replace = torch.zeros(bs, ch, width, height)
        cutoffs = (width * batch["p"]).int().clamp(0, 226).numpy().tolist()
        debuglogger.debug(f'cutoffs: {cutoffs}')
        for i_c, c in enumerate(cutoffs):
            mask[i_c, :, :, c:] = 0
            mask_replace[i_c, :, :, c:] = 1
        batch['im_1'] = torch.mul(mask, batch['images']) + mask_replace
        batch['im_2'] = torch.mul(1 - mask, batch['images']) + (1 - mask_replace)
        if i == 0:
            save_image(batch['images'], data_path + '/example_ims_orig.png', pad_value=0.5)
            save_image(batch['im_1'], data_path + '/example_ims_1.png', pad_value=0.5)
            save_image(batch['im_2'], data_path + '/example_ims_2.png', pad_value=0.5)

        # Extract image feats
        # TODO
        # Convert texts to vector
        # TODO

        yield batch


if __name__ == "__main__":
    data_path = '/Users/lauragraesser/Documents/NYU_Courses/Comms/data/oneshape_simple_textselect'
    mode = 'train'
    size = 100
    ds_type = 'agreement'
    name = 'oneshape_simple_textselect'
    batch_size = 8
    random_seed = 12
    img_feats = 'avgpool_512'
    shuffle = True
    dataloader = load_shapeworld_dataset(data_path, mode, size, ds_type, name, batch_size, random_seed, shuffle, img_feats, truncate_final_batch=False)
    for i_batch, batch in enumerate(dataloader):
        pprint.pprint(batch)
        break
