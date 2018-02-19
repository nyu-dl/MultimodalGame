import logging
import sys
import random
import torch
import math
import pprint
import string
import gflags
import numpy as np

from torchvision.utils import save_image
from torch.autograd import Variable
from skimage.transform import resize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from shapeworld import dataset
from misc import embed, cbow_general
from utils.package_data import FeatureModel

FLAGS = gflags.FLAGS

FORMAT = '[%(asctime)s %(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT)
debuglogger = logging.getLogger('main_logger')
debuglogger.setLevel('DEBUG')

SHAPES = ['circle', 'cross', 'ellipse', 'pentagon', 'rectangle', 'semicircle', 'square', 'triangle']
COLORS = ['blue', 'cyan', 'gray', 'green', 'magenta', 'red', 'yellow']


def clean_and_tokenize(desc):
    words = word_tokenize(desc.lower())  # lowercase and tokenize
    words = [w for w in words if w not in string.punctuation]
    return words


def upscale(ims):
    '''Upscales images to ResNet input size'''
    (bs, width, height, ch) = ims.shape
    new_ims = np.zeros((bs, 227, 227, 3))
    for i in range(bs):
        new_ims[i] = resize(ims[i], (227, 227))
    return new_ims


def convert_texts(texts, word_dict=None):
    ''' Takes a dataset of n texts per example. Each example is a list of strings
    Returns: texts converted to lists of ints
             If word_dict is not None (e.g. validation data texts) then word_dict is used to make the conversion and blank word2id and id2words are returned
             If word_dict is None then the function also builds and returns a word2id idct, and an id2word dict
    '''
    word2id = {'UNK': {"id": 0}} if word_dict is None else word_dict
    id2word = {0: 'UNK'} if word_dict is None else None
    texts_ints = []
    vocab_size = 0
    num_texts = 0
    for t in texts:
        num_texts += 1
        curr_t = []
        for elem in t:
            curr_elem = []
            desc = clean_and_tokenize(elem)
            for w in desc:
                if word_dict is not None:
                    # Use existing vocab
                    if w in word2id:
                        curr_elem.append(word2id[w]["id"])
                    else:
                        curr_elem.append(word2id['UNK']["id"])
                else:
                    # Build vocab
                    if w not in word2id:
                        vocab_size += 1
                        word2id[w] = {"id": vocab_size}
                        id2word[vocab_size] = w
                    curr_elem.append(word2id[w]["id"])
            assert len(curr_elem) == len(desc)
            curr_t.append(curr_elem)
            curr_elem = []
        texts_ints.append(curr_t)
        curr_t = []
    debuglogger.info(f'Num_examples: {num_texts}, Vocab size: {vocab_size}')
    return texts_ints, word2id, id2word


def load_shapeworld_dataset(data_path, embed_path, mode, size, ds_type, name, batch_size, random_seed, shuffle, img_feats, cuda, truncate_final_batch=False):
    """
    Reads ShapeWorld dataset into random num_batches
    Args:
        - data_path: path to folder containing the shapeworld data
        - embed_path: path to folder containing pretrained word vectors
        - mode: 'train', 'eval', or 'test'
        - size: size of dataset
        - ds_type: problem type e.g. 'agreement'
        - name: name of dataset, e.g. 'oneshape_simple_textselect'
        - batch_size: size of each batch
        - random_seed: int to use to set random seed
        - shuffle: whether to shuffle the dataset
        - img_feats: what type of image features to use e.g. 'avgpool_512', 'layer4_2'
        - whether to use cuda
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
    texts_str: set of natural language descriptions of the image (only one is correct)
    texts_int: set of integer descriptions of the image (only one is correct)
    texts_vec: vector representation of the set of natural language image descriptions for each example
    texts_extra: dict for individual word vectors for each description for each example and their corresponding lengths
    target: index of correct textual description
    """
    # Read data
    debuglogger.info(f'Reading in dataset...')
    load_cmd = 'load(' + data_path + ')'
    data = dataset(dtype=ds_type, name=name, config=load_cmd)
    generated = data.generate(n=size, mode=mode)
    debuglogger.info(f'Dataset read...')
    order = list(range(size))
    assert len(generated['texts_str']) == size

    # Convert texts to vector
    texts_str = generated['texts_str']
    texts_int, word2id, id2word = convert_texts(texts_str)
    word2id = embed(word2id, embed_path)

    # Create feature extraction model
    model = FeatureModel()
    model.fn.eval()
    model.eval()

    if cuda:
        model.fn.cuda()
        model.cuda()

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
        debuglogger.debug(f'batch idxs: {batch_indices}')

        # Upscale images and convert to tensors
        ims = generated['world'][batch_indices]
        ims = upscale(ims)
        batch['images'] = torch.from_numpy(ims).float().permute(0, 3, 1, 2)

        # Extract target and texts
        batch['target'] = torch.from_numpy(generated['target'][batch_indices]).long()
        batch["texts_str"] = [generated['texts_str'][j] for j in batch_indices]
        batch["caption_str"] = [generated['caption_str'][j] for j in batch_indices]
        batch["texts_int"] = [texts_int[j] for j in batch_indices]

        # Get shape and color for batch
        batch["shapes"] = []
        batch["colors"] = []
        for cap in batch["caption_str"]:
            cap = cap.split()
            color = None
            shape = None
            for w in cap:
                if w in SHAPES:
                    shape = w
                if w in COLORS:
                    color = w
            batch["shapes"].append(shape)
            batch["colors"].append(color)
        assert len(batch["shapes"]) == batch_size
        assert len(batch["colors"]) == batch_size

        # Generate p
        batch['p'] = torch.from_numpy(np.random.rand(batch_size))
        # debuglogger.debug(f'p: {batch["p"]}')

        # Mask images
        (bs, ch, width, height) = batch['images'].shape
        mask = torch.ones(bs, ch, width, height)
        cutoffs = (width * batch["p"]).int().clamp(0, 226).numpy().tolist()
        debuglogger.debug(f'cutoffs: {cutoffs}')
        for i_c, c in enumerate(cutoffs):
            mask[i_c, :, :, c:] = 0
        batch['masked_im_1'] = torch.mul(mask, batch['images']) + (1 - mask)
        batch['masked_im_2'] = torch.mul(1 - mask, batch['images']) + mask
        if i == 0:
            # Save example batch
            save_image(batch['images'], data_path + '/example_ims_orig.png', pad_value=0.5)
            save_image(batch['masked_im_1'], data_path + '/example_ims_1.png', pad_value=0.5)
            save_image(batch['masked_im_2'], data_path + '/example_ims_2.png', pad_value=0.5)

        # Build descriptions
        desc_cbow, desc_set, desc_set_lens = cbow_general(batch["texts_int"], word2id, id2word)
        batch["texts_vec"] = desc_cbow
        batch["texts_extra"] = {"desc_set": desc_set,
                                "desc_set_lens": desc_set_lens}

        # Extract image feats
        m_im_1 = Variable(batch['masked_im_1'])
        m_im_2 = Variable(batch['masked_im_2'])
        if cuda:
            m_im_1 = m_im_1.cuda()
            m_im_2 = m_im_2.cuda()
        batch["im_feats_1"] = (model(m_im_1, request=img_feats)[0]).detach()
        batch["im_feats_2"] = (model(m_im_2, request=img_feats)[0]).detach()
        yield batch


if __name__ == "__main__":
    # Settings
    gflags.DEFINE_enum("resnet", "34", ["18", "34", "50", "101", "152"], "Specify Resnet variant.")
    FLAGS(sys.argv)

    data_path = './ShapeWorld/data/oneshape_simple_textselect'
    embed_path = './glove.6B/glove.6B.100d.txt'
    mode = 'train'
    size = 100
    ds_type = 'agreement'
    name = 'oneshape_simple_textselect'
    batch_size = 8
    random_seed = 12
    img_feats = 'avgpool_512'
    shuffle = True
    cuda = False
    dataloader = load_shapeworld_dataset(data_path, embed_path, mode, size, ds_type, name, batch_size, random_seed, shuffle, img_feats, cuda, truncate_final_batch=False)
    for i_batch, batch in enumerate(dataloader):
        pprint.pprint(batch)
        break
