import torch
from torch.autograd import Variable
import numpy as np
import datetime
import os
import sys
import json
import h5py
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import itertools
import logging

try:
    from visdom import Visdom
except:
    pass


FORMAT = '[%(asctime)s %(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT)
debuglogger = logging.getLogger('main_logger')
debuglogger.setLevel('DEBUG')


"""
Notes

A. Loading Description File and Assigning Labels

Description File should be in the CSV format,

    label_id,label,description

Concretely,

    3,aardvark,nocturnal burrowing mammal of the grasslands of Africa that feeds on termites; sole extant representative of the order Tubulidentata
    11,armadillo,burrowing chiefly nocturnal mammal with body covered with strong horny plates

Note that the label_id need not be ordered nor within any predefined range. Should simply match
the "Target" attribute of the data.hdf5. Once the dataset has been loaded, the label_ids will be
converted into range(len(classes)), and there will be a mapping from the label_ids to this range.

"""


def recursively_set_device(inp, gpu):
    if hasattr(inp, 'keys'):
        for k in inp.keys():
            inp[k] = recursively_set_device(inp[k], gpu)
    elif isinstance(inp, list):
        return [recursively_set_device(ii, gpu) for ii in inp]
    elif isinstance(inp, tuple):
        return (recursively_set_device(ii, gpu) for ii in inp)
    elif hasattr(inp, 'cpu'):
        if gpu >= 0:
            inp = inp.cuda()
        else:
            inp = inp.cpu()
    return inp


def torch_save(filename, data, models_dict, optimizers_dict, gpu):
    models_to_save = {k: recursively_set_device(
        v.state_dict(), gpu=-1) for k, v in models_dict.items()}
    optimizers_to_save = {k: recursively_set_device(
        v.state_dict(), gpu=-1) for k, v in optimizers_dict.items()}

    # Always sends Tensors to CPU.
    torch.save({
        'data': data,
        'optimizers': optimizers_to_save,
        'models': models_to_save,
    }, filename)

    if gpu >= 0:
        for m in models_dict.values():
            recursively_set_device(m.state_dict(), gpu=gpu)
        for o in optimizers_dict.values():
            recursively_set_device(o.state_dict(), gpu=gpu)


def torch_load(filename, models_dict, optimizers_dict):
    filename = os.path.expanduser(filename)

    if not os.path.exists(filename):
        raise Exception("File does not exist: " + filename)

    checkpoint = torch.load(filename)

    for k, v in models_dict.items():
        v.load_state_dict(checkpoint['models'][k])

    for k, v in optimizers_dict.items():
        v.load_state_dict(checkpoint['optimizers'][k])

    return checkpoint['data']


class VisdomLogger(object):
    """
    Logs data to visdom

    """

    def __init__(self, env, experiment_name, minimum=2, enabled=False):
        self.enabled = enabled
        self.experiment_name = experiment_name
        self.env = env
        self.minimum = minimum

        self.q = dict()

        if enabled:
            self.viz = Visdom()

    def get_metrics(self, key, val, step):
        metric = self.q.setdefault(key, [])
        metric.append((step, val))
        if len(metric) >= self.minimum:
            del self.q[key]
            return metric
        return None

    def viz_success(self, win):
        if win == "win does not exist":
            return False
        return True

    def log(self, key, val, step):
        if not self.enabled:
            return

        metrics = self.get_metrics(key, val, step)

        # Visdom requires 2+ data points to be written.
        if metrics is None:
            return

        steps, vals = zip(*metrics)
        steps = np.array(steps, dtype=np.int32)
        vals = np.array(vals, dtype=np.float32)

        viz = self.viz
        experiment_name = self.experiment_name
        env = self.env

        win = viz.updateTrace(X=steps, Y=vals,
                              name=experiment_name, win=key, env=env,
                              append=True)

        if not self.viz_success(win):
            viz.line(X=steps, Y=vals,
                     win=key, env=env,
                     opts={"legend": [experiment_name], "title": key})


class FileLogger(object):
    # A logging alternative that doesn't leave logs open between writes,
    # so as to allow AFS synchronization.

    # Level constants
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3

    def __init__(self, log_path=None, json_log_path=None, min_print_level=0, min_file_level=0):
        # log_path: The full path for the log file to write. The file will be appended
        #   to if it exists.
        # min_print_level: Only messages with level above this level will be printed to stderr.
        # min_file_level: Only messages with level above this level will be
        #   written to disk.
        self.log_path = log_path
        self.json_log_path = json_log_path
        self.min_print_level = min_print_level
        self.min_file_level = min_file_level

    def Log(self, message, level=INFO):
        if level >= self.min_print_level:
            # Write to STDERR
            sys.stderr.write("[%i] %s\n" % (level, message))
        if self.log_path and level >= self.min_file_level:
            # Write to the log file then close it
            with open(self.log_path, 'a') as f:
                datetime_string = datetime.datetime.now().strftime(
                    "%y-%m-%d %H:%M:%S")
                f.write("%s [%i] %s\n" % (datetime_string, level, message))

    def LogJSON(self, message_obj, level=INFO):
        if self.json_log_path and level >= self.min_file_level:
            with open(self.json_log_path, 'w') as f:
                print >>f, json.dumps(message_obj)
        else:
            sys.stderr.write('WARNING: No JSON log filename.')


def read_log_load(filename, last=True):
    ret = None
    cur = None
    reading = False
    begin = "Flag Values"
    end = "}"

    with open(filename) as f:
        for line in f:
            if begin in line and not reading:
                cur = ""
                reading = True
                continue

            if reading:
                cur += line.strip()

                if end in line:
                    ret = json.loads(cur)
                    reading = False

                    if not last:
                        return ret

    return ret


def clean_desc(desc):
    words = word_tokenize(desc.lower())  # lowercase and tokenize
    words = list(set(words))  # remove duplicates
    words = [w for w in words if w not in stopwords.words(
        'english')]  # remove stopwords
    words = [w for w in words if w not in string.punctuation]
    return words


def read_data(input_descr):
    descr = {}
    word_dict = {}
    dict_size = 0
    num_descr = 0
    label_id_to_idx = {}
    idx_to_label = {}
    with open(input_descr, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            parts = line.split(",")
            label_id, label = parts[:2]
            desc = line[len(label_id) + len(label) + 2:]
            desc = clean_desc(desc)
            # print label, sorted(desc)
            for w in desc:
                if w not in word_dict:
                    dict_size += 1
                    word_dict[w] = {"id": dict_size}
            descr[num_descr] = {"name": label, "desc": desc}
            num_descr += 1
            label_id_to_idx[int(label_id)] = i
            idx_to_label[i] = label
    _desc = set([w for ii in descr.keys() for w in descr[ii]['desc']])
    # print sorted(_desc)
    return descr, word_dict, dict_size, label_id_to_idx, idx_to_label


def load_hdf5(hdf5_file, batch_size, random_seed, shuffle, truncate_final_batch=False, map_labels=int):
    """
    Reads images into random batches
    """
    # Read data
    f = h5py.File(os.path.expanduser(hdf5_file), "r")
    target = f["Target"]
    dataset_size = target.shape[0]
    f.close()
    order = list(range(dataset_size))

    # Shuffle
    if shuffle:
        random.seed(11 + random_seed)
        random.shuffle(order)

    # Generate batches
    num_batches = dataset_size // batch_size

    if truncate_final_batch:
        if dataset_size - (num_batches * batch_size) > 0:
            num_batches = num_batches + 1

    for i in range(num_batches):

        batch_indices = sorted(order[i * batch_size:(i + 1) * batch_size])

        f = h5py.File(os.path.expanduser(hdf5_file), "r")

        batch = dict()

        # TODO: We probably need to map the label_ids some way.
        batch['target'] = torch.LongTensor(
            list(map(map_labels, f["Target"][batch_indices])))
        # Location format broken in hdf5 in python 3
        #batch['example_ids'] = f["Location"][batch_indices]

        batch['layer4_2'] = torch.from_numpy(
            f["layer4_2"][batch_indices]).float().squeeze()
        batch['avgpool_512'] = torch.from_numpy(
            f["avgpool_512"][batch_indices]).float().squeeze()
        batch['fc'] = torch.from_numpy(
            f["fc"][batch_indices]).float().squeeze()

        f.close()

        yield batch


# Function returning word embeddings from GloVe
def embed(word_dict, emb):
    glove = {}
    print("Vocab Size: {}".format(len(word_dict.keys())))
    with open(emb, "r") as f:
        for line in f:
            word = line.strip()
            word = word.split(" ")
            if word[0] in word_dict:
                embed = torch.Tensor([float(s) for s in word[1:]])
                glove[word[0]] = embed
    print("Found {} in glove.".format(len(glove.keys())))
    for k in word_dict:
        embed = glove.get(k, None)
        word_dict[k]["emb"] = embed
    return word_dict


# Function computing CBOW for each description
def cbow(descr, word_dict):
    # TODO: Faster summing please!
    emb_size = len(list(word_dict.values())[0]["emb"])
    for mammal in descr:
        num_w = 0
        desc_len = len(descr[mammal]["desc"])
        desc_set = torch.FloatTensor(desc_len, emb_size).fill_(0)
        for i_w, w in enumerate(descr[mammal]["desc"]):
            if word_dict[w]["emb"] is not None:
                desc_set[i_w] = word_dict[w]["emb"]
                num_w += 1
        desc_cbow = desc_set.clone().sum(0).squeeze()
        if num_w > 0:
            desc_cbow = desc_cbow / num_w
        descr[mammal]["cbow"] = desc_cbow
        descr[mammal]["set"] = desc_set
    return descr


# Function computing CBOW for each set of descriptions
def cbow_general(texts, word2id, id2word):
    '''Takes a batch of n texts per example. Each example is a list of ints corresponding to a textual description of the image
    Returns: two tensors.
                1. cbow vector for each example
                    size: batch_size x desc_per_elem x embedding_dim
                2. individual word vectors for each example
                    size: batch_size x desc_per_elem x max_description_length x embedding_dim
                    sentences shorter than max_length are 0 padded at the end'''
    emb_size = len(list(word2id.values())[1]["emb"])
    desc_per_eg = len(texts[0])
    max_len = max([max([len(e) for e in t]) for t in texts])  # max length of sentence
    debuglogger.debug(f'batch_size: {len(texts)}, desc per eg: {desc_per_eg}, emb size: {emb_size}, max len: {max_len}')
    desc_set = torch.FloatTensor(len(texts), desc_per_eg, max_len, emb_size).fill_(0)
    desc_set_lens = torch.FloatTensor(len(texts), desc_per_eg).fill_(0)
    desc_cbow = torch.FloatTensor(len(texts), desc_per_eg, emb_size).fill_(0)
    for i_t, t in enumerate(texts):
        for i_e, e in enumerate(t):
            num_w = 0
            for i_w, w in enumerate(e):
                if word2id[id2word[w]]["emb"] is not None:
                    desc_set[i_t, i_e, i_w, :] = word2id[id2word[w]]["emb"]
                    num_w += 1
            desc_cbow[i_t, i_e] = desc_set[i_t, i_e, :, :].sum(0).squeeze()
            if num_w > 0:
                desc_cbow[i_t, i_e] /= num_w
            desc_set_lens[i_t, i_e] = num_w
    # debuglogger.debug(f'cbow: {desc_cbow}')
    # debuglogger.debug(f'lens: {desc_set_lens}')
    return desc_cbow, desc_set, desc_set_lens


"""
Initialization Schemes
Source: https://github.com/alykhantejani/nninit/blob/master/nninit.py
"""


def _calculate_fan_in_and_fan_out(tensor):
    if tensor.ndimension() < 2:
        raise ValueError(
            "fan in and fan out can not be computed for tensor of size ", tensor.size())

    if tensor.ndimension() == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = np.prod(tensor.numpy().shape[2:])
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_normal(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method described in "Understanding the difficulty of training
       deep feedforward neural networks" - Glorot, X. and Bengio, Y., using a normal distribution.
       The resulting tensor will have values sampled from normal distribution with mean=0 and
       std = gain * sqrt(2/(fan_in + fan_out))
    Args:
        tensor: a n-dimension torch.Tensor
        gain: an optional scaling factor to be applied
    Examples:
        >>> w = torch.Tensor(3, 5)
        >>> nninit.xavier_normal(w, gain=np.sqrt(2.0))
    """
    if isinstance(tensor, Variable):
        xavier_normal(tensor.data, gain=gain)
        return tensor
    else:
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        return tensor.normal_(0, std)


def build_mask(region_str, size):
    # Read input string
    regions = region_str.split(',')
    regions = [r.split(':') for r in regions]
    regions = [[int(r[0])] if len(r) == 1 else
               list(range(int(r[0]), int(r[1]))) for r in regions]  # python style indexing

    # Flattens the list of lists
    index = torch.LongTensor(list(itertools.chain(*regions)))

    # Generate mask
    mask = torch.FloatTensor(size, 1).fill_(0)
    mask[index] = 1

    return mask
