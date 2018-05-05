import os
import sys
import json
import time
import numpy as np
import random
import h5py
import functools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as _Variable
import torch.optim as optim
from torch.nn.parameter import Parameter

import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms

from sklearn.metrics import confusion_matrix

from misc import recursively_set_device, torch_save, torch_load
from misc import VisdomLogger as Logger
from misc import FileLogger
from misc import read_log_load
from misc import load_hdf5
from misc import read_data
from misc import embed
from misc import cbow
from misc import xavier_normal
from misc import build_mask

from sparks import sparks

from binary_vectors import extract_binary

import gflags

FLAGS = gflags.FLAGS

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
debuglogger = logging.getLogger('main_logger')
debuglogger.setLevel(30)

def Variable(*args, **kwargs):
    var = _Variable(*args, **kwargs)
    if FLAGS.cuda:
        var = var.cuda()
    return var


class Sender(nn.Module):
    """Agent 1 Network: Sender
    """

    def __init__(self, feature_type, feat_dim, h_dim, w_dim, bin_dim_out, use_binary,
                 use_attn, attn_dim, attn_extra_context, attn_context_dim):
        super(Sender, self).__init__()
        self.feature_type = feature_type
        self.feat_dim = feat_dim
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.bin_dim_out = bin_dim_out
        self.use_binary = use_binary
        self.use_attn = use_attn
        self.attn_dim = attn_dim
        self.attn_extra_context = attn_extra_context
        self.attn_context_dim = attn_context_dim

        self.image_layer = nn.Linear(self.feat_dim, self.h_dim)
        self.code_layer = nn.Linear(self.w_dim, self.h_dim)
        self.code_bias = Parameter(torch.Tensor(self.bin_dim_out))
        # Layer for binary vector
        if FLAGS.sender_mix == "mou":
            self.binary_layer = nn.Linear(self.h_dim * 4, self.bin_dim_out)
            if FLAGS.ignore_code:
                self.code_bias_mou = Parameter(torch.Tensor(self.bin_dim_out))
        else:
            self.binary_layer = nn.Linear(self.h_dim, self.bin_dim_out)

        # self.binary_layer.bias.data.fill_(-2.)

        if self.use_attn:
            self.attn_W_x = nn.Linear(self.feat_dim, self.attn_dim)
            self.attn_W_w = nn.Linear(self.w_dim, self.attn_dim)
            self.attn_U = nn.Linear(self.attn_dim, 1)

            if FLAGS.attn_extra_context:
                self.attn_W_g = nn.Linear(self.attn_context_dim, self.attn_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.set_(xavier_normal(m.weight.data))
                if m.bias is not None:
                    m.bias.data.zero_()
        if hasattr(self, 'code_bias'):
            self.code_bias.data.normal_()

    def reset_state(self):
        """Initialize state for Sender.

        The Sender is stateless in its decisions, but some computation
        can be reused at each time step.

        """
        # Cached computation.
        self.h_x_attn_flat = None
        self.h_g_flat = None
        self.fn_x = None

        # Used for debugging.
        self.attn_scores = []

    def attention_func(self, w, x, g):
        batch_size, n_feats, channels = x.size()

        h_w_attn = self.attn_W_w(w)
        h_w_attn_broadcast = h_w_attn.contiguous().unsqueeze(
            1).expand(batch_size, n_feats, self.attn_dim)
        h_w_attn_flat = h_w_attn_broadcast.contiguous().view(
            batch_size * n_feats, self.attn_dim)

        if not self.h_x_attn_flat:
            x_flat = x.contiguous().view(batch_size * n_feats, channels)
            self.h_x_attn_flat = self.attn_W_x(x_flat)

        if self.attn_extra_context:
            if not self.h_g_flat:
                h_g = self.attn_W_g(g)
                hg_broadcast = h_g.contiguous().unsqueeze(
                    1).expand(batch_size, n_feats, self.attn_dim)
                self.h_g_flat = hg_broadcast.contiguous().view(
                    batch_size * n_feats, self.attn_dim)

        if self.attn_extra_context:
            attn_U_inp = nn.Tanh()(h_w_attn_flat + self.h_x_attn_flat + self.h_g_flat)
        else:
            attn_U_inp = nn.Tanh()(h_w_attn_flat + self.h_x_attn_flat)

        attn_scores_flat = self.attn_U(attn_U_inp)

        return attn_scores_flat

    def forward(self, x, w, g, t):
        """Respond to communication query.

        Communication Response:
            z_hat = U_z(U_x x + U_w w)
            z = bernoulli(sig(z_hat)) or round(sig(z_hat))

        Image Attention (https://arxiv.org/pdf/1502.03044.pdf):
            \beta_i = U tanh(W_r z_r + W_x x_i [+ W_g g])
            \alpha = 1 / |x|        if t == 0
            \alpha = softmax(\beta) otherwise
            x = \sum_i \alpha x_i

        Args:
            x: Image features.
            w: Communication query from Receiver.
            g: (attention) Image features used as query in attention.
            t: (attention) Timestep. Used to change attention equation in first iteration.
        Output:
            features: A binary (or continuous) message in response to Receiver's query.
            feature_probs: If the message is binary, then these are probability of ``1`` for each bit
                in the message.
        """

        if self.use_attn:
            batch_size, channels, height, width = x.size()
            n_feats = height * width
            x = x.view(batch_size, channels, n_feats)
            x = x.transpose(1, 2)

            attn_scores_flat = self.attention_func(w, x, g)

            # attention scores
            if t == 0:
                attn_scores = Variable(torch.FloatTensor(
                    batch_size, n_feats).fill_(1), volatile=not self.training)
                attn_scores = attn_scores / n_feats
            else:
                attn_scores = F.softmax(
                    attn_scores_flat.view(batch_size, n_feats))

            # x = \sum_i a_i x_i
            x_attn = torch.bmm(attn_scores.unsqueeze(1), x).squeeze()

            # Cache values for inspection
            self.attn_scores.append(attn_scores)

            _x = x_attn
        else:
            _x = x

        debuglogger.info("=========================== SENDER FORWARD START ==================")
        debuglogger.info('Sender input feat sizes...')
        debuglogger.info(f'x: {x.size()}')
        debuglogger.info(f'w: {w.size()}')
        debuglogger.info(f'g: {g}')
        debuglogger.info(f't: {t}')
        self.h_x = h_x = self.image_layer(_x)
        debuglogger.info(f'h_x: {self.h_x.size()}')
        if t == 0:
            batch_size = x.size(0)
            # Same first code for all batch items.
            first_code = F.sigmoid(self.code_bias.view(1, -1))
            h_w = self.code_layer(first_code).expand(batch_size, self.h_dim)
        elif t > 0 and FLAGS.ignore_code and FLAGS.sender_mix == "mou":
            batch_size = x.size(0)
            # Same code for all batch items.
            code_mou = F.sigmoid(self.code_bias_mou.view(1, -1))
            h_w = self.code_layer(code_mou).expand(batch_size, self.h_dim)
        else:
            h_w = self.code_layer(w)
        debuglogger.info(f'h_w: {h_w.size()}')
        if FLAGS.ignore_code:
            if FLAGS.sender_mix == "sum" or FLAGS.sender_mix == "prod":
                features = self.binary_layer(F.tanh(h_x))
            elif FLAGS.sender_mix == "mou":
                features = self.binary_layer(
                    F.tanh(torch.cat([h_x, h_w, h_x - h_w, h_x * h_w], 1)))
        else:
            if FLAGS.sender_mix == "sum":
                features = self.binary_layer(F.tanh(h_x + h_w))
            elif FLAGS.sender_mix == "prod":
                features = self.binary_layer(F.tanh(h_x * h_w))
            elif FLAGS.sender_mix == "mou":
                features = self.binary_layer(
                    F.tanh(torch.cat([h_x, h_w, h_x - h_w, h_x * h_w], 1)))
        debuglogger.info(f'features: {features.size()}')
        if self.use_binary:
            probs = F.sigmoid(features)
            if self.training:
                probs_ = probs.data.cpu().numpy()
                binary_features = Variable(torch.from_numpy(
                    (np.random.rand(*probs_.shape) < probs_).astype('float32')))
            else:
                binary_features = torch.round(probs).detach()
            if probs.is_cuda:
                binary_features = binary_features.cuda()

            if FLAGS.flipout_sen is not None and (self.training or FLAGS.flipout_dev):
                binary_features = flipout(binary_features, FLAGS.flipout_sen)

            debuglogger.info(f'binary features: {binary_features.size()}')
            debuglogger.info(f'probs: {probs.size()}')
            debuglogger.info("=========================== SENDER FORWARD END ==================")
            return binary_features, probs
        else:
            return features, None


class Receiver(nn.Module):
    """Agent 2 Network: Receiver
    """

    def __init__(self, z_dim, desc_dim, hid_dim, out_dim, w_dim, s_dim, use_binary):
        super(Receiver, self).__init__()
        self.z_dim = z_dim
        self.desc_dim = desc_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.w_dim = w_dim
        self.s_dim = s_dim
        self.use_binary = use_binary

        # RNN network
        self.rnn = nn.GRUCell(self.z_dim, self.hid_dim)
        # Network for Receiver communications
        self.w_h = nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        self.w_d = nn.Linear(self.desc_dim, self.hid_dim, bias=False)
        self.w = nn.Linear(self.hid_dim, self.w_dim)
        # Network for Receiver predicitons
        self.y1 = nn.Linear(self.hid_dim + self.desc_dim, self.hid_dim)
        self.y2 = nn.Linear(self.hid_dim, self.out_dim)
        # Network for Receiver decisions
        self.s = nn.Linear(self.hid_dim, self.s_dim)

        if FLAGS.desc_attn:
            self.attn_dim = FLAGS.desc_attn_dim
            self.d_d = nn.Linear(self.desc_dim, self.attn_dim)
            self.d_h = nn.Linear(self.hid_dim, self.attn_dim)
            self.d_attn = nn.Linear(self.attn_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.set_(xavier_normal(m.weight.data))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GRUCell):
                for mm in m.parameters():
                    if mm.data.ndimension() == 2:
                        mm.data.set_(xavier_normal(mm.data))
                    elif mm.data.ndimension() == 1:  # Bias
                        mm.data.zero_()
        if hasattr(self, 'code_bias'):
            self.code_bias.data.uniform_(-1, 1)

    def reset_state(self):
        """Initialize state for Receiver.

        The Receiver is stateful, keeping tracking of previous messages it
        has sent and received.

        """
        self.h_z = None
        self.s_prob_prod = None

    def initial_state(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hid_dim))

    def forward(self, z, desc, desc_set=None, desc_set_lens=None):
        """Send communication query.

        Update State:
            h_z = rnn(z, h_z)

        Predictions:
            y_i = f_y(h_z, desc_i)

        Communication Query:
            desc = \sum_i y_i desc_i
            w_hat = tanh(W_h h_z + W_d desc)
            w = bernoulli(sig(w_hat)) or round(sig(w_hat))

        STOP Bit:
            s_hat = W_s h_z
            s = bernoulli(sig(s_hat)) or round(sig(s_hat))

        Args:
            z: Communication response from Receiver.
            desc: List of description vectors used in communication and predictions.
        Output:
            s, s_probs: A STOP bit and its associated probability, indicating whether the Receiver has decided to stop
                or continue its conversation with the Sender.
            w, w_probs: A binary (or continuous) message, which is a query incorporating the descriptions. If the
                message is binary, then the probability of each bit in the message being ``1`` is included.
            y: A prediction for each class described in the descriptions.
        """
        debuglogger.info("=================== RECEIVER FORWARD START ======================")
        debuglogger.info("receiver input sizes...")
        debuglogger.info(f'z: {z.size()}')
        debuglogger.info(f'desc: {desc.size()}')
        debuglogger.info(f'desc set: {desc_set.size()}')
        debuglogger.info(f'desc set size: len: {len(desc_set_lens)} vals:  {desc_set_lens}')

        # BatchSize x BinaryDim
        batch_size, binary_dim = z.size()

        # Initialize hidden state if necessary
        if self.h_z is None:
            self.h_z = self.initial_state(batch_size)

        # Run z through RNN
        self.h_z = self.rnn(z, self.h_z)
        
        debuglogger.info(f'h_z: {self.h_z.size()}')
        # Build input for prediction using descriptions
        # size of inp_with_desc: B*D x (WV+h)
        if FLAGS.desc_attn:
            nwords, desc_dim = desc_set.size()
            desc_set = Variable(desc_set)  # NW x WV
            # Broadcast and Flatten
            desc_set_broadcast = desc_set.unsqueeze(0).expand(
                batch_size, nwords, self.desc_dim)  # B x NW x WV

            # Broadcast and Flatten
            dd = self.d_d(desc_set)  # NW x A
            dd_broadcast = dd.unsqueeze(0).expand(
                batch_size, nwords, self.attn_dim)  # B x NW x A
            dd_flat = dd_broadcast.contiguous().view(
                batch_size * nwords, self.attn_dim)  # B*NW x A

            # Broadcast and Flatten
            dh = self.d_h(self.h_z)  # B x A
            dh_broadcast = dh.unsqueeze(1).expand(
                batch_size, nwords, self.attn_dim)  # B x NW x A
            dh_flat = dh_broadcast.contiguous().view(
                batch_size * nwords, self.attn_dim)  # B*NW x A

            # Get and Apply Attention Scores
            d_attn = self.d_attn(F.tanh(dd_flat + dh_flat)
                                 ).view(batch_size, nwords)  # B x NW

            # Partitioned Scores
            cumlen = 0
            d_attn_scores = []
            for idesc, ndesc in enumerate(desc_set_lens):
                start = cumlen
                end = cumlen + ndesc
                cumlen = end

                scores = F.softmax(d_attn[:, start:end])  # B x NW_i
                d_attn_scores.append(scores)
            self.d_attn_scores = d_attn_scores = torch.cat(
                d_attn_scores, 1)  # B x NW

            # Attend
            d_attn_broadcast = d_attn_scores.unsqueeze(
                2).expand_as(desc_set_broadcast)  # B x NW x WV
            desc_set_weighted = desc_set_broadcast * d_attn_broadcast  # B x NW x WV

            # Partitioned Weighted Sum
            cumlen = 0
            weighted_desc = []
            for idesc, ndesc in enumerate(desc_set_lens):
                start = cumlen
                end = cumlen + ndesc
                cumlen = end

                cbow = desc_set_weighted[:, start:end, :].sum(1)  # B x 1 x WV
                weighted_desc.append(cbow)
            weighted_desc = torch.cat(weighted_desc, 1)  # B x D x WV

            # Build Input
            nclasses = weighted_desc.size(1)
            weighted_desc = weighted_desc.view(
                batch_size * nclasses, self.desc_dim)  # B*D x WV

            h_z_broadcast = self.h_z.unsqueeze(1).expand(
                batch_size, nclasses, self.hid_dim)  # B x D x h
            h_z_flat = h_z_broadcast.contiguous().view(
                batch_size * nclasses, self.hid_dim)  # B*D x h

            inp_with_desc = torch.cat(
                [weighted_desc, h_z_flat], 1)  # B*D x (WV+h)
        else:
            inp_with_desc = build_inp(self.h_z, desc)  # B*D x (WV+h)
        debuglogger.info(f'inp_with_desc: {inp_with_desc.size()}')

        s_score = self.s(self.h_z)
        s_prob = F.sigmoid(s_score)
        debuglogger.debug(f'stop score: {s_score}, s_prob: {s_prob}')
        if self.training:
            # Sample decisions
            prob_ = s_prob.data.cpu().numpy()
            debuglogger.debug(f'stop probs:\n {prob_}')
            s_binary = Variable(torch.from_numpy(
                (np.random.rand(*prob_.shape) < prob_).astype('float32')))
        else:
            # Infer decisions
            if self.s_prob_prod is None or not FLAGS.s_prob_prod:
                self.s_prob_prod = s_prob
            else:
                self.s_prob_prod = self.s_prob_prod * s_prob
            s_binary = torch.round(self.s_prob_prod).detach()
        debuglogger.debug(f'stop decisions: {s_binary}')
        if s_prob.is_cuda:
            s_binary = s_binary.cuda()

        # Obtain predictions
        y = self.y1(inp_with_desc).clamp(min=0)
        y = self.y2(y).view(batch_size, -1)
        debuglogger.info(f'class predictions: {y.size()}')
        # Obtain communications
        # size of y = batch_size x # descriptions
        # size of desc = # descriptions x self.desc_dim
        # size of wd_inp = batch_size x self.desc_dim
        n_desc = y.size(1)
        # Reweight descriptions based on current model confidence
        y_scores = F.softmax(y, dim=1).detach()
        debuglogger.info(f'class scores: {y_scores.size()}')
        y_broadcast = y_scores.unsqueeze(2).expand(
            batch_size, n_desc, self.desc_dim)
        debuglogger.info(f'class scores broadcast: {y_broadcast.size()}')
        if FLAGS.desc_attn:
            wd_inp = weighted_desc.view(batch_size, nclasses, self.desc_dim)
        else:
            wd_inp = desc.unsqueeze(0).expand(
                batch_size, n_desc, self.desc_dim)
        debuglogger.info(f'wd_inp: {wd_inp.size()}')
        wd_inp = (y_broadcast * wd_inp).sum(1).squeeze(1)
        debuglogger.info(f'wd_inp: {wd_inp.size()}')

        # Hidden state for Receiver message
        self.h_w = F.tanh(self.w_h(self.h_z) + self.w_d(wd_inp))
        debuglogger.info(f'h_w: {self.h_w.size()}')
        
        w_scores = self.w(self.h_w)
        debuglogger.info(f'w_scores: {w_scores.size()}')
        if self.use_binary:
            w_probs = F.sigmoid(w_scores)
            debuglogger.info(f'w_probs: {w_probs.size()}')
            if self.training:
                probs_ = w_probs.data.cpu().numpy()
                debuglogger.info(f'probs: {probs_.shape}')
                w_binary = Variable(torch.from_numpy(
                    (np.random.rand(*probs_.shape) < probs_).astype('float32')))
                debuglogger.info(f'w_binary: {w_binary.size()}')
            else:
                w_binary = torch.round(w_probs).detach()
            if w_probs.is_cuda:
                w_binary = w_binary.cuda()
            w_feats = w_binary

            if FLAGS.flipout_rec is not None and (self.training or FLAGS.flipout_dev):
                w_feats = flipout(w_feats, FLAGS.flipout_rec)

            if FLAGS.ignore_receiver:
                w_feats = Variable(torch.zeros(w_feats.size()),
                                   volatile=not self.training)
        else:
            w_feats = w_scores
            w_probs = None
        
        debuglogger.info("=================== RECEIVER FORWARD END ======================")

        return (s_binary, s_prob), (w_feats, w_probs), y


class Baseline(nn.Module):
    """Baseline
    """

    def __init__(self, hid_dim, x_dim, binary_dim, inp_dim):
        super(Baseline, self).__init__()
        self.x_dim = x_dim
        self.binary_dim = binary_dim
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim

        # Additional layers on top of feature extractor
        self.linear1 = nn.Linear(
            x_dim + self.binary_dim + self.inp_dim, self.hid_dim)
        self.linear2 = nn.Linear(self.hid_dim, 1)

    def forward(self, x, binary, inp):
        """Estimate agent's loss based on the agent's input.

        Args:
            x: Image features.
            binary: Communication message.
            inp: Hidden state (used when agent is the Receiver).
        Output:
            score: An estimate of the agent's loss.
        """
        features = []
        if x is not None:
            features.append(x)
        if binary is not None:
            features.append(binary)
        if inp is not None:
            features.append(inp)
        features = torch.cat(features, 1)
        hidden = self.linear1(features).clamp(min=0)
        pred_score = self.linear2(hidden)
        return pred_score


def build_inp(binary_features, descs):
    """Function preparing input for Receiver network

    Args:
        binary_features: List of communication vectors, length ``B``.
        descs: List of description vectors, length ``D``.
    Output:
        b_cat_d: The cartesian product of binary features and descriptions, length ``B`` x ``D``.
    """
    debuglogger.info("=========== BUILD INP START ================")
    if descs is not None:
        batch_size = binary_features.size(0)
        num_desc, desc_dim = descs.size()

        # Expand binary features.
        debuglogger.info(f'binary features {len(binary_features)}')
        debuglogger.info(f'descs: {len(descs)}')
        binary_index = torch.from_numpy(
            np.arange(batch_size).repeat(num_desc).astype(np.int32)).long()
        debuglogger.info(f'binary index {binary_index.size()}')
        if binary_features.is_cuda:
            binary_index = binary_index.cuda()
        binary_copied = torch.index_select(
            binary_features, 0, Variable(binary_index))

        debuglogger.info(f'binary copied {binary_copied.size()}')
        # Expand descriptions.
        desc_index = torch.from_numpy(np.concatenate(
            [np.arange(num_desc)] * batch_size).astype(np.int32)).long()
        if descs.is_cuda:
            desc_index = desc_index.cuda()
        debuglogger.info(f'desc index: {desc_index.size()}')
        desc_copied = torch.index_select(descs, 0, Variable(desc_index))
        debuglogger.info(f'desc copied: {desc_copied.size()}')
        # Concat binary vector with description vectors
        inp = torch.cat([binary_copied, desc_copied], 1)
        debuglogger.info(f'inp: {inp.size()}')
        debuglogger.info("=========== BUILD INP END ================")
        return inp
    else:
        debuglogger.info("=========== BUILD INP END ================")
        return binary_features


def flipout(binary, p):
    """
    Args:
        binary: Tensor of binary values.
        p: Probability of flipping a binary value.
    Output:
        outp: Tensor with same size as `binary` where bits have been
            flipped with probability `p`.
    """
    mask = torch.FloatTensor(binary.size()).fill_(p).numpy()
    mask = Variable(torch.from_numpy(
        (np.random.rand(*mask.shape) < mask).astype('float32')))
    outp = (binary - mask).abs()

    return outp


def loglikelihood(log_prob, target):
    """
    Args: log softmax scores (N, C) where N is the batch size
          and C is the number of classes
    Output: log likelihood (N)
    """
    return log_prob.gather(1, target)


def eval_dev(dev_file, batch_size, epoch, shuffle, cuda, top_k,
             sender, receiver, desc_dict, map_labels, file_name,
             callback=None):
    """
    Function computing development accuracy
    """

    desc = desc_dict["desc"]
    desc_set = desc_dict.get("desc_set", None)
    desc_set_lens = desc_dict.get("desc_set_lens", None)

    extra = dict()

    # Keep track of conversation lengths
    conversation_lengths = []

    # Keep track of message diversity
    hamming_sen = []
    hamming_rec = []

    # Keep track of labels
    true_labels = []
    pred_labels = []

    # Keep track of number of correct observations
    total = 0
    correct = 0

    # Load development images
    dev_loader = load_hdf5(dev_file, batch_size, epoch, shuffle,
                           truncate_final_batch=True, map_labels=map_labels)

    for batch in dev_loader:
        # Extract images and targets

        target = batch["target"]
        data = batch[FLAGS.img_feat]
        _batch_size = target.size(0)

        true_labels.append(target.cpu().numpy().reshape(-1))

        # GPU support
        if cuda:
            data = data.cuda()
            target = target.cuda()
            desc = desc.cuda()

        exchange_args = dict()
        exchange_args["data"] = data
        if FLAGS.attn_extra_context:
            exchange_args["data_context"] = batch[FLAGS.data_context]
        exchange_args["target"] = target
        exchange_args["desc"] = desc
        exchange_args["desc_set"] = desc_set
        exchange_args["desc_set_lens"] = desc_set_lens
        exchange_args["train"] = False
        exchange_args["break_early"] = not FLAGS.fixed_exchange
        exchange_args["corrupt"] = FLAGS.bit_flip
        exchange_args["corrupt_region"] = FLAGS.corrupt_region

        s, sen_w, rec_w, y, bs, br = exchange(
            sender, receiver, None, None, exchange_args)

        s_masks, s_feats, s_probs = s
        sen_feats, sen_probs = sen_w
        rec_feats, rec_probs = rec_w

        # Mask if dynamic exchange length
        if FLAGS.fixed_exchange:
            y_masks = None
        else:
            y_masks = [torch.min(1 - m1, m2)
                       for m1, m2 in zip(s_masks[1:], s_masks[:-1])]

        outp, _ = get_rec_outp(y, y_masks)

        # Obtain top k predictions
        dist = F.log_softmax(outp, dim=1)
        top_k_ind = torch.from_numpy(
            dist.data.cpu().numpy().argsort()[:, -top_k:]).long()
        target = target.view(-1, 1).expand(_batch_size, top_k)

        # Store top 1 prediction for confusion matrix
        _, argmax = dist.data.max(1)
        pred_labels.append(argmax.cpu().numpy())

        # Update accuracy counts
        total += float(batch_size)
        correct += (top_k_ind == target.cpu()).sum()

        # Keep track of conversation lengths
        conversation_lengths += torch.cat(s_feats,
                                          1).data.float().sum(1).view(-1).tolist()

        # Keep track of message diversity
        mean_hamming_rec = 0
        mean_hamming_sen = 0
        prev_rec = torch.FloatTensor(_batch_size, FLAGS.rec_w_dim).fill_(0)
        prev_sen = torch.FloatTensor(_batch_size, FLAGS.rec_w_dim).fill_(0)

        for msg in sen_feats:
            mean_hamming_sen += (msg.data.cpu() - prev_sen).abs().sum(1).mean()
            prev_sen = msg.data.cpu()
        mean_hamming_sen = mean_hamming_sen / float(len(sen_feats))

        for msg in rec_feats:
            mean_hamming_rec += (msg.data.cpu() - prev_rec).abs().sum(1).mean()
            prev_rec = msg.data.cpu()
        mean_hamming_rec = mean_hamming_rec / float(len(rec_feats))

        hamming_sen.append(mean_hamming_sen)
        hamming_rec.append(mean_hamming_rec)

        if callback is not None:
            callback_dict = dict(
                s_masks=s_masks,
                s_feats=s_feats,
                s_probs=s_probs,
                sen_feats=sen_feats,
                sen_probs=sen_probs,
                rec_feats=rec_feats,
                rec_probs=rec_probs,
                y=y)
            callback(sender, receiver, batch, callback_dict)

    # Print confusion matrix
    true_labels = np.concatenate(true_labels).reshape(-1)
    pred_labels = np.concatenate(pred_labels).reshape(-1)

    np.savetxt(FLAGS.conf_mat, confusion_matrix(
        true_labels, pred_labels), delimiter=',', fmt='%d')

    # Compute statistics
    conversation_lengths = np.array(conversation_lengths)
    hamming_sen = np.array(hamming_sen)
    hamming_rec = np.array(hamming_rec)
    extra['conversation_lengths_mean'] = conversation_lengths.mean()
    extra['conversation_lengths_std'] = conversation_lengths.std()
    extra['hamming_sen_mean'] = hamming_sen.mean()
    extra['hamming_rec_mean'] = hamming_rec.mean()

    # Return accuracy
    return correct / total, extra


def exchange(sender, receiver, baseline_sen, baseline_rec, exchange_args):
    """Run a batched conversation between Sender and Receiver.

    The Sender has only the image, and the Receiver has descriptions of each of the image's
    possible classes and a history of each message it has sent and received.

    The Receiver begins the conversation by sending a query of Os. The Sender inspects this query
    and the image, then formulates a response. The Receiver inspects the response and its set of
    descriptions, then formulates a new query. The conversation continues this way until it has
    reached some predetermined length, or the Receiver has decided it has processed a sufficient
    amount of information at which point it ignores all future conversation. When each Receiver
    in the batch has received sufficient information, then the batched conversation may terminate
    early.

    Exchange Args:
        data: Image features.
        data_context: Optional additional image features that can be used as query in visual attention.
        target: Class labels.
        desc: List of description vectors.
        train: Boolean value indicating training mode (True) or evaluation mode (False).
        break_early: Boolean value. If True, then terminate batched conversation if all Receivers are satisfied.
    Args:
        sender: Agent 1. The Sender.
        receiver: Agent 2. The Receiver.
        baseline_sen: Baseline network for Sender.
        baseline_rec: Baseline network for Receiver.
        exchange_args: Other useful arguments.
    Output:
        s: All STOP bits. (Masks, Values, Probabilities)
        sen_w: All sender messages. (Values, Probabilities)
        rec_w: All receiver messages. (Values, Probabilities)
        y: All predictions that were made.
        bs: Estimated loss of sender.
        br: Estimated loss of receiver.
    """

    data = exchange_args["data"]
    data_context = exchange_args.get("data_context", None)
    target = exchange_args["target"]
    desc = exchange_args["desc"]
    desc_set = exchange_args.get("desc_set", None)
    desc_set_lens = exchange_args.get("desc_set_lens", None)
    train = exchange_args["train"]
    break_early = exchange_args.get("break_early", False)
    corrupt = exchange_args.get("corrupt", False)
    corrupt_region = exchange_args.get("corrupt_region", None)

    batch_size = data.size(0)

    # Pad with one column of ones.
    stop_mask = [Variable(torch.ones(batch_size, 1).byte())]
    stop_feat = []
    stop_prob = []
    sen_feats = []
    sen_probs = []
    rec_feats = []
    rec_probs = []
    y = []
    bs = []
    br = []

    w_binary = Variable(torch.FloatTensor(batch_size, sender.w_dim).fill_(
        FLAGS.first_rec), volatile=not train)

    if train:
        sender.train()
        receiver.train()
        baseline_sen.train()
        baseline_rec.train()
    else:
        sender.eval()
        receiver.eval()

    sender.reset_state()  # only for debugging/performance
    receiver.reset_state()

    for i_exchange in range(FLAGS.max_exchange):
        debuglogger.info(f' ================== EXCHANGE {i_exchange} ====================')
        z_r = w_binary  # rename variable to z_r which makes more sense

        # Run data through Sender
        if data_context is not None:
            z_binary, z_probs = sender(Variable(data, volatile=not train), Variable(z_r.data, volatile=not train),
                                       Variable(data_context, volatile=not train), i_exchange)
        else:
            z_binary, z_probs = sender(Variable(data, volatile=not train), Variable(z_r.data, volatile=not train),
                                       None, i_exchange)

        # Optionally corrupt Sender's message
        if corrupt:
            # Obtain mask
            mask = Variable(build_mask(corrupt_region, sender.w_dim))
            mask_broadcast = mask.view(1, sender.w_dim).expand_as(z_binary)
            # Subtract the mask to change values, but need to get absolute value
            # to set -1 values to 1 to essentially "flip" all the bits.
            z_binary = (z_binary - mask_broadcast).abs()

        # Generate input for Receiver
        z_s = z_binary  # rename variable to z_s which makes more sense

        # Run batch through Receiver
        (s_binary, s_prob), (w_binary, w_probs), outp = receiver(
            Variable(z_s.data, volatile=not train), Variable(
                desc.data, volatile=not train),
            desc_set, desc_set_lens)

        if train:
            sen_h_x = sender.h_x

            # Score from Baseline (Sender)
            baseline_sen_scores = baseline_sen(
                Variable(sen_h_x.data), Variable(z_r.data), None)

            rec_h_z = receiver.h_z if receiver.h_z is not None  else receiver.initial_state(
                batch_size)

            # Score from Baseline (Receiver)
            baseline_rec_scores = baseline_rec(
                None, Variable(z_s.data), Variable(rec_h_z.data))

        outp = outp.view(batch_size, -1)

        # Obtain predictions
        dist = F.log_softmax(outp, dim=1)
        maxdist, argmax = dist.data.max(1)

        # Save for later
        stop_mask.append(torch.min(stop_mask[-1], s_binary.byte()))
        stop_feat.append(s_binary)
        stop_prob.append(s_prob)
        sen_feats.append(z_binary)
        sen_probs.append(z_probs)
        rec_feats.append(w_binary)
        rec_probs.append(w_probs)
        y.append(outp)

        if train:
            br.append(baseline_rec_scores)
            bs.append(baseline_sen_scores)

        # Terminate exchange if everyone is done conversing
        if break_early and stop_mask[-1].float().sum().data[0] == 0:
            break

    # The final mask must always be zero.
    stop_mask[-1].data.fill_(0)

    s = (stop_mask, stop_feat, stop_prob)
    sen_w = (sen_feats, sen_probs)
    rec_w = (rec_feats, rec_probs)

    return s, sen_w, rec_w, y, bs, br


def get_rec_outp(y, masks):
    def negent(yy):
        probs = F.softmax(yy, dim=1)
        return (torch.log(probs + 1e-8) * probs).sum(1).mean()

    # TODO: This is wrong for the dynamic exchange, and we might want a "per example"
    # entropy for either exchange (this version is mean across batch).
    negentropy = list(map(negent, y))
    debuglogger.info(f'negentropy type: {type(negentropy)}')

    if masks is not None:

        batch_size = y[0].size(0)
        exchange_steps = len(masks)

        inp = torch.cat([yy.view(batch_size, 1, -1) for yy in y], 1)
        mask = torch.cat(masks, 1).view(
            batch_size, exchange_steps, 1).expand_as(inp)
        outp = torch.masked_select(inp, mask.detach()).view(batch_size, -1)

        if FLAGS.debug:
            # Each mask index should have exactly 1 true value.
            assert all([mm.data[0] == 1 for mm in torch.cat(masks, 1).sum(1)])

        return outp, negentropy
    else:
        return y[-1], negentropy


def calculate_loss_binary(binary_features, binary_probs, logs, baseline_scores, entropy_penalty):
    log_p_z = Variable(binary_features.data) * torch.log(binary_probs + 1e-8) + \
        (1 - Variable(binary_features.data)) * \
        torch.log(1 - binary_probs + 1e-8)
    log_p_z = log_p_z.sum(1)
    weight = Variable(logs.data) - \
        Variable(baseline_scores.clone().detach().data)
    if logs.size(0) > 1:
        weight = weight / np.maximum(1., torch.std(weight.data))
    loss = torch.mean(-1 * weight * log_p_z)

    # Must do both sides of negent, otherwise is skewed towards 0.
    initial_negent = (torch.log(binary_probs + 1e-8)
                      * binary_probs).sum(1).mean()
    inverse_negent = (torch.log((1. - binary_probs) + 1e-8)
                      * (1. - binary_probs)).sum(1).mean()
    negentropy = initial_negent + inverse_negent

    if entropy_penalty is not None:
        loss = (loss + entropy_penalty * negentropy)
    return loss, negentropy


def multistep_loss_binary(binary_features, binary_probs, logs, baseline_scores, masks, entropy_penalty):
    if masks is not None:
        def mapped_fn(feat, prob, scores, mask, mask_sums):
            if mask_sums == 0:
                return Variable(torch.zeros(1))

            feat_size = feat.size()
            prob_size = prob.size()
            logs_size = logs.size()
            scores_size = scores.size()

            feat = feat[mask.expand_as(feat)].view(-1, feat_size[1])
            prob = prob[mask.expand_as(prob)].view(-1, prob_size[1])
            _logs = logs[mask.expand_as(logs)].view(-1, logs_size[1])
            scores = scores[mask.expand_as(scores)].view(-1, scores_size[1])
            return calculate_loss_binary(feat, prob, _logs, scores, entropy_penalty)

        _mask_sums = [m.float().sum().data[0] for m in masks]

        if FLAGS.debug:
            assert len(masks) > 0
            assert len(masks) == len(binary_features)
            assert len(masks) == len(binary_probs)
            assert len(masks) == len(baseline_scores)
            assert sum(_mask_sums) > 0

        outp = map(mapped_fn, binary_features, binary_probs,
                   baseline_scores, masks, _mask_sums)
        losses = [o[0] for o in outp]
        entropies = [o[1] for o in outp]
        _losses = [l * ms for l, ms in zip(losses, _mask_sums)]
        loss = sum(_losses) / sum(_mask_sums)
    else:
        outp = map(lambda feat, prob, scores: calculate_loss_binary(feat, prob, logs, scores, entropy_penalty),
                   binary_features, binary_probs, baseline_scores)
        losses = [o[0] for o in outp]
        entropies = [o[1] for o in outp]
        loss = sum(losses) / len(binary_features)
    return loss, entropies


def calculate_loss_bas(baseline_scores, logs):
    loss_bas = nn.MSELoss()(baseline_scores, Variable(logs.data))
    return loss_bas


def multistep_loss_bas(baseline_scores, logs, masks):
    if masks is not None:
        losses = map(lambda scores, mask: calculate_loss_bas(
            scores[mask].view(-1, 1), logs[mask].view(-1, 1)),
            baseline_scores, masks)
        _mask_sums = [m.sum().float() for m in masks]
        _losses = [l * ms for l, ms in zip(losses, _mask_sums)]
        loss = sum(_losses) / sum(_mask_sums)
    else:
        losses = map(lambda scores: calculate_loss_bas(scores, logs),
                     baseline_scores)
        loss = sum(losses) / len(baseline_scores)
    return loss


def bin_to_alpha(binary):
    ret = []
    interval = 5
    offset = 65
    for i in range(0, len(binary), interval):
        val = int(binary[i:i + interval], 2)
        ret.append(unichr(offset + val))
    return " ".join(ret)


def run():
    flogger = FileLogger(FLAGS.log_file)
    logger = Logger(
        env=FLAGS.env, experiment_name=FLAGS.experiment_name, enabled=FLAGS.visdom)

    flogger.Log("Flag Values:\n" +
                json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True))

    if not os.path.exists(FLAGS.json_file):
        with open(FLAGS.json_file, "w") as f:
            f.write(json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True))

    # Sender model
    sender = Sender(feature_type=FLAGS.img_feat,
                    feat_dim=FLAGS.img_feat_dim,
                    h_dim=FLAGS.img_h_dim,
                    w_dim=FLAGS.rec_w_dim,
                    bin_dim_out=FLAGS.sender_out_dim,
                    use_binary=FLAGS.use_binary,
                    use_attn=FLAGS.visual_attn,
                    attn_dim=FLAGS.attn_dim,
                    attn_extra_context=FLAGS.attn_extra_context,
                    attn_context_dim=FLAGS.attn_context_dim)

    flogger.Log("Architecture: {}".format(sender))
    total_params = sum([functools.reduce(lambda x, y: x * y, p.size(), 1.0)
                        for p in sender.parameters()])
    flogger.Log("Total Parameters: {}".format(total_params))

    # Baseline model
    baseline_sen = Baseline(hid_dim=FLAGS.baseline_hid_dim,
                            x_dim=FLAGS.img_h_dim,
                            binary_dim=FLAGS.rec_w_dim,
                            inp_dim=0)

    flogger.Log("Architecture: {}".format(baseline_sen))
    total_params = sum([functools.reduce(lambda x, y: x * y, p.size(), 1.0)
                        for p in baseline_sen.parameters()])
    flogger.Log("Total Parameters: {}".format(total_params))

    # Receiver network
    receiver = Receiver(hid_dim=FLAGS.rec_hidden,
                        out_dim=FLAGS.rec_out_dim,
                        z_dim=FLAGS.sender_out_dim,
                        desc_dim=FLAGS.wv_dim,
                        w_dim=FLAGS.rec_w_dim,
                        s_dim=FLAGS.rec_s_dim,
                        use_binary=FLAGS.use_binary)

    flogger.Log("Architecture: {}".format(receiver))
    total_params = sum([functools.reduce(lambda x, y: x * y, p.size(), 1.0)
                        for p in receiver.parameters()])
    flogger.Log("Total Parameters: {}".format(total_params))

    # Baseline model
    baseline_rec = Baseline(hid_dim=FLAGS.baseline_hid_dim,
                            x_dim=0,
                            binary_dim=FLAGS.rec_w_dim,
                            inp_dim=FLAGS.rec_hidden)

    flogger.Log("Architecture: {}".format(baseline_rec))
    total_params = sum([functools.reduce(lambda x, y: x * y, p.size(), 1.0)
                        for p in baseline_rec.parameters()])
    flogger.Log("Total Parameters: {}".format(total_params))

    # Get description vectors
    if FLAGS.wv_type == "fake":
        num_desc = 10
        desc = Variable(torch.randn(num_desc, FLAGS.wv_dim).float())
    elif FLAGS.wv_type == "glove.6B":
        # Train
        descr_train, word_dict_train, dict_size_train, label_id_to_idx_train, idx_to_label_train = read_data(
            FLAGS.descr_train)

        def map_labels_train(x): return label_id_to_idx_train.get(x)
        word_dict_train = embed(word_dict_train, FLAGS.glove_path)
        descr_train = cbow(descr_train, word_dict_train)
        desc_train = torch.cat([descr_train[i]["cbow"].view(1, -1)
                                for i in descr_train.keys()], 0)
        desc_train = Variable(desc_train)
        desc_train_set = torch.cat(
            [descr_train[i]["set"].view(-1, FLAGS.wv_dim) for i in descr_train.keys()], 0)
        desc_train_set_lens = [len(descr_train[i]["desc"])
                               for i in descr_train.keys()]

        # Dev
        descr_dev, word_dict_dev, dict_size_dev, label_id_to_idx_dev, idx_to_label_dev = read_data(
            FLAGS.descr_dev)

        def map_labels_dev(x): return label_id_to_idx_dev.get(x)
        word_dict_dev = embed(word_dict_dev, FLAGS.glove_path)
        descr_dev = cbow(descr_dev, word_dict_dev)
        desc_dev = torch.cat([descr_dev[i]["cbow"].view(1, -1)
                              for i in descr_dev.keys()], 0)
        desc_dev = Variable(desc_dev)
        desc_dev_set = torch.cat(
            [descr_dev[i]["set"].view(-1, FLAGS.wv_dim) for i in descr_dev.keys()], 0)
        desc_dev_set_lens = [len(descr_dev[i]["desc"])
                             for i in descr_dev.keys()]

        desc_dev_dict = dict(
            desc=desc_dev,
            desc_set=desc_dev_set,
            desc_set_lens=desc_dev_set_lens)
    elif FLAGS.wv_type == "none":
        desc = None
    else:
        raise NotImplementedError

    # Optimizer
    if FLAGS.optim_type == "SGD":
        optimizer_rec = optim.SGD(
            receiver.parameters(), lr=FLAGS.learning_rate)
        optimizer_sen = optim.SGD(sender.parameters(), lr=FLAGS.learning_rate)
        optimizer_bas_rec = optim.SGD(
            baseline_rec.parameters(), lr=FLAGS.learning_rate)
        optimizer_bas_sen = optim.SGD(
            baseline_sen.parameters(), lr=FLAGS.learning_rate)
    elif FLAGS.optim_type == "Adam":
        optimizer_rec = optim.Adam(
            receiver.parameters(), lr=FLAGS.learning_rate)
        optimizer_sen = optim.Adam(sender.parameters(), lr=FLAGS.learning_rate)
        optimizer_bas_rec = optim.Adam(
            baseline_rec.parameters(), lr=FLAGS.learning_rate)
        optimizer_bas_sen = optim.Adam(
            baseline_sen.parameters(), lr=FLAGS.learning_rate)
    elif FLAGS.optim_type == "RMSprop":
        optimizer_rec = optim.RMSprop(
            receiver.parameters(), lr=FLAGS.learning_rate)
        optimizer_sen = optim.RMSprop(
            sender.parameters(), lr=FLAGS.learning_rate)
        optimizer_bas_rec = optim.RMSprop(
            baseline_rec.parameters(), lr=FLAGS.learning_rate)
        optimizer_bas_sen = optim.RMSprop(
            baseline_sen.parameters(), lr=FLAGS.learning_rate)
    else:
        raise NotImplementedError

    optimizers_dict = dict(optimizer_rec=optimizer_rec, optimizer_sen=optimizer_sen,
                           optimizer_bas_rec=optimizer_bas_rec, optimizer_bas_sen=optimizer_bas_sen)
    models_dict = dict(receiver=receiver, sender=sender,
                       baseline_rec=baseline_rec, baseline_sen=baseline_sen)

    # Training metrics
    epoch = 0
    step = 0
    best_dev_acc = 0

    # Optionally load previously saved model
    if os.path.exists(FLAGS.checkpoint):
        flogger.Log("Loading from: " + FLAGS.checkpoint)
        data = torch_load(FLAGS.checkpoint, models_dict, optimizers_dict)
        flogger.Log("Loaded at step: {} and best dev acc: {}".format(
            data['step'], data['best_dev_acc']))
        step = data['step']
        best_dev_acc = data['best_dev_acc']

    # GPU support
    if FLAGS.cuda:
        for m in models_dict.values():
            m.cuda()
        for o in optimizers_dict.values():
            recursively_set_device(o.state_dict(), gpu=0)

    # Alternatives to training.
    if FLAGS.eval_only:
        if not os.path.exists(FLAGS.checkpoint):
            raise Exception("Must provide valid checkpoint.")
        dev_acc, extra = eval_dev(FLAGS.dev_file, FLAGS.batch_size_dev, epoch,
                                  FLAGS.shuffle_dev, FLAGS.cuda, FLAGS.top_k_dev,
                                  sender, receiver, desc_dev_dict, map_labels_dev, FLAGS.experiment_name)
        flogger.Log("Dev Accuracy: " + str(dev_acc))
        with open(FLAGS.eval_csv_file, 'w') as f:
            f.write(
                "checkpoint,eval_file,topk,step,best_dev_acc,eval_acc,convlen_mean,convlen_std\n")
            f.write("{},{},{},{},{},{},{},{}\n".format(
                FLAGS.checkpoint, FLAGS.dev_file, FLAGS.top_k_dev,
                step, best_dev_acc, dev_acc,
                extra['conversation_lengths_mean'], extra['conversation_lengths_std']))
        sys.exit()
    elif FLAGS.binary_only:
        if not os.path.exists(FLAGS.checkpoint):
            raise Exception("Must provide valid checkpoint.")
        extract_binary(FLAGS, load_hdf5, exchange, FLAGS.dev_file, FLAGS.batch_size_dev, epoch,
                       FLAGS.shuffle_dev, FLAGS.cuda, FLAGS.top_k_dev,
                       sender, receiver, desc_dev_dict, map_labels_dev, FLAGS.experiment_name)
        sys.exit()

    # Training loop
    while epoch < FLAGS.max_epoch:

        flogger.Log("Starting epoch: {}".format(epoch))

        # Read images randomly into batches - image_dim = [3, 227, 227]
        if FLAGS.images == "cifar":
            dataset = dset.CIFAR10(root="./", download=True, train=False,
                                   transform=transforms.Compose([
                                       transforms.Scale(227),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])
                                   )
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=FLAGS.batch_size,
                                                     shuffle=True)
        elif FLAGS.images == "mammal":
            dataloader = load_hdf5(FLAGS.train_file, FLAGS.batch_size,
                                   epoch, FLAGS.shuffle_train, map_labels=map_labels_train)
        else:
            raise NotImplementedError

        # Keep track of metrics
        batch_accuracy = []
        dev_accuracy = []

        # Iterate through batches
        for i_batch, batch in enumerate(dataloader):
            target = batch["target"]
            data = batch[FLAGS.img_feat]

            # GPU support
            if FLAGS.cuda:
                data = data.cuda()
                target = target.cuda()
                desc_train = desc_train.cuda()
                desc_train_set = desc_train_set.cuda()

            exchange_args = dict()
            exchange_args["data"] = data
            if FLAGS.attn_extra_context:
                exchange_args["data_context"] = batch[FLAGS.data_context]
            exchange_args["target"] = target
            exchange_args["desc"] = desc_train
            exchange_args["desc_set"] = desc_train_set
            exchange_args["desc_set_lens"] = desc_train_set_lens
            exchange_args["train"] = True
            exchange_args["break_early"] = not FLAGS.fixed_exchange

            s, sen_w, rec_w, y, bs, br = exchange(
                sender, receiver, baseline_sen, baseline_rec, exchange_args)

            s_masks, s_feats, s_probs = s
            sen_feats, sen_probs = sen_w
            rec_feats, rec_probs = rec_w

            # Mask loss if dynamic exchange length
            if FLAGS.fixed_exchange:
                binary_s_masks = None
                binary_rec_masks = None
                binary_sen_masks = None
                bas_rec_masks = None
                bas_sen_masks = None
                y_masks = None
            else:
                binary_s_masks = s_masks[:-1]
                binary_rec_masks = s_masks[1:-1]
                binary_sen_masks = s_masks[:-1]
                bas_rec_masks = s_masks[:-1]
                bas_sen_masks = s_masks[:-1]
                y_masks = [torch.min(1 - m1, m2)
                           for m1, m2 in zip(s_masks[1:], s_masks[:-1])]

            outp, ent_y_rec = get_rec_outp(y, y_masks)

            # Obtain predictions
            debuglogger.info(f'outp: {outp.size()}')
            dist = F.log_softmax(outp, dim=1)
            debuglogger.info(f'dist: {dist.size()}')
            maxdist, argmax = dist.data.max(1)
            debuglogger.debug(f'maxdist: {maxdist}, argmax: {argmax}')

            # Receiver classification loss
            nll_loss = nn.NLLLoss()(dist, Variable(target))

            # Individual log-likelihoods across the batch
            logs = loglikelihood(Variable(dist.data),
                                 Variable(target.view(-1, 1)))

            if FLAGS.use_binary:
                if not FLAGS.fixed_exchange:
                    loss_binary_s, ent_binary_s = multistep_loss_binary(
                        s_feats, s_probs, logs, br, binary_s_masks, FLAGS.entropy_s)

                # The receiver might have no z-loss if we stop after first
                # message from sender.
                if len(rec_feats[:-1]) > 0:
                    loss_binary_rec, ent_binary_rec = multistep_loss_binary(
                        rec_feats[:-1], rec_probs[:-1], logs, br[:-1], binary_rec_masks, FLAGS.entropy_rec)
                else:
                    loss_binary_rec, ent_binary_rec = Variable(
                        torch.zeros(1)), []

                loss_binary_sen, ent_binary_sen = multistep_loss_binary(
                    sen_feats, sen_probs, logs, bs, binary_sen_masks, FLAGS.entropy_sen)
                loss_bas_rec = multistep_loss_bas(br, logs, bas_rec_masks)
                loss_bas_sen = multistep_loss_bas(bs, logs, bas_sen_masks)

            loss_rec = nll_loss
            if FLAGS.use_binary:
                loss_rec = loss_rec + loss_binary_rec
                if not FLAGS.fixed_exchange:
                    loss_rec = loss_rec + loss_binary_s
                loss_sen = loss_binary_sen
            else:
                loss_sen = Variable(torch.zeros(1))
                loss_bas_rec = Variable(torch.zeros(1))
                loss_bas_sen = Variable(torch.zeros(1))

            # Update receiver
            optimizer_rec.zero_grad()
            loss_rec.backward()
            nn.utils.clip_grad_norm(receiver.parameters(), max_norm=1.)
            optimizer_rec.step()

            if FLAGS.use_binary:
                # Update sender
                optimizer_sen.zero_grad()
                loss_sen.backward()
                nn.utils.clip_grad_norm(sender.parameters(), max_norm=1.)
                optimizer_sen.step()

                # Update baseline
                optimizer_bas_rec.zero_grad()
                loss_bas_rec.backward()
                nn.utils.clip_grad_norm(baseline_rec.parameters(), max_norm=1.)
                optimizer_bas_rec.step()

                # Update baseline
                optimizer_bas_sen.zero_grad()
                loss_bas_sen.backward()
                nn.utils.clip_grad_norm(baseline_sen.parameters(), max_norm=1.)
                optimizer_bas_sen.step()

            # Obtain top-k accuracy
            top_k_ind = torch.from_numpy(dist.data.cpu().numpy().argsort()[
                                         :, -FLAGS.top_k_train:]).long()
            target_exp = target.view(-1,
                                     1).expand(FLAGS.batch_size, FLAGS.top_k_train)
            accuracy = (top_k_ind == target_exp.cpu()).sum() / \
                float(FLAGS.batch_size)
            batch_accuracy.append(accuracy)

            # Print logs regularly
            if step % FLAGS.log_interval == 0:
                # Average batch accuracy
                avg_batch_acc = np.array(
                    batch_accuracy[-FLAGS.log_interval:]).mean()

                # Log accuracy
                log_acc = "Epoch: {} Step: {} Batch: {} Training Accuracy: {}"\
                          .format(epoch, step, i_batch, avg_batch_acc)
                flogger.Log(log_acc)

                # Sender
                log_loss_sen = "Epoch: {} Step: {} Batch: {} Loss Sender: {}".format(
                    epoch, step, i_batch, loss_sen.data[0])
                flogger.Log(log_loss_sen)

                # Receiver
                log_loss_rec_y = "Epoch: {} Step: {} Batch: {} Loss Receiver (Y): {}".format(
                    epoch, step, i_batch, nll_loss.data[0])
                flogger.Log(log_loss_rec_y)
                if FLAGS.use_binary:
                    log_loss_rec_z = "Epoch: {} Step: {} Batch: {} Loss Receiver (Z): {}".format(
                        epoch, step, i_batch, loss_binary_rec.data[0])
                    flogger.Log(log_loss_rec_z)
                    if not FLAGS.fixed_exchange:
                        log_loss_rec_s = "Epoch: {} Step: {} Batch: {} Loss Receiver (S): {}".format(
                            epoch, step, i_batch, loss_binary_s.data[0])
                        flogger.Log(log_loss_rec_s)

                # Baslines
                if FLAGS.use_binary:
                    log_loss_bas_s = "Epoch: {} Step: {} Batch: {} Loss Baseline (S): {}".format(
                        epoch, step, i_batch, loss_bas_sen.data[0])
                    flogger.Log(log_loss_bas_s)
                    log_loss_bas_r = "Epoch: {} Step: {} Batch: {} Loss Baseline (R): {}".format(
                        epoch, step, i_batch, loss_bas_rec.data[0])
                    flogger.Log(log_loss_bas_r)

                # Log predictions
                log_pred = "Predictions: {}".format(
                    torch.cat([target, argmax], 0).view(-1, FLAGS.batch_size))
                flogger.Log(log_pred)

                # Log Entropy
                if FLAGS.use_binary:
                    if len(ent_binary_sen) > 0:
                        log_ent_sen_bin = "Entropy Sender Binary"
                        for i, ent in enumerate(ent_binary_sen):
                            log_ent_sen_bin += "\n{}. {}".format(
                                i, -ent.data[0])
                        log_ent_sen_bin += "\n"
                        flogger.Log(log_ent_sen_bin)

                    if len(ent_binary_rec) > 0:
                        log_ent_rec_bin = "Entropy Receiver Binary"
                        for i, ent in enumerate(ent_binary_rec):
                            log_ent_rec_bin += "\n{}. {}".format(
                                i, -ent.data[0])
                        log_ent_rec_bin += "\n"
                        flogger.Log(log_ent_rec_bin)

                if len(ent_y_rec) > 0:
                    log_ent_rec_y = "Entropy Receiver Predictions"
                    for i, ent in enumerate(ent_y_rec):
                        log_ent_rec_y += "\n{}. {}".format(i, -ent.data[0])
                    log_ent_rec_y += "\n"
                    flogger.Log(log_ent_rec_y)

                # Optionally print sampled and inferred binary vectors from
                # most recent exchange.
                if FLAGS.exchange_samples > 0:

                    current_exchange = len(sen_feats)

                    log_train = "Train:"
                    for i_sample in range(FLAGS.exchange_samples):
                        prev_sen = torch.FloatTensor(FLAGS.rec_w_dim).fill_(0)
                        prev_rec = torch.FloatTensor(FLAGS.rec_w_dim).fill_(0)
                        for i_exchange in range(current_exchange):
                            sen_probs_i = sen_probs[i_exchange][i_sample].data.tolist(
                            )
                            sen_spark = sparks(
                                [1] + sen_probs_i)[1:].encode('utf-8')
                            rec_probs_i = rec_probs[i_exchange][i_sample].data.tolist(
                            )
                            rec_spark = sparks(
                                [1] + rec_probs_i)[1:].encode('utf-8')
                            s_probs_i = s_probs[i_exchange][i_sample].data.tolist(
                            )
                            s_spark = sparks(
                                [1] + s_probs_i)[1:].encode('utf-8')

                            sen_binary = sen_feats[i_exchange][i_sample].data.cpu(
                            )
                            sen_hamming = (prev_sen - sen_binary).abs().sum()
                            prev_sen = sen_binary
                            rec_binary = rec_feats[i_exchange][i_sample].data.cpu(
                            )
                            rec_hamming = (prev_rec - rec_binary).abs().sum()
                            prev_rec = rec_binary

                            sen_msg = "".join(
                                map(str, map(int, sen_binary.tolist())))
                            rec_msg = "".join(
                                map(str, map(int, rec_binary.tolist())))
                            if FLAGS.use_alpha:
                                sen_msg = bin_to_alpha(sen_msg)
                                rec_msg = bin_to_alpha(rec_msg)
                            if i_exchange == 0:
                                log_train += "\n{:>3}".format(i_sample)
                            else:
                                log_train += "\n   "
                            log_train += "        {}".format(sen_spark)
                            log_train += "           {}    {}".format(
                                s_spark, rec_spark)
                            log_train += "\n    {:>3} S: {} {:4}".format(
                                i_exchange, sen_msg, sen_hamming)
                            log_train += "    s={} R: {} {:4}".format(
                                s_masks[1:][i_exchange][i_sample].data[0], rec_msg, rec_hamming)
                    log_train += "\n"
                    flogger.Log(log_train)

                    exchange_args["train"] = False
                    s, sen_w, rec_w, y, bs, br = exchange(
                        sender, receiver, baseline_sen, baseline_rec, exchange_args)
                    s_masks, s_feats, s_probs = s
                    sen_feats, sen_probs = sen_w
                    rec_feats, rec_probs = rec_w

                    current_exchange = len(sen_feats)

                    log_eval = "Eval:"
                    for i_sample in range(FLAGS.exchange_samples):
                        prev_sen = torch.FloatTensor(FLAGS.rec_w_dim).fill_(0)
                        prev_rec = torch.FloatTensor(FLAGS.rec_w_dim).fill_(0)
                        for i_exchange in range(current_exchange):
                            sen_probs_i = sen_probs[i_exchange][i_sample].data.tolist(
                            )
                            sen_spark = sparks(
                                [1] + sen_probs_i)[1:].encode('utf-8')
                            rec_probs_i = rec_probs[i_exchange][i_sample].data.tolist(
                            )
                            rec_spark = sparks(
                                [1] + rec_probs_i)[1:].encode('utf-8')
                            s_probs_i = s_probs[i_exchange][i_sample].data.tolist(
                            )
                            s_spark = sparks(
                                [1] + s_probs_i)[1:].encode('utf-8')

                            sen_binary = sen_feats[i_exchange][i_sample].data.cpu(
                            )
                            sen_hamming = (prev_sen - sen_binary).abs().sum()
                            prev_sen = sen_binary
                            rec_binary = rec_feats[i_exchange][i_sample].data.cpu(
                            )
                            rec_hamming = (prev_rec - rec_binary).abs().sum()
                            prev_rec = rec_binary

                            sen_msg = "".join(
                                map(str, map(int, sen_binary.tolist())))
                            rec_msg = "".join(
                                map(str, map(int, rec_binary.tolist())))
                            if FLAGS.use_alpha:
                                sen_msg = bin_to_alpha(sen_msg)
                                rec_msg = bin_to_alpha(rec_msg)
                            if i_exchange == 0:
                                log_eval += "\n{:>3}".format(i_sample)
                            else:
                                log_eval += "\n   "
                            log_eval += "        {}".format(sen_spark)
                            log_eval += "           {}    {}".format(
                                s_spark, rec_spark)
                            log_eval += "\n    {:>3} S: {} {:4}".format(
                                i_exchange, sen_msg, sen_hamming)
                            log_eval += "    s={} R: {} {:4}".format(
                                s_masks[1:][i_exchange][i_sample].data[0], rec_msg, rec_hamming)
                    log_eval += "\n"
                    flogger.Log(log_eval)

                # Sender
                logger.log(key="Loss Sender",
                           val=loss_sen.data[0], step=step)

                # Receiver
                logger.log(key="Loss Receiver (Y)",
                           val=nll_loss.data[0], step=step)
                if FLAGS.use_binary:
                    logger.log(key="Loss Receiver (Z)",
                               val=loss_binary_rec.data[0], step=step)
                    if not FLAGS.fixed_exchange:
                        logger.log(key="Loss Receiver (S)",
                                   val=loss_binary_s.data[0], step=step)

                # Baselines
                if FLAGS.use_binary:
                    logger.log(key="Loss Baseline (S)",
                               val=loss_bas_sen.data[0], step=step)
                    logger.log(key="Loss Baseline (R)",
                               val=loss_bas_rec.data[0], step=step)

                logger.log(key="Training Accuracy",
                           val=avg_batch_acc, step=step)

            # Report development accuracy
            if step % FLAGS.log_dev == 0:
                dev_acc, extra = eval_dev(FLAGS.dev_file, FLAGS.batch_size_dev, epoch,
                                          FLAGS.shuffle_dev, FLAGS.cuda, FLAGS.top_k_dev,
                                          sender, receiver, desc_dev_dict, map_labels_dev, FLAGS.experiment_name)
                dev_accuracy.append(dev_acc)
                logger.log(key="Development Accuracy",
                           val=dev_accuracy[-1], step=step)
                logger.log(key="Conversation Length (avg)",
                           val=extra['conversation_lengths_mean'], step=step)
                logger.log(key="Conversation Length (std)",
                           val=extra['conversation_lengths_std'], step=step)
                logger.log(key="Hamming Receiver (avg)",
                           val=extra['hamming_rec_mean'], step=step)
                logger.log(key="Hamming Sender (avg)",
                           val=extra['hamming_sen_mean'], step=step)

                flogger.Log("Epoch: {} Step: {} Batch: {} Development Accuracy: {}"
                            .format(epoch, step, i_batch, dev_accuracy[-1]))
                flogger.Log("Epoch: {} Step: {} Batch: {} Conversation Length (avg/std): {}/{}"
                            .format(epoch, step, i_batch,
                                    extra['conversation_lengths_mean'],
                                    extra['conversation_lengths_std']))
                flogger.Log("Epoch: {} Step: {} Batch: {} Mean Hamming Distance (R/S): {}/{}"
                            .format(epoch, step, i_batch, extra['hamming_rec_mean'], extra['hamming_sen_mean']))
                if step >= FLAGS.save_after and dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    flogger.Log(
                        "Checkpointing with best Development Accuracy: {}".format(best_dev_acc))
                    # Optionally store additional information
                    data = dict(step=step, best_dev_acc=best_dev_acc)
                    torch_save(FLAGS.checkpoint + "_best", data, models_dict,
                               optimizers_dict, gpu=0 if FLAGS.cuda else -1)

            # Save model periodically
            if step >= FLAGS.save_after and step % FLAGS.save_interval == 0:
                flogger.Log("Checkpointing.")
                # Optionally store additional information
                data = dict(step=step, best_dev_acc=best_dev_acc)
                torch_save(FLAGS.checkpoint, data, models_dict,
                           optimizers_dict, gpu=0 if FLAGS.cuda else -1)

            # Increment batch step
            step += 1
            #break

        # Increment epoch
        epoch += 1
        #break

    flogger.Log("Finished training.")


"""
Preset Model Configurations

1. Fixed - Fixed conversation length.
2. Adaptive - Adaptive conversation length using STOP bit.
3. FixedAttention - Fixed with Visual Attention.
4. AdaptiveAttention - Adaptive with Visual Attention.
"""


def Fixed():
    FLAGS.img_feat = "avgpool_512"
    FLAGS.img_feat_dim = 512
    FLAGS.fixed_exchange = True
    FLAGS.visual_attn = False


def Adaptive():
    FLAGS.img_feat = "avgpool_512"
    FLAGS.img_feat_dim = 512
    FLAGS.fixed_exchange = False
    FLAGS.visual_attn = False


def FixedAttention():
    FLAGS.img_feat = "layer4_2"
    FLAGS.img_feat_dim = 512
    FLAGS.fixed_exchange = True
    FLAGS.visual_attn = True
    FLAGS.attn_dim = 256
    FLAGS.attn_extra_context = True
    FLAGS.attn_context_dim = 1000


def AdaptiveAttention():
    FLAGS.img_feat = "layer4_2"
    FLAGS.img_feat_dim = 512
    FLAGS.fixed_exchange = False
    FLAGS.visual_attn = True
    FLAGS.attn_dim = 256
    FLAGS.attn_extra_context = True
    FLAGS.attn_context_dim = 1000


def flags():
    # Debug settings
    gflags.DEFINE_string("branch", None, "")
    gflags.DEFINE_string("sha", None, "")
    gflags.DEFINE_boolean("debug", False, "")

    # Convenience settings
    gflags.DEFINE_integer("save_after", 1000, "")
    gflags.DEFINE_integer("save_interval", 100, "")
    gflags.DEFINE_string("checkpoint", None, "")
    gflags.DEFINE_string("conf_mat", None, "")
    gflags.DEFINE_string("log_path", "./logs", "")
    gflags.DEFINE_string("log_file", None, "")
    gflags.DEFINE_string("eval_csv_file", None, "")
    gflags.DEFINE_string("json_file", None, "")
    gflags.DEFINE_string("log_load", None, "")
    gflags.DEFINE_boolean("eval_only", False, "")

    # Extract Settings
    gflags.DEFINE_boolean("binary_only", False, "")
    gflags.DEFINE_string("binary_output", None, "")

    # Performance settings
    gflags.DEFINE_boolean("cuda", False, "")

    # Display settings
    gflags.DEFINE_string("env", "main", "")
    gflags.DEFINE_boolean("visdom", False, "")
    gflags.DEFINE_boolean("use_alpha", False, "")
    gflags.DEFINE_string("experiment_name", None, "")
    gflags.DEFINE_integer("log_interval", 50, "")
    gflags.DEFINE_integer("log_dev", 1000, "")

    # Data settings
    gflags.DEFINE_enum("wv_type", "glove.6B", ["fake", "glove.6B", "none"], "")
    gflags.DEFINE_integer("wv_dim", 100, "")
    gflags.DEFINE_string("descr_train", "descriptions.csv", "")
    gflags.DEFINE_string("descr_dev", "descriptions.csv", "")
    gflags.DEFINE_string("train_file", "train.hdf5", "")
    gflags.DEFINE_string("dev_file", "dev.hdf5", "")
    gflags.DEFINE_enum("images", "mammal", ["cifar", "mammal"], "")
    gflags.DEFINE_string(
        "glove_path", "./glove.6B/glove.6B.100d.txt", "")
    gflags.DEFINE_boolean("shuffle_train", True, "")
    gflags.DEFINE_boolean("shuffle_dev", False, "")

    # Model settings
    gflags.DEFINE_enum("model_type", None, [
                       "Fixed", "Adaptive", "FixedAttention", "AdaptiveAttention"], "Preset model configurations.")
    gflags.DEFINE_enum("img_feat", "avgpool_512", [
                       "layer4_2", "avgpool_512", "fc"], "Specify which layer output to use as image")
    gflags.DEFINE_enum("data_context", "fc", [
                       "fc"], "Specify which layer output to use as context for attention")
    gflags.DEFINE_enum("sender_mix", "sum", ["sum", "prod", "mou"], "")
    gflags.DEFINE_integer("img_feat_dim", 4096, "")
    gflags.DEFINE_integer("img_h_dim", 100, "")
    gflags.DEFINE_integer("baseline_hid_dim", 500, "")
    gflags.DEFINE_integer("sender_out_dim", 50, "")
    gflags.DEFINE_integer("rec_hidden", 128, "")
    gflags.DEFINE_integer("rec_out_dim", 1, "")
    gflags.DEFINE_integer("rec_w_dim", 50, "")
    gflags.DEFINE_integer("rec_s_dim", 1, "")
    gflags.DEFINE_boolean("use_binary", True,
                          "Encoding whether Sender uses binary features")
    gflags.DEFINE_boolean("ignore_receiver", False,
                          "Sender ignores messages from Receiver")
    gflags.DEFINE_boolean("ignore_code", False,
                          "Sender ignores messages from Receiver")
    gflags.DEFINE_boolean(
        "block_y", True, "Halt gradient flow through description scores")
    gflags.DEFINE_float("first_rec", 0, "")
    gflags.DEFINE_float("flipout_rec", None, "Dropout for bit flipping")
    gflags.DEFINE_float("flipout_sen", None, "Dropout for bit flipping")
    gflags.DEFINE_boolean("flipout_dev", False, "Dropout for bit flipping")
    gflags.DEFINE_boolean("s_prob_prod", True,
                          "Simulate sampling during test time")
    gflags.DEFINE_boolean("visual_attn", False, "Sender attends over image")
    gflags.DEFINE_integer("attn_dim", 256, "")
    gflags.DEFINE_boolean("attn_extra_context", False, "")
    gflags.DEFINE_integer("attn_context_dim", 4096, "")
    gflags.DEFINE_boolean("desc_attn", False, "Receiver attends over text")
    gflags.DEFINE_integer("desc_attn_dim", 64, "Receiver attends over text")
    gflags.DEFINE_integer("top_k_dev", 6, "Top-k error in development")
    gflags.DEFINE_integer("top_k_train", 6, "Top-k error in training")

    # Optimization settings
    gflags.DEFINE_enum("optim_type", "RMSprop", ["Adam", "SGD", "RMSprop"], "")
    gflags.DEFINE_integer("batch_size", 32, "Minibatch size for train set.")
    gflags.DEFINE_integer("batch_size_dev", 50, "Minibatch size for dev set.")
    gflags.DEFINE_float("learning_rate", 1e-4, "Used in optimizer.")
    gflags.DEFINE_integer("max_epoch", 500, "")
    gflags.DEFINE_float("entropy_s", None, "")
    gflags.DEFINE_float("entropy_sen", None, "")
    gflags.DEFINE_float("entropy_rec", None, "")

    # Conversation settings
    gflags.DEFINE_integer("exchange_samples", 3, "")
    gflags.DEFINE_integer("max_exchange", 3, "")
    gflags.DEFINE_boolean("fixed_exchange", True, "")
    gflags.DEFINE_boolean(
        "bit_flip", False, "Whether sender's messages are corrupted.")
    gflags.DEFINE_string("corrupt_region", None,
                         "Comma-separated ranges of bit indexes (e.g. ``0:3,5'').")


def default_flags():
    if FLAGS.log_load:
        log_flags = json.loads(open(FLAGS.log_load).read())
        for k in log_flags.keys():
            if k in FLAGS.FlagValuesDict().keys():
                setattr(FLAGS, k, log_flags[k])
        FLAGS(sys.argv)  # Optionally override predefined flags.

    if FLAGS.model_type:
        eval(FLAGS.model_type)()
        FLAGS(sys.argv)  # Optionally override predefined flags.

    assert FLAGS.sender_out_dim == FLAGS.rec_w_dim, \
        "Both sender and receiver should communicate with same dim vectors for now."

    if not FLAGS.use_binary:
        FLAGS.exchange_samples = 0

    if not FLAGS.experiment_name:
        timestamp = str(int(time.time()))
        FLAGS.experiment_name = "{}-so_{}-wv_{}-bs_{}-{}".format(
            FLAGS.images,
            FLAGS.sender_out_dim,
            FLAGS.wv_dim,
            FLAGS.batch_size,
            timestamp,
        )

    if not FLAGS.conf_mat:
        FLAGS.conf_mat = os.path.join(
            FLAGS.log_path, FLAGS.experiment_name + ".conf_mat.txt")

    if not FLAGS.log_file:
        FLAGS.log_file = os.path.join(
            FLAGS.log_path, FLAGS.experiment_name + ".log")

    if not FLAGS.eval_csv_file:
        FLAGS.eval_csv_file = os.path.join(
            FLAGS.log_path, FLAGS.experiment_name + ".eval.csv")

    if not FLAGS.json_file:
        FLAGS.json_file = os.path.join(
            FLAGS.log_path, FLAGS.experiment_name + ".json")

    if not FLAGS.checkpoint:
        FLAGS.checkpoint = os.path.join(
            FLAGS.log_path, FLAGS.experiment_name + ".pt")

    if not FLAGS.binary_output:
        FLAGS.binary_output = os.path.join(
            FLAGS.log_path, FLAGS.experiment_name + ".bv.hdf5")

    if not FLAGS.branch:
        FLAGS.branch = os.popen(
            'git rev-parse --abbrev-ref HEAD').read().strip()

    if not FLAGS.sha:
        FLAGS.sha = os.popen('git rev-parse HEAD').read().strip()

    if not torch.cuda.is_available():
        FLAGS.cuda = False

    if FLAGS.debug:
        np.seterr(all='raise')

    # silly expanduser
    FLAGS.glove_path = os.path.expanduser(FLAGS.glove_path)


if __name__ == '__main__':
    flags()

    FLAGS(sys.argv)

    default_flags()
    
    print(sys.argv)
    
    run()
