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

FORMAT = '[%(asctime)s %(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT)
debuglogger = logging.getLogger('main_logger')
debuglogger.setLevel('INFO')


def Variable(*args, **kwargs):
    var = _Variable(*args, **kwargs)
    if FLAGS.cuda:
        var = var.cuda()
    return var


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


def corrupt_message(corrupt_region, agent, binary_message):
    # Obtain mask
    mask = Variable(build_mask(corrupt_region, agent.m_dim))
    mask_broadcast = mask.view(1, agent_1.m_dim).expand_as(binary_message)
    # Subtract the mask to change values, but need to get absolute value
    # to set -1 values to 1 to essentially "flip" all the bits.
    binary_message = (binary_message - mask_broadcast).abs()
    return binary_message


def exchange(agent1, agent2, exchange_args):
    """Run a batched conversation between two agents.

    # TODO - explain

    Exchange Args:
        data: Image features
            - dict containing the image features for agent 1 and agent 2, and the percentage of the
              image each agent received
              e.g.  { "im_feats_1": im_feats_1,
                      "im_feats_2": im_feats_2,
                      "p_1": p_1,
                      "p_2": p_2}
        target: Class labels.
        desc: List of description vectors.
        train: Boolean value indicating training mode (True) or evaluation mode (False).
        break_early: Boolean value. If True, then terminate batched conversation if both agents are satisfied
    Args:
        agent1: agent1
        agent2: agent2
        exchange_args: Other useful arguments.
    Output:
        s: All STOP bits. (Masks, Values, Probabilities)
        w_1: All agent_1 messages. (Values, Probabilities)
        w_2: All agent_2 messages. (Values, Probabilities)
        y_1: All predictions that were made by agent 1 (Before comms, after comms)
        y_2: All predictions that were made by agent 2 (Before comms, after comms)
        r_1: Estimated rewards of agent_1.
        r_2: Estimated rewards of agent_2.
    """

    data = exchange_args["data"]
    data_context = None
    target = exchange_args["target"]
    desc = exchange_args["desc"]
    train = exchange_args["train"]
    break_early = exchange_args.get("break_early", False)
    corrupt = exchange_args.get("corrupt", False)
    corrupt_region = exchange_args.get("corrupt_region", None)

    batch_size = data["im_feats_1"].size(0)

    # Pad with one column of ones.
    stop_mask_1 = [Variable(torch.ones(batch_size, 1).byte())]
    stop_feat_1 = []
    stop_prob_1 = []
    stop_mask_1 = [Variable(torch.ones(batch_size, 1).byte())]
    stop_feat_1 = []
    stop_prob_1 = []
    feats_1 = []
    probs_1 = []
    feats_2 = []
    probs_2 = []
    y_1_nc = None
    y_2_nc = None
    y_1 = []
    y_2 = []
    r_1 = []
    r_2 = []

    # First message
    m_binary = Variable(torch.FloatTensor(batch_size, agent_1.m_dim).fill_(
        FLAGS.first_msg), volatile=not train)

    if train:
        agent_1.train()
        agent_2.train()
    else:
        agent_1.eval()
        agent_2.eval()

    agent_1.reset_state()
    agent_2.reset_state()

    # The message is ignored initially
    use_message = False
    # Run data through both agents
    # No data context at the moment - # TODO
    if data_context is not None:
        pass
    else:
        s_1e, m_1e, y_1e, r_1e = agent_1(
            Variable(data['im_feats_1'], volatile=not train),
            Variable(m_binary.data, volatile=not train),
            0,
            Variable(desc.data, volatile=not train),
            use_message,
            batch_size,
            train)

        s_2e, m_2e, y_2e, r_2e = agent_2(
            Variable(data['im_feats_2'], volatile=not train),
            Variable(m_binary.data, volatile=not train),
            0,
            Variable(desc.data, volatile=not train),
            use_message,
            batch_size,
            train)

    # Add no message selections to results
    y_1_nc = y_1e
    y_2_nc = y_2e

    for i_exchange in range(FLAGS.max_exchange):
        debuglogger.info(
            f' ================== EXCHANGE {i_exchange} ====================')
        # The messages are now used
        use_message = True

        # Agent 1's message
        m_1e_binary, m_1e_probs = m_1e

        # Optionally corrupt agent 1's message
        if corrupt:
            m_1e_binary = corrupt_message(corrupt_region, agent1, m_1e_binary)

        # Run data through agent 2
        if data_context is not None:
            pass
        else:
            s_2e, m_2e, y_2e, r_2e = agent_2(
                Variable(data['im_feats_2'], volatile=not train),
                Variable(m_1e_binary.data, volatile=not train),
                i_exchange,
                Variable(desc.data, volatile=not train),
                use_message,
                batch_size,
                train)

        # Agent 2's message
        m_2e_binary, m_2e_probs = m_2e

        # Optionally corrupt agent 2's message
        if corrupt:
            m_2e_binary = corrupt_message(corrupt_region, agent2, m_2e_binary)

        # Run data through agent 1
        if data_context is not None:
            pass
        else:
            s_1e, m_1e, y_1e, r_1e = agent_1(
                Variable(data['im_feats_1'], volatile=not train),
                Variable(m_2e_binary.data, volatile=not train),
                i_exchange,
                Variable(desc.data, volatile=not train),
                use_message,
                batch_size,
                train)

        # TODO - check Not used
        # # Obtain predictions agent 1
        # dist_1 = F.log_softmax(y_1e, dim=1)
        # maxdist_1, argmax_1 = dist.data.max(1)
        #
        # # Obtain predictions agent 2
        # dist_2 = F.log_softmax(y_2e, dim=1)
        # maxdist_2, argmax_2 = dist.data.max(1)

        s_binary_1, s_prob_1 = s_1e
        s_binary_2, s_prob_2 = s_2e
        m_binary_1, m_probs_1 = m_1e
        m_binary_2, m_probs_2 = m_2e

        # Save for later
        # TODO check stop mask
        stop_mask_1.append(torch.min(stop_mask[-1], s_binary_1.byte()))
        stop_mask_2.append(torch.min(stop_mask[-1], s_binary_2.byte()))
        stop_feat_1.append(s_binary_1)
        stop_feat_2.append(s_binary_2)
        stop_prob_1.append(s_prob_1)
        stop_prob_2.append(s_prob_2)
        feats_1.append(s_binary_1)
        feats_2.append(s_binary_2)
        probs_1.append(s_prob_1)
        probs_2.append(s_prob_2)
        y_1.append(y_1e)
        y_2.append(y_2e)
        r_1.append(r_1e)
        r_2.append(r_2e)

        # Terminate exchange if everyone is done conversing
        if break_early and stop_mask_1[-1].float().sum().data[0] == 0 and stop_mask_2[-1].float().sum().data[0] == 0:
            break

    # The final mask must always be zero.
    stop_mask_1[-1].data.fill_(0)
    stop_mask_2[-1].data.fill_(0)

    s = [(stop_mask_1, stop_feat_1, stop_prob_1),
         (stop_mask_1, stop_feat_1, stop_prob_1)]
    message_1 = (feats_1, probs_1)
    message_2 = (feats_2, probs_2)
    y = (y_1, y_2)
    y_nc = (y_1_nc, y_2_nc)
    y_all = [y_nc, y]
    r = (r_1, r_2)

    return s, message_1, message_2, y_all, r


def get_outp(y, masks):
    def negent(yy):
        probs = F.softmax(yy, dim=1)
        return (torch.log(probs + 1e-8) * probs).sum(1).mean()

    # TODO: This is wrong for the dynamic exchange, and we might want a "per example"
    # entropy for either exchange (this version is mean across batch).
    negentropy = list(map(negent, y))

    # TODO check ok for new agents
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


def calculate_loss_binary(binary_features, binary_probs, rewards, baseline_rewards, entropy_penalty):
    log_p_z = Variable(binary_features.data) * torch.log(binary_probs + 1e-8) + \
        (1 - Variable(binary_features.data)) * \
        torch.log(1 - binary_probs + 1e-8)
    log_p_z = log_p_z.sum(1)
    weight = Variable(rewards.data) - Variable(baseline_rewards.clone().detach().data)
    if logs.size(0) > 1:  # TODO - check if this is needed
        weight = weight / np.maximum(1., torch.std(weight.data))
    loss = torch.mean(-1 * weight * log_p_z)

    # Must do both sides of negent, otherwise is skewed towards 0.
    initial_negent = (torch.log(binary_probs + 1e-8) * binary_probs).sum(1).mean()
    inverse_negent = (torch.log((1. - binary_probs) + 1e-8) * (1. - binary_probs)).sum(1).mean()
    negentropy = initial_negent + inverse_negent

    if entropy_penalty is not None:
        loss = (loss + entropy_penalty * negentropy)
    return loss, negentropy


def multistep_loss_binary(binary_features, binary_probs, logs, baseline_scores, masks, entropy_penalty):
    # TODO - check for new agents
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


def calculate_loss_bas(baseline_scores, rewards):
    loss_bas = nn.MSELoss()(baseline_scores, Variable(rewards.data))
    return loss_bas


def multistep_loss_bas(baseline_scores, logs, masks):
    # TODO - check for new agents
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


def calculate_accuracy(prediction_dist, target):
    assert FLAGS.batch_size == target.size(0)
    target_exp = target.view(-1, 1).expand(FLAGS.batch_size, FLAGS.top_k_train)
    top_k_ind = torch.from_numpy(prediction_dist.data.cpu().numpy().argsort()[:, -FLAGS.top_k_train:]).long()
    correct = (top_k_ind == target_exp.cpu()).sum(axis=1)
    accuracy = correct.sum() / float(FLAGS.batch_size)
    return accuracy, correct


def log_exchange(s, message_1, message_2, log_type="Train:"):
    # TODO - check makes sense with symmetric agents
    log_string = log_type
    s_masks_1, s_feats_1, s_probs_1 = s[0]
    s_masks_2, s_feats_2, s_probs_2 = s[1]
    feats_1, probs_1 = message_1
    feats_2, probs_2 = message_2
    current_exchange = len(feats_1)
    for i_sample in range(FLAGS.exchange_samples):
        prev_1 = torch.FloatTensor(FLAGS.m_dim).fill_(0)
        prev_2 = torch.FloatTensor(FLAGS.m_dim).fill_(0)
        for i_exchange in range(current_exchange):
            probs_1_i = probs_1[i_exchange][i_sample].data.tolist(
            )
            spark_1 = sparks(
                [1] + probs_1_i)[1:].encode('utf-8')
            probs_2_i = probs_2[i_exchange][i_sample].data.tolist(
            )
            spark_2 = sparks(
                [1] + probs_2_i)[1:].encode('utf-8')
            s_probs_1_i = s_probs_1[i_exchange][i_sample].data.tolist(
            )
            s_spark_1 = sparks(
                [1] + s_probs_1_i)[1:].encode('utf-8')

            binary_1 = feats_1[i_exchange][i_sample].data.cpu(
            )
            hamming_1 = (prev_1 - binary_1).abs().sum()
            prev_1 = binary_1
            binary_2 = feats_2[i_exchange][i_sample].data.cpu(
            )
            hamming_2 = (prev_2 - binary_2).abs().sum()
            prev_2 = binary_2

            msg_1 = "".join(
                map(str, map(int, binary_1.tolist())))
            msg_2 = "".join(
                map(str, map(int, binary_2.tolist())))
            if FLAGS.use_alpha:
                msg_1 = bin_to_alpha(msg_1)
                msg_2 = bin_to_alpha(msg_2)
            if i_exchange == 0:
                log_string += "\n{:>3}".format(i_sample)
            else:
                log_string += "\n   "
            log_string += "        {}".format(spark_1)
            log_string += "           {}    {}".format(
                s_spark_1, spark_2)
            log_string += "\n    {:>3} S: {} {:4}".format(
                i_exchange, msg_1, hamming_1)
            log_string += "    s={} R: {} {:4}".format(
                s_masks_1[1:][i_exchange][i_sample].data[0], msg_2, hamming_2)
    log_string += "\n"
    return log_string


def get_classification_loss_and_stats(predictions, targets):
    '''
    Arguments:
        - predictions: predicted logits for the classes
        - targets: correct classes
    Returns:
        - dist: logs of the predicted probability distribution over the classes
        - argmax: predicted class
        - argmax_prob: predicted class probability
        - ent: average entropy of the predicted probability distributions (over the batch)
        - nll_loss: Negative Log Likelihood loss between the predictions and targets
        - logs: Individual log likelihoods across the batch
    '''
    dist = F.log_softmax(predictions, dim=1)
    maxdist, argmax = dist.data.max(1)
    probs = F.softmax(predictions, dim=1)
    ent = (torch.log(probs + 1e-8) * probs).sum(1).mean()
    debuglogger.debug(f'Mean entropy: {ent.size()}')
    nll_loss = nn.NLLLoss()(dist, Variable(targets))
    logs = loglikelihood(Variable(dist.data),
                         Variable(targets.view(-1, 1)))
    return (dist, maxdist, argmax, ent, nll_loss, logs)


def run():
    flogger = FileLogger(FLAGS.log_file)
    logger = Logger(
        env=FLAGS.env, experiment_name=FLAGS.experiment_name, enabled=FLAGS.visdom)

    flogger.Log("Flag Values:\n" +
                json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True))

    if not os.path.exists(FLAGS.json_file):
        with open(FLAGS.json_file, "w") as f:
            f.write(json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True))

    # Initialize Agents
    agent1 = Agent(feature_type=FLAGS.img_feat,
                   feat_dim=FLAGS.img_feat_dim,
                   h_dim=FLAGS.img_h_dim,
                   m_dim=FLAGS.m_dim,  # TODO update flag
                   desc_dim=FLAGS.desc_dim,  # TODO update flag
                   num_classes=FLAGS.num_classes,  # TODO update flag
                   s_dim=1  # TODO check
                   use_binary=FLAGS.use_binary,
                   use_attn=FLAGS.visual_attn,
                   attn_dim=FLAGS.attn_dim)

    flogger.Log("Agent 1 Architecture: {}".format(agent1))
    total_params = sum([functools.reduce(lambda x, y: x * y, p.size(), 1.0)
                        for p in agent1.parameters()])
    flogger.Log("Total Parameters: {}".format(total_params))

    agent2 = Agent(feature_type=FLAGS.img_feat,
                   feat_dim=FLAGS.img_feat_dim,
                   h_dim=FLAGS.img_h_dim,
                   m_dim=FLAGS.m_dim,  # TODO update flag
                   desc_dim=FLAGS.desc_dim,  # TODO update flag
                   num_classes=FLAGS.num_classes,  # TODO update flag
                   s_dim=1  # TODO check
                   use_binary=FLAGS.use_binary,
                   use_attn=FLAGS.visual_attn,
                   attn_dim=FLAGS.attn_dim)

    flogger.Log("Agent 2 Architecture: {}".format(agent2))
    total_params = sum([functools.reduce(lambda x, y: x * y, p.size(), 1.0)
                        for p in agent2.parameters()])
    flogger.Log("Total Parameters: {}".format(total_params))

    # Optimizer
    # TODO potentially separate out baseline optimizers by selecting subset of params
    if FLAGS.optim_type == "SGD":
        optimizer_agent1 = optim.SGD(
            agent1.parameters(), lr=FLAGS.learning_rate)
        optimizer_agent2 = optim.SGD(
            agent2.parameters(), lr=FLAGS.learning_rate)
    elif FLAGS.optim_type == "Adam":
        optimizer_agent1 = optim.Adam(
            agent1.parameters(), lr=FLAGS.learning_rate)
        optimizer_agent2 = optim.Adam(
            agent2.parameters(), lr=FLAGS.learning_rate)
    elif FLAGS.optim_type == "RMSprop":
        optimizer_agent1 = optim.RMSprop(
            agent1.parameters(), lr=FLAGS.learning_rate)
        optimizer_agent2 = optim.RMSprop(
            agent2.parameters(), lr=FLAGS.learning_rate)
    else:
        raise NotImplementedError

    optimizers_dict = dict(optimizer_1=optimizer_agent1,
                           optimizer_2=optimizer_agent2)
    models_dict = dict(agent1=agent1, agent2=agent2)

    # Training metrics
    epoch = 0
    step = 0
    best_dev_acc = 0

    # Optionally load previously saved model
    # TODO check this works with new models
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
        # TODO fix for new agents
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
        # TODO fix for new agents
        extract_binary(FLAGS, load_hdf5, exchange, FLAGS.dev_file, FLAGS.batch_size_dev, epoch,
                       FLAGS.shuffle_dev, FLAGS.cuda, FLAGS.top_k_dev,
                       sender, receiver, desc_dev_dict, map_labels_dev, FLAGS.experiment_name)
        sys.exit()

    # Training loop
    while epoch < FLAGS.max_epoch:

        flogger.Log("Starting epoch: {}".format(epoch))

        # Read images randomly into batches - image_dim = [3, 227, 227]
        if FLAGS.dataset == "test":
            # TODO
            dataloader = load_toy()
        elif FLAGS.dataset == "simple":
            # TODO
            dataloader = load_shapes_data()
        else:
            raise NotImplementedError

        # Keep track of metrics
        batch_accuracy = {'total_nc': [],  # no communicaton
                          'total_com': [],  # after communication
                          'rewards': [],  # total_com - total_nc
                          'total_acc_both_nc': []  # % both agents right before comms
                          'total_acc_both_com': []  # % both agents right after comms
                          'total_acc_atl1_nc': []  # % at least 1 agent right before comms
                          'total_acc_atl1_com': []  # % at least 1 agent right after comms
                          'agent1_nc': [],  # no communicaton
                          'agent2_nc': [],  # no communicaton
                          'agent1_com': [],  # after communicaton
                          'agent2_com': []  # after communicaton
                          }
        dev_accuracy = {'total_nc': [],  # no communicaton
                        'total_com': [],  # after communication
                        'rewards': [],  # total_com - total_nc
                        'total_acc_both_nc': []  # % both agents right before comms
                        'total_acc_both_com': []  # % both agents right after comms
                        'total_acc_atl1_nc': []  # % at least 1 agent right before comms
                        'total_acc_atl1_com': []  # % at least 1 agent right after comms
                        'agent1_nc': [],  # no communicaton
                        'agent2_nc': [],  # no communicaton
                        'agent1_com': [],  # after communicaton
                        'agent2_com': []  # after communicaton
                        }

        # Iterate through batches
        for i_batch, batch in enumerate(dataloader):
            target = batch["target"]
            data = batch[FLAGS.img_feat]
            desc = batch["text_descriptions"]

            # GPU support
            if FLAGS.cuda:
                data = data.cuda()
                target = target.cuda()
                desc_train = desc_train.cuda()

            exchange_args = dict()
            exchange_args["data"] = data
            exchange_args["target"] = target
            exchange_args["desc"] = desc_train
            exchange_args["train"] = True
            exchange_args["break_early"] = not FLAGS.fixed_exchange

            s, message_1, message_2, y_all, r = exchange(agent1, agent2, exchange_args)

            s_masks_1, s_feats_1, s_probs_1 = s[0]
            s_masks_2, s_feats_2, s_probs_2 = s[1]
            feats_1, probs_1 = message_1
            feats_2, probs_2 = message_2
            y_nc = y_all[0]
            y = y_all[1]

            # Mask loss if dynamic exchange length
            if FLAGS.fixed_exchange:
                binary_s_masks = None
                binary_agent1_masks = None
                binary_agent2_masks = None
                bas_agent1_masks = None
                bas_agent2_masks = None
                y1_masks = None
                y2_masks = None
                outp_1 = y[0][-1]
                outp_2 = y[1][-1]
            else:
                # TODO
                # outp_1, ent_y1 = get_outp(y[0], y1_masks)
                # outp_2, ent_y2 = get_outp(y[1], y2_masks)
                pass

            # Obtain predictions, loss and stats agent 1
            # Before communication predictions
            (dist_1_nc, maxdist_1_nc, argmax_1_nc, ent_1_nc, nll_loss_1_nc, logs_1_nc) = get_classification_loss_and_stats(y_nc[0], target)
            # After communication predictions
            (dist_2_nc, maxdist_2_nc, argmax_2_nc, ent_2_nc, nll_loss_2_nc, logs_2_nc) = get_classification_loss_and_stats(y_nc[1], target)
            # Obtain predictions, loss and stats agent 1
            # Before communication predictions
            (dist_1, maxdist_1, argmax_1, ent_1, nll_loss_1_com, logs_1) = get_classification_loss_and_stats(outp_1, target)
            # After communication predictions
            (dist_2, maxdist_2, argmax_2, ent_2, nll_loss_2_com, logs_2) = get_classification_loss_and_stats(outp_2, target)

            # Store prediction entropies
            if FLAGS.fixed_exchange:
                ent_agent1_y = [ent_1]
                ent_agent2_y = [ent_2]
            else:
                # TODO - not implemented yet
                ent_agent1_y = []
                ent_agent2_y = []

            # Calculate accuracy
            accuracy_1_nc, correct_1_nc = calculate_accuracy(dist_1_nc)
            accuracy_1, correct_1 = calculate_accuracy(dist_1)
            accuracy_2_nc, correct_2_nc = calculate_accuracy(dist_2_nc)
            accuracy_2, correct_2 = calculate_accuracy(dist_2)

            # Calculate rewards
            total_correct_nc = correct_1_nc + correct_2_nc
            total_correct_com = correct_1 + correct_2
            total_accuracy_nc = (total_correct_nc == 2).sum() / batch_size
            total_accuracy_com = (total_correct_com == 2).sum() / batch_size
            atleast1_accuracy_nc = (total_correct_nc > 0).sum() / batch_size
            atleast1_accuracy_com = (total_correct_com > 0).sum() / batch_size
            # rewards = difference between performance before and after communication
            rewards = both_correct_com - both_correct_nc

            # Store results
            batch_accuracy['agent1_nc'].append(accuracy_1_nc)
            batch_accuracy['agent2_nc'].append(accuracy_2_nc)
            batch_accuracy['agent1_com'].append(accuracy_1)
            batch_accuracy['agent2_com'].append(accuracy_2)
            batch_accuracy['total_nc'].append(total_correct_nc)
            batch_accuracy['total_com'].append(total_correct_com)
            batch_accuracy['rewards'].append(rewards)
            batch_accuracy['total_acc_both_nc'].append(total_accuracy_nc)
            batch_accuracy['total_acc_both_com'].append(total_accuracy_com)
            batch_accuracy['total_acc_atl1_nc'].append(atleast1_accuracy_nc)
            batch_accuracy['total_acc_atl1_com'].append(atleast1_accuracy_com)

            # Cross entropy loss for each agent
            nll_loss_1 = nll_loss_1_nc + nll_loss_1_com
            nll_loss_2 = nll_loss_2_nc + nll_loss_2_com
            loss_agent1 = nll_loss_1
            loss_agent2 = nll_loss_2

            # If training communication channel
            if FLAGS.use_binary:
                if not FLAGS.fixed_exchange:
                    # TODO - fix
                    # Stop loss
                    # The receiver might have no z-loss if we stop after first message from sender.
                    debuglogger.warning(f'Error: multistep fixed exchange not implemented yet')
                    sys.exit()
                elif FLAGS.max_exchange == 1:
                    loss_binary_1, ent_bin_1 = calculate_loss_binary(feats_1, probs_1, rewards, r[0], FLAGS.entropy_agent1)
                    loss_binary_2, ent_bin_2 = calculate_loss_binary(feats_2, probs_2, rewards, r[1], FLAGS.entropy_agent2)
                    loss_baseline_1 = calculate_loss_bas(r[0], rewards)
                    loss_baseline_2 = calculate_loss_bas(r[1], rewards)
                    ent_agent1_bin = [ent_bin_1]
                    ent_agent2_bin = [ent_bin_2]
                elif FLAGS.max_exchange > 1:
                    # TODO
                    ent_agent1_bin = []
                    ent_agent2_bin = []
                    debuglogger.warning(f'Error: multistep fixed exchange not implemented yet')
                    sys.exit()

            if FLAGS.use_binary:
                loss_agent1 += loss_binary_1
                loss_agent2 += loss_binary_2
                if not FLAGS.fixed_exchange:
                    # TODO
                    pass
            else:
                loss_baseline_1 = Variable(torch.zeros(1))
                loss_baseline_2 = Variable(torch.zeros(1))

            # TODO - maybe separate out baseline training
            loss_agent1 += loss_baseline_1
            loss_agent2 += loss_baseline_2

            # Update agent1
            optimizer_agent1.zero_grad()
            loss_agent1.backward()
            nn.utils.clip_grad_norm(agent1.parameters(), max_norm=1.)
            optimizer_agent1.step()

            # Update agent2
            optimizer_agent2.zero_grad()
            loss_agent2.backward()
            nn.utils.clip_grad_norm(agent2.parameters(), max_norm=1.)
            optimizer_agent2.step()

            # Print logs regularly
            if step % FLAGS.log_interval == 0:
                # Average batch accuracy
                avg_batch_acc_total_nc = np.array(
                    batch_accuracy['total_acc_both_nc'][-FLAGS.log_interval:]).mean()
                avg_batch_acc_total_com = np.array(
                    batch_accuracy['total_acc_both_com'][-FLAGS.log_interval:]).mean()
                avg_batch_acc_atl1_nc = np.array(
                    batch_accuracy['total_acc_atl1_nc'][-FLAGS.log_interval:]).mean()
                avg_batch_acc_atl1_com = np.array(
                    batch_accuracy['total_acc_atl1_com'][-FLAGS.log_interval:]).mean()

                # Log accuracy
                log_acc = "Epoch: {} Step: {} Batch: {} Training Accuracy:\nBefore comms: Both correct: {} At least 1 correct: {}\nAfter comms: Both correct: {} At least 1 correct: {}".format(epoch, step, i_batch, avg_batch_acc_total_nc, avg_batch_acc_atl1_nc, avg_batch_acc_total_com, avg_batch_acc_atl1_com)
                flogger.Log(log_acc)

                # Agent1
                log_loss_agent1 = "Epoch: {} Step: {} Batch: {} Loss Agent1: {}".format(
                    epoch, step, i_batch, loss_agent1.data[0])
                flogger.Log(log_loss_agent1)
                # Agent 1 breakdown
                log_loss_agent1_detail = "Epoch: {} Step: {} Batch: {} Loss Agent1: NLL: {} (BC:{} / AC:{}), RL: {}, Baseline: {} ".format(
                    epoch, step, i_batch, nll_loss_1.data[0], nll_loss_1_nc.data[0], nll_loss_1_com.data[0], loss_binary_1.data[0], loss_baseline_1.data[0])
                flogger.Log(log_loss_agent1_detail)

                # Agent2
                log_loss_agent2 = "Epoch: {} Step: {} Batch: {} Loss Agent2: {}".format(
                    epoch, step, i_batch, loss_agent2.data[0])
                flogger.Log(log_loss_agent2)
                # Agent 2 breakdown
                log_loss_agent2_detail = "Epoch: {} Step: {} Batch: {} Loss Agent2: NLL: {} (BC:{} / AC:{}), RL: {}, Baseline: {} ".format(
                    epoch, step, i_batch, nll_loss_2.data[0], nll_loss_2_nc.data[0], nll_loss_2_com.data[0], loss_binary_2.data[0], loss_baseline_2.data[0])
                flogger.Log(log_loss_agent2_detail)

                # Log predictions
                log_pred = "Predictions: Target | Agent1 BC | Agent1 AC | Agent2 BC | Agent2 AC: {}".format(
                    torch.cat([target, argmax_1_nc, argmax_1, argmax_2_nc, argmax_2], 0).view(-1, FLAGS.batch_size))
                flogger.Log(log_pred)

                # Log Entropy for both Agents
                if FLAGS.use_binary:
                    if len(ent_agent1_bin) > 0:
                        log_ent_agent1_bin = "Entropy Agent1 Binary"
                        for i, ent in enumerate(ent_agent1_bin):
                            log_ent_agent1_bin += "\n{}. {}".format(
                                i, -ent.data[0])
                        log_ent_agent1_bin += "\n"
                        flogger.Log(log_ent_agent1_bin)

                    if len(ent_agent2_bin) > 0:
                        log_ent_agent2_bin = "Entropy Agent2 Binary"
                        for i, ent in enumerate(ent_agent2_bin):
                            log_ent_agent2_bin += "\n{}. {}".format(
                                i, -ent.data[0])
                        log_ent_agent2_bin += "\n"
                        flogger.Log(log_ent_agent2_bin)

                if len(ent_agent1_y) > 0:
                    log_ent_agent1_y = "Entropy Agent1 Predictions"
                    log_ent_agent1_y += "No comms entropy {}\n Comms entropy\n".format(ent_1_nc.data[0])
                    for i, ent in enumerate(ent_agent1_y):
                        log_ent_agent1_y += "\n{}. {}".format(i, -ent.data[0])
                    log_ent_agent1_y += "\n"
                    flogger.Log(log_ent_agent1_y)

                if len(ent_agent2_y) > 0:
                    log_ent_agent2_y = "Entropy Agent2 Predictions"
                    log_ent_agent2_y += "No comms entropy {}\n Comms entropy\n".format(ent_2_nc.data[0])
                    for i, ent in enumerate(ent_agent2_y):
                        log_ent_agent2_y += "\n{}. {}".format(i, -ent.data[0])
                    log_ent_agent2_y += "\n"
                    flogger.Log(log_ent_agent2_y)

                # Optionally print sampled and inferred binary vectors from
                # most recent exchange.
                if FLAGS.exchange_samples > 0:

                    log_train = log_exchange(s, message_1, message_2, current_exchange, log_type="Train:")
                    flogger.Log(log_train)

                    exchange_args["train"] = False
                    s, message_1, message_2, y_all, r = exchange(agent1, agent2, exchange_args)

                    log_train = log_exchange(s, message_1, message_2, current_exchange, log_type="Eval:")
                    flogger.Log(log_train)

                # Agent 1
                logger.log(key="Loss Agent 1 (Total)",
                           val=loss_agent1.data[0], step=step)
                logger.log(key="Loss Agent 1 (NLL)",
                           val=nll_loss_1.data[0], step=step)
                logger.log(key="Loss Agent 1 (NLL NC)",
                           val=nll_loss_1_nc.data[0], step=step)
                logger.log(key="Loss Agent 1 (NLL COM)",
                           val=nll_loss_1_com.data[0], step=step)
                if FLAGS.use_binary:
                    logger.log(key="Loss Agent 1 (RL)",
                               val=loss_binary_1.data[0], step=step)
                    logger.log(key="Loss Agent 1 (BAS)",
                               val=loss_baseline_1.data[0], step=step)
                    if not FLAGS.fixed_exchange:
                        # TODO
                        pass

                # Agent 2
                logger.log(key="Loss Agent 2 (Total)",
                           val=loss_agent2.data[0], step=step)
                logger.log(key="Loss Agent 2 (NLL)",
                           val=nll_loss_2.data[0], step=step)
                logger.log(key="Loss Agent 2 (NLL NC)",
                           val=nll_loss_2_nc.data[0], step=step)
                logger.log(key="Loss Agent 2 (NLL COM)",
                           val=nll_loss_2_com.data[0], step=step)
                if FLAGS.use_binary:
                    logger.log(key="Loss Agent 2 (RL)",
                               val=loss_binary_2.data[0], step=step)
                    logger.log(key="Loss Agent 2 (BAS)",
                               val=loss_baseline_2.data[0], step=step)
                    if not FLAGS.fixed_exchange:
                        # TODO
                        pass

                # Accuracy metrics
                logger.log(key="Training Accuracy (Total, BC)",
                           val=avg_batch_acc_total_nc, step=step)
                logger.log(key="Training Accuracy (At least 1, BC)",
                           val=avg_batch_acc_atl1_nc, step=step)
                logger.log(key="Training Accuracy (Total, COM)",
                           val=avg_batch_acc_total_com, step=step)
                logger.log(key="Training Accuracy (At least 1, COM)",
                           val=avg_batch_acc_atl1_com, step=step)

            # Report development accuracy
            # HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!
            if step % FLAGS.log_dev == 0:
                # TODO - fix for symmetric agents
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
            # break

        # Increment epoch
        epoch += 1
        # break

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
    gflags.DEFINE_float("entropy_agent1", None, "")
    gflags.DEFINE_float("entropy_agent2", None, "")

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
