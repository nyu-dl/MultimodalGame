import os
import sys
import json
import time
import numpy as np
import random
import h5py
import copy
import functools
import logging
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as _Variable
import torch.optim as optim
from torch.nn.parameter import Parameter

import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image

from sklearn.metrics import confusion_matrix

from misc import recursively_set_device, torch_save, torch_load, torch_load_communities
from misc import VisdomLogger as Logger
from misc import FileLogger
from misc import read_log_load
from misc import xavier_normal
from misc import build_mask

from agents import Agent
from dataset_loader import load_shapeworld_dataset
from community_util import sample_agents, build_train_matrix, build_eval_list

import gflags
FLAGS = gflags.FLAGS

SHAPES = ['circle', 'cross', 'ellipse', 'pentagon', 'rectangle', 'semicircle', 'square', 'triangle']
COLORS = ['blue', 'cyan', 'gray', 'green', 'magenta', 'red', 'yellow']
MAX_EXAMPLES_TO_SAVE = 200


def Variable(*args, **kwargs):
    var = _Variable(*args, **kwargs)
    if FLAGS.cuda:
        var = var.cuda()
    return var


def loglikelihood(log_prob, target):
    """
    Args: log softmax scores (N, C) where N is the batch size
          and C is the number of classes
    Output: log likelihood (N)
    """
    return log_prob.gather(1, target)


def store_exemplar_batch(data, data_type, logger, flogger):
    '''Writes MAX_EXAMPLES_TO_SAVE examples in the data to file for debugging

    data: dictionary containing data and results
        data = {"masked_im_1": [],
                "masked_im_2": [],
                "msg_1": [],
                "msg_2": [],
                "p": [],
                "target": [],
                "caption": [],
                "shapes": [],
                "colors": [],
                "texts": [],
                }
    data_type: flag giving the name of the data to be stored.
               e.g. "correct", "incorrect"
    '''
    debuglogger.info(f'Num {data_type}: {len(data["masked_im_1"])}')
    debuglogger.info("Writing exemplar batch to file...")
    assert len(data["masked_im_1"]) == len(data["masked_im_2"]) == len(data["p"]) == len(data["caption"]) == len(data["shapes"]) == len(data["colors"]) == len(data["texts"])
    num_examples = min(len(data["shapes"]), MAX_EXAMPLES_TO_SAVE)
    path = FLAGS.log_path
    prefix = FLAGS.experiment_name + "_" + data_type
    if not os.path.exists(path + "/" + prefix):
        os.makedirs(path + "/" + prefix)
    # Save images
    masked_im_1 = torch.stack(data["masked_im_1"][:num_examples], dim=0)
    debuglogger.debug(f'Masked im 1: {type(masked_im_1)}')
    debuglogger.debug(f'Masked im 1: {masked_im_1.size()}')
    save_image(masked_im_1, path + '/' + prefix + '/im1.png', nrow=16, pad_value=0.5)
    masked_im_2 = torch.stack(data["masked_im_2"][:num_examples], dim=0)
    save_image(masked_im_2, path + '/' + prefix + '/im2.png', nrow=16, pad_value=0.5)
    # Save other relevant info
    keys = ['p', 'caption', 'shapes', 'colors']
    for k in keys:
        filename = path + '/' + prefix + '/' + k + '.txt'
        with open(filename, "w") as wf:
            for i in range(num_examples):
                wf.write(f'Example {i+1}: {data[k][i]}\n')
    # Write texts
    filename = path + '/' + prefix + '/texts.txt'
    with open(filename, "w") as wf:
        for i in range(num_examples):
            s = ""
            for t in data["texts"][i]:
                s += t + ", "
            wf.write(f'Example {i+1}: {s}\n')
    # Print average and std p
    np_p = np.array(data["p"])
    debuglogger.info(f'p: mean: {np.mean(np_p)} std: {np.std(np_p)}')


def calc_message_mean_and_std(m_store):
    '''Calculate the mean and std deviation of messages per agent per shape, color and shape-color combination'''
    for k in m_store:
        msgs = m_store[k]["message"]
        msgs = torch.stack(msgs, dim=0)
        debuglogger.debug(f'Key: {k}, Count: {m_store[k]["count"]}, Messages: {msgs.size()}')
        mean = torch.mean(msgs, dim=0).cpu()
        std = torch.std(msgs, dim=0).cpu()
        m_store[k]["mean"] = mean
        m_store[k]["std"] = std
    return m_store


def log_message_stats(message_stats, logger, flogger, data_type, epoch, step, i_batch):
    ''' Helper function to write the message stats to file and log them to stdout
     Logs the mean and std deviation per set of messages per shape, per color and per shape-color for each message set.
      Additionally logs the distances between the mean message for each agent type per shape, color and shape-color'''
    debuglogger.info('Logging message stats')
    shape_colors = []
    for s in SHAPES:
        for c in COLORS:
            shape_colors.append(str(s) + "_" + str(c))
    # log shape stats
    for s in SHAPES:
        num = 0
        if s in message_stats[0]["shape"]:
            num = message_stats[0]["shape"][s]["count"]
        means = []
        stds = []
        for i, m in enumerate(message_stats):
            if s in message_stats[i]["shape"]:
                assert num == message_stats[i]["shape"][s]["count"]
                m = message_stats[i]["shape"][s]["mean"]
                st = message_stats[i]["shape"][s]["std"]
                means.append(m)
                stds.append(st)
        dists = []
        assert len(means) != 1
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                d = torch.dist(means[i], means[j])
                dists.append((i, j, d))
            if i == len(means) - 2:
                break
        logger.log(key=data_type + ": " + s + " message stats: count: ", val=num, step=step)
        for i in range(len(means)):
            logger.log(key=data_type + ": " + s + " message stats: Agent " + str(i) + ": mean: ",
                       val=means[i], step=step)
            logger.log(key=data_type + ": " + s + " message stats: Agent " + str(i) + ": std: ",
                       val=stds[i], step=step)
            flogger.Log("Epoch: {} Step: {} Batch: {} {} message stats: shape {}: count: {}, agent {}: mean: {}, std: {}".format(
                epoch, step, i_batch, data_type, s, num, i, means[i], stds[i]))
        for i in range(len(dists)):
            logger.log(key=data_type + ": " + s + " message stats: distances: [" + str(dists[i][0]) + ":" + str(dists[i][1]) + "]: ", val=dists[i][2], step=step)
        flogger.Log("Epoch: {} Step: {} Batch: {} {} message stats: shape {}: dists: {}".format(epoch, step, i_batch, data_type, s, dists))

    # log color stats
    for s in COLORS:
        num = 0
        if s in message_stats[0]["color"]:
            num = message_stats[0]["color"][s]["count"]
        means = []
        stds = []
        for i, m in enumerate(message_stats):
            if s in message_stats[i]["color"]:
                assert num == message_stats[i]["color"][s]["count"]
                m = message_stats[i]["color"][s]["mean"]
                st = message_stats[i]["color"][s]["std"]
                means.append(m)
                stds.append(st)
        dists = []
        assert len(means) != 1
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                d = torch.dist(means[i], means[j])
                dists.append((i, j, d))
            if i == len(means) - 2:
                break
        logger.log(key=data_type + ": " + s + " message stats: count: ", val=num, step=step)
        for i in range(len(means)):
            logger.log(key=data_type + ": " + s + " message stats: Agent " + str(i) + ": mean: ",
                       val=means[i], step=step)
            logger.log(key=data_type + ": " + s + " message stats: Agent " + str(i) + ": std: ",
                       val=stds[i], step=step)
            flogger.Log("Epoch: {} Step: {} Batch: {} {} message stats: color {}: count: {}, agent {}: mean: {}, std: {}".format(
                epoch, step, i_batch, data_type, s, num, i, means[i], stds[i]))
        for i in range(len(dists)):
            logger.log(key=data_type + ": " + s + " message stats: distances: [" + str(dists[i][0]) + ":" + str(dists[i][1]) + "]: ", val=dists[i][2], step=step)
        flogger.Log("Epoch: {} Step: {} Batch: {} {} message stats: color {}: dists: {}".format(epoch, step, i_batch, data_type, s, dists))

    # log shape - color stats
    for s in shape_colors:
        num = 0
        if s in message_stats[0]["shape_color"]:
            num = message_stats[0]["shape_color"][s]["count"]
        means = []
        stds = []
        for i, m in enumerate(message_stats):
            if s in message_stats[i]["shape_color"]:
                assert num == message_stats[i]["shape_color"][s]["count"]
                m = message_stats[i]["shape_color"][s]["mean"]
                st = message_stats[i]["shape_color"][s]["std"]
                means.append(m)
                stds.append(st)
        dists = []
        assert len(means) != 1
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                d = torch.dist(means[i], means[j])
                dists.append((i, j, d))
            if i == len(means) - 2:
                break
        logger.log(key=data_type + ": " + s + " message stats: count: ", val=num, step=step)
        for i in range(len(means)):
            logger.log(key=data_type + ": " + s + " message stats: Agent " + str(i) + ": mean: ", val=means[i], step=step)
            logger.log(key=data_type + ": " + s + " message stats: Agent " + str(i) + ": std: ", val=stds[i], step=step)
            flogger.Log("Epoch: {} Step: {} Batch: {} {} message stats: shape_color {}: count: {}, agent {}: mean: {}, std: {}".format(epoch, step, i_batch, data_type, s, num, i, means[i], stds[i]))
        for i in range(len(dists)):
            logger.log(key=data_type + ": " + s + " message stats: distances: [" + str(dists[i][0]) + ":" + str(dists[i][1]) + "]: ", val=dists[i][2], step=step)
        flogger.Log("Epoch: {} Step: {} Batch: {} {} message stats: shape_color {}: dists: {}".format(epoch, step, i_batch, data_type, s, dists))
    path = FLAGS.log_path + "/" + FLAGS.experiment_name + "_" + data_type + "_message_stats.pkl"
    pickle.dump(message_stats, open(path, "wb"))
    debuglogger.info(f'Saved message stats to log file')


def run_analyze_messages(data, data_type, logger, flogger, epoch, step, i_batch):
    '''Calculates the mean and std deviation per set of messages per shape, per color and per shape-color for each message set.
      Additionally caculates the distances between the mean message for each agent type per shape, color and shape-color

    data: dictionary containing log of data_type examples
    data_type: flag explaining the type of data
               e.g. "correct", "incorrect"

    Each message list should have the same length and the shape and colors lists
    Also saves the messages and analysis to file
    '''
    message_stats = []
    messages = [data["msg_1"], data["msg_2"]]
    shapes = data["shapes"]
    colors = data["colors"]
    for m_set in messages:
        assert len(m_set) == len(shapes)
        assert len(m_set) == len(colors)
        d = {"shape": {},
             "color": {},
             "shape_color": {}
             }
        message_stats.append(d)
    debuglogger.info(f'Messages: {len(messages[0])}, {len(messages[0][0])}')
    for i, m_set in enumerate(messages):
        s_store = message_stats[i]["shape"]
        c_store = message_stats[i]["color"]
        s_c_store = message_stats[i]["shape_color"]
        # Collect all messages
        j = 0
        for m, s, c in zip(m_set, shapes, colors):
            if s in s_store:
                # Potentially multiple exchanges
                for m_i in m:
                    s_store[s]["count"] += 1
                    s_store[s]["message"].append(m_i.data)
            else:
                s_store[s] = {}
                s_store[s]["count"] = 1
                s_store[s]["message"] = [m[0].data]
                if len(m) > 1:
                    for m_i in m[1:]:
                        s_store[s]["count"] += 1
                        s_store[s]["message"].append(m_i.data)
            if c in c_store:
                # Potentially multiple exchanges
                for m_i in m:
                    c_store[c]["count"] += 1
                    c_store[c]["message"].append(m_i.data)
            else:
                c_store[c] = {}
                c_store[c]["count"] = 1
                c_store[c]["message"] = [m[0].data]
                if len(m) > 1:
                    for m_i in m[1:]:
                        c_store[c]["count"] += 1
                        c_store[c]["message"].append(m_i.data)

            s_c = str(s) + "_" + str(c)
            if s_c in s_c_store:
                # Potentially multiple exchanges
                for m_i in m:
                    s_c_store[s_c]["count"] += 1
                    s_c_store[s_c]["message"].append(m_i.data)
            else:
                s_c_store[s_c] = {}
                s_c_store[s_c]["count"] = 1
                s_c_store[s_c]["message"] = [m[0].data]
                if len(m) > 1:
                    for m_i in m[1:]:
                        s_c_store[s_c]["count"] += 1
                        s_c_store[s_c]["message"].append(m_i.data)
            if j == 5:
                debuglogger.debug(f's_store: {s_store}')
                debuglogger.debug(f'c_store: {c_store}')
                debuglogger.debug(f's_c_store: {s_c_store}')
            j += 1
        # Calculate and log mean and std_dev
        s_store = calc_message_mean_and_std(s_store)
        c_store = calc_message_mean_and_std(c_store)
        s_c_store = calc_message_mean_and_std(s_c_store)
    log_message_stats(message_stats, logger, flogger, data_type, epoch, step, i_batch)


def add_data_point(batch, i, data_store, messages_1, messages_2, probs_1, probs_2):
    '''Adds the relevant data from a batch to a data store to analyze later'''
    data_store["masked_im_1"].append(batch["masked_im_1"][i])
    data_store["masked_im_2"].append(batch["masked_im_2"][i])
    data_store["p"].append(batch["p"][i])
    data_store["target"].append(batch["target"][i])
    data_store["caption"].append(batch["caption_str"][i])
    data_store["shapes"].append(batch["shapes"][i])
    data_store["colors"].append(batch["colors"][i])
    data_store["texts"].append(batch["texts_str"][i])
    # Add messages and probs from each exchange
    m_1 = []
    p_1 = []
    for exchange, prob in zip(messages_1, probs_1):
        m_1.append(exchange[i].data.cpu())
        p_1.append(prob[i].data.cpu())
    data_store["msg_1"].append(m_1)
    data_store["probs_1"].append(p_1)
    m_2 = []
    p_2 = []
    for exchange, prob in zip(messages_2, probs_2):
        m_2.append(exchange[i].data.cpu())
        p_2.append(prob[i].data.cpu())
    data_store["msg_2"].append(m_2)
    data_store["probs_2"].append(p_2)
    return data_store


def save_messages_and_stats(correct, incorrect, agent_tag):
    '''Saves all messages and message probs between two agents, along with relevant tags such as the shape, color and caption, and whether the message was correct or incorrect
    Data is stored as a list of dicts and pickled. Each dict corresponds to one exchange. The dicts have the following keys
        - msg_1: message(s) sent from agent 1
        - msg_2: message(s) sent from agent 2
        - probs_1: message 1 probs
        - probs_2: message 2 probs
        - caption: the correct caption
        - shape: the shape in the caption
        - color: the color in the caption
        - correct: boolean, whether both agents solved the task after communication
    '''
    message_data = []
    num_correct = len(correct["caption"])
    num_incorrect = len(incorrect["caption"])
    for i in range(num_correct):
        elem = {}
        elem["caption"] = correct["caption"][i]
        elem["msg_1"] = correct["msg_1"][i]
        elem["msg_2"] = correct["msg_2"][i]
        elem["probs_1"] = correct["probs_1"][i]
        elem["probs_2"] = correct["probs_2"][i]
        elem["shape"] = correct["shapes"][i]
        elem["color"] = correct["colors"][i]
        elem["correct"] = True
        message_data.append(elem)
    for i in range(num_incorrect):
        elem = {}
        elem["caption"] = incorrect["caption"][i]
        elem["msg_1"] = incorrect["msg_1"][i]
        elem["msg_2"] = incorrect["msg_2"][i]
        elem["probs_1"] = incorrect["probs_1"][i]
        elem["probs_2"] = incorrect["probs_2"][i]
        elem["shape"] = incorrect["shapes"][i]
        elem["color"] = incorrect["colors"][i]
        elem["correct"] = False
        message_data.append(elem)
    debuglogger.info(f'Saving messages...')
    debuglogger.info(f'{num_correct} correct, {num_incorrect} incorrect, {num_correct + num_incorrect} total')
    path = FLAGS.log_path + "/" + FLAGS.experiment_name + "_" + agent_tag + "_message_stats.pkl"
    pickle.dump(message_data, open(path, "wb"))
    debuglogger.info(f'Messages saved')


def eval_dev(dataset_path, top_k, agent1, agent2, logger, flogger, epoch, step, i_batch, in_domain_eval=True, callback=None, store_examples=False, analyze_messages=True, save_messages=False, agent_tag="_", agent_dicts=None):
    """
    Function computing development accuracy and other metrics
    """

    extra = dict()
    correct_to_analyze = {"masked_im_1": [],
                          "masked_im_2": [],
                          "msg_1": [],
                          "msg_2": [],
                          "probs_1": [],
                          "probs_2": [],
                          "p": [],
                          "target": [],
                          "caption": [],
                          "shapes": [],
                          "colors": [],
                          "texts": [],
                          }
    incorrect_to_analyze = {"masked_im_1": [],
                            "masked_im_2": [],
                            "msg_1": [],
                            "msg_2": [],
                            "probs_1": [],
                            "probs_2": [],
                            "p": [],
                            "target": [],
                            "caption": [],
                            "shapes": [],
                            "colors": [],
                            "texts": [],
                            }

    # Keep track of shapes and color accuracy
    shapes_accuracy = {}
    for s in SHAPES:
        shapes_accuracy[s] = {"correct": 0,
                              "total": 0}

    colors_accuracy = {}
    for c in COLORS:
        colors_accuracy[c] = {"correct": 0,
                              "total": 0}

    # Keep track of agent specific performance (given other agent gets it both right)
    agent1_performance = {"11": 0,  # both right
                          "01": 0,  # wrong before comms, right after
                          "10": 0,  # right before comms, wrong after
                          "00": 0,  # both wrong
                          "total": 0}

    agent2_performance = {"11": 0,  # both right
                          "01": 0,  # wrong before comms, right after
                          "10": 0,  # right before comms, wrong after
                          "00": 0,  # both wrong
                          "total": 0}

    # Keep track of conversation lengths
    conversation_lengths_1 = []
    conversation_lengths_2 = []

    # Keep track of message diversity
    hamming_1 = []
    hamming_2 = []

    # Keep track of labels
    true_labels = []
    pred_labels_1_nc = []
    pred_labels_1_com = []
    pred_labels_2_nc = []
    pred_labels_2_com = []

    # Keep track of number of correct observations
    total = 0
    total_correct_nc = 0
    total_correct_com = 0
    atleast1_correct_nc = 0
    atleast1_correct_com = 0

    # Keep track of score when messages are changed
    test_compositionality = {}
    if agent_dicts is not None:
        test_compositionality = {"total": 0, "correct": [], "agent_w_changed_msg": [], "shape": [], "color": [], "orig_shape": [], "orig_color": [], "originally_correct": []}

    # Load development images
    if in_domain_eval:
        eval_mode = "train"
        debuglogger.info("Evaluating on in domain validation set")
    else:
        eval_mode = FLAGS.dataset_eval_mode
        debuglogger.info("Evaluating on out of domain validation set")
    dev_loader = load_shapeworld_dataset(dataset_path, FLAGS.glove_path, eval_mode, FLAGS.dataset_size_dev, FLAGS.dataset_type, FLAGS.dataset_name, FLAGS.batch_size_dev, FLAGS.random_seed, FLAGS.shuffle_dev, FLAGS.img_feat, FLAGS.cuda, truncate_final_batch=False)

    for batch in dev_loader:
        target = batch["target"]
        im_feats_1 = batch["im_feats_1"]
        im_feats_2 = batch["im_feats_2"]
        p = batch["p"]
        desc = Variable(batch["texts_vec"])
        _batch_size = target.size(0)

        true_labels.append(target.cpu().numpy().reshape(-1))

        # GPU support
        if FLAGS.cuda:
            im_feats_1 = im_feats_1.cuda()
            im_feats_2 = im_feats_2.cuda()
            target = target.cuda()
            desc = desc.cuda()

        data = {"im_feats_1": im_feats_1,
                "im_feats_2": im_feats_2,
                "p": p}

        exchange_args = dict()
        exchange_args["data"] = data
        exchange_args["target"] = target
        exchange_args["desc"] = desc
        exchange_args["train"] = False
        exchange_args["break_early"] = not FLAGS.fixed_exchange

        s, message_1, message_2, y_all, r = exchange(
            agent1, agent2, exchange_args)

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
            pass

        # Before communication predictions
        # Obtain predictions, loss and stats agent 1
        (dist_1_nc, maxdist_1_nc, argmax_1_nc, ent_1_nc, nll_loss_1_nc,
         logs_1_nc) = get_classification_loss_and_stats(y_nc[0], target)
        # Obtain predictions, loss and stats agent 2
        (dist_2_nc, maxdist_2_nc, argmax_2_nc, ent_2_nc, nll_loss_2_nc,
         logs_2_nc) = get_classification_loss_and_stats(y_nc[1], target)
        # After communication predictions
        # Obtain predictions, loss and stats agent 1
        (dist_1, maxdist_1, argmax_1, ent_1, nll_loss_1_com,
         logs_1) = get_classification_loss_and_stats(outp_1, target)
        # Obtain predictions, loss and stats agent 2
        (dist_2, maxdist_2, argmax_2, ent_2, nll_loss_2_com,
         logs_2) = get_classification_loss_and_stats(outp_2, target)

        # Store top 1 prediction for confusion matrix
        pred_labels_1_nc.append(argmax_1_nc.cpu().numpy())
        pred_labels_1_com.append(argmax_1.cpu().numpy())
        pred_labels_2_nc.append(argmax_2_nc.cpu().numpy())
        pred_labels_2_com.append(argmax_2.cpu().numpy())

        # Calculate number of correct observations for different types
        accuracy_1_nc, correct_1_nc, top_1_1_nc = calculate_accuracy(
            dist_1_nc, target, FLAGS.batch_size_dev, FLAGS.top_k_dev)
        accuracy_1, correct_1, top_1_1 = calculate_accuracy(
            dist_1, target, FLAGS.batch_size_dev, FLAGS.top_k_dev)
        accuracy_2_nc, correct_2_nc, top_1_2_nc = calculate_accuracy(
            dist_2_nc, target, FLAGS.batch_size_dev, FLAGS.top_k_dev)
        accuracy_2, correct_2, top_1_2 = calculate_accuracy(
            dist_2, target, FLAGS.batch_size_dev, FLAGS.top_k_dev)
        batch_correct_nc = correct_1_nc.float() + correct_2_nc.float()
        batch_correct_com = correct_1.float() + correct_2.float()
        batch_correct_top_1_nc = top_1_1_nc.float() + top_1_2_nc.float()
        batch_correct_top_1_com = top_1_1.float() + top_1_2.float()

        debuglogger.info(f'A1 correct com: {correct_1}')
        debuglogger.info(f'A2 correct nc: {correct_2}')
        debuglogger.info(f'eval batch correct com: {batch_correct_com}')
        debuglogger.info(f'eval batch correct nc: {batch_correct_nc}')
        debuglogger.debug(
            f'eval batch top 1 correct com: {batch_correct_top_1_com}')
        debuglogger.debug(
            f'eval batch top 1 correct nc: {batch_correct_top_1_nc}')

        # Update accuracy counts
        total += float(_batch_size)
        total_correct_nc += (batch_correct_nc == 2).sum()
        total_correct_com += (batch_correct_com == 2).sum()
        atleast1_correct_nc += (batch_correct_nc > 0).sum()
        atleast1_correct_com += (batch_correct_com > 0).sum()

        debuglogger.debug(f'eval total correct com: {total_correct_com}')
        debuglogger.debug(f'eval total correct nc: {total_correct_nc}')
        debuglogger.debug(f'eval atleast1 correct com: {atleast1_correct_com}')
        debuglogger.debug(f'eval atleast1 correct nc: {atleast1_correct_nc}')

        debuglogger.debug(f'batch agent 1 nc correct: {correct_1_nc}')
        debuglogger.debug(f'batch agent 1 com correct: {correct_1}')
        debuglogger.debug(f'batch agent 2 nc correct: {correct_2_nc}')
        debuglogger.debug(f'batch agent 2 com correct: {correct_2}')

        # Track agent specific stats
        # Agent 1 given Agent 2 both correct
        a2_idx = (correct_2_nc.float() + correct_2.float()) == 2
        a1_00 = (a2_idx & ((correct_1_nc.float() + correct_1.float()) == 0)).sum()
        a1_10 = (a2_idx & ((correct_1_nc.float() + (1 - correct_1.float()) == 2))).sum()
        a1_01 = (a2_idx & (((1 - correct_1_nc.float()) + correct_1.float()) == 2)).sum()
        a1_11 = (a2_idx & ((correct_1_nc.float() + correct_1.float()) == 2)).sum()
        a1_tot = a2_idx.sum()
        assert a1_tot == (a1_00 + a1_01 + a1_10 + a1_11)

        agent1_performance["11"] += a1_11
        agent1_performance["01"] += a1_01
        agent1_performance["10"] += a1_10
        agent1_performance["00"] += a1_00
        agent1_performance["total"] += a1_tot

        # Agent 2 given Agent 1 both correct
        a1_idx = (correct_1_nc.float() + correct_1.float()) == 2
        a2_00 = (a1_idx & ((correct_2_nc.float() + correct_2.float()) == 0)).sum()
        a2_10 = (a1_idx & ((correct_2_nc.float() + (1 - correct_2.float()) == 2))).sum()
        a2_01 = (a1_idx & (((1 - correct_2_nc.float()) + correct_2.float()) == 2)).sum()
        a2_11 = (a1_idx & ((correct_2_nc.float() + correct_2.float()) == 2)).sum()
        a2_tot = a1_idx.sum()
        assert a2_tot == (a2_00 + a2_01 + a2_10 + a2_11)

        agent2_performance["11"] += a2_11
        agent2_performance["01"] += a2_01
        agent2_performance["10"] += a2_10
        agent2_performance["00"] += a2_00
        agent2_performance["total"] += a2_tot

        debuglogger.debug('Agent 1: total {}, 11: {}, 01: {} 00: {}, 10: {}'.format(
            agent1_performance["total"],
            agent1_performance["11"],
            agent1_performance["01"],
            agent1_performance["00"],
            agent1_performance["10"]))
        if agent1_performance["total"] > 0:
            debuglogger.debug('Agent 1: total {}, 11: {}, 01: {} 00: {}, 10: {}'.format(
                agent1_performance["total"] / agent1_performance["total"],
                agent1_performance["11"] / agent1_performance["total"],
                agent1_performance["01"] / agent1_performance["total"],
                agent1_performance["00"] / agent1_performance["total"],
                agent1_performance["10"] / agent1_performance["total"]))

        debuglogger.debug('Agent 2: total {}, 11: {}, 01: {} 00: {}, 10: {}'.format(
            agent2_performance["total"],
            agent2_performance["11"],
            agent2_performance["01"],
            agent2_performance["00"],
            agent2_performance["10"]))
        if agent2_performance["total"] > 0:
            debuglogger.debug('Agent 2: total {}, 11: {}, 01: {} 00: {}, 10: {}'.format(
                agent2_performance["total"] / agent2_performance["total"],
                agent2_performance["11"] / agent2_performance["total"],
                agent2_performance["01"] / agent2_performance["total"],
                agent2_performance["00"] / agent2_performance["total"],
                agent2_performance["10"] / agent2_performance["total"]))

        # Gather shape and color stats
        correct_indices_nc = batch_correct_nc == 2
        correct_indices_com = batch_correct_com == 2
        for _i in range(_batch_size):
            if batch['shapes'][_i] is not None:
                shape = batch['shapes'][_i]
                shapes_accuracy[shape]["total"] += 1
                if correct_indices_com[_i]:
                    shapes_accuracy[shape]["correct"] += 1
            if batch['colors'][_i] is not None:
                color = batch['colors'][_i]
                colors_accuracy[color]["total"] += 1
                if correct_indices_com[_i]:
                    colors_accuracy[color]["correct"] += 1
            # Time consuming, so only do this if necessary
            if store_examples or analyze_messages or save_messages:
                # Store batch data to analyze
                if correct_indices_com[_i]:
                    correct_to_analyze = add_data_point(batch, _i, correct_to_analyze, feats_1, feats_2, probs_1, probs_2)
                else:
                    incorrect_to_analyze = add_data_point(batch, _i, incorrect_to_analyze, feats_1, feats_2, probs_1, probs_2)
                if _i == 5:
                    debuglogger.debug(f'Message 1: {feats_1}, probs 1: {probs_1}')
                    debuglogger.debug(f'Message 2: {feats_2}, probs 2: {probs_2}')
                    debuglogger.debug(f'Correct dict: {correct_to_analyze}')
                    debuglogger.debug(f'Incorrect dict: {incorrect_to_analyze}')

        debuglogger.debug(f'shapes dict: {shapes_accuracy}')
        debuglogger.debug(f'colors dict: {colors_accuracy}')

        # Keep track of conversation lengths
        # TODO not relevant yet
        conversation_lengths_1 += torch.cat(s_feats_1,
                                            1).data.float().sum(1).view(-1).tolist()
        conversation_lengths_2 += torch.cat(s_feats_2,
                                            1).data.float().sum(1).view(-1).tolist()

        debuglogger.debug(f'Conversation length 1: {conversation_lengths_1}')
        debuglogger.debug(f'Conversation length 2: {conversation_lengths_2}')

        # Keep track of message diversity
        mean_hamming_1 = 0
        mean_hamming_2 = 0
        prev_1 = torch.FloatTensor(_batch_size, FLAGS.m_dim).fill_(0)
        prev_2 = torch.FloatTensor(_batch_size, FLAGS.m_dim).fill_(0)

        for msg in feats_1:
            mean_hamming_1 += (msg.data.cpu() - prev_1).abs().sum(1).mean()
            prev_1 = msg.data.cpu()
        mean_hamming_1 = mean_hamming_1 / float(len(feats_1))

        for msg in feats_2:
            mean_hamming_2 += (msg.data.cpu() - prev_2).abs().sum(1).mean()
            prev_2 = msg.data.cpu()
        mean_hamming_2 = mean_hamming_2 / float(len(feats_2))

        hamming_1.append(mean_hamming_1)
        hamming_2.append(mean_hamming_2)

        if callback is not None:
            callback_dict = dict(
                s_masks_1=s_masks_1,
                s_feats_1=s_feats_1,
                s_probs_1=s_probs_1,
                s_masks_2=s_masks_2,
                s_feats_2=s_feats_2,
                s_probs_2=s_probs_2,
                feats_1=feats_1,
                feats_2=feats_2,
                probs_1=probs_1,
                probs_2=probs_2,
                y_nc=y_nc,
                y=y)
            callback(agent1, agent2, batch, callback_dict)

        if agent_dicts is not None:
            # Test compositionality
            for _ in range(_batch_size):
                # Construct batch of size 1
                data = {"im_feats_1": im_feats_1[_].unsqueeze(0),
                        "im_feats_2": im_feats_2[_].unsqueeze(0),
                        "p": p[_]}
                exchange_args = dict()
                exchange_args["data"] = data
                exchange_args["desc"] = desc[_].unsqueeze(0)
                exchange_args["train"] = False
                exchange_args["break_early"] = not FLAGS.fixed_exchange
                exchange_args["test_compositionality"] = True
                # Construct candidate example to change message to
                # Only select examples where one agent is blind
                debuglogger.info(f'Iterating through texts and changing messages...')
                debuglogger.info(f'Agent with sight: {batch["non_blank_partition"][_]}')
                if batch['non_blank_partition'][_] != 0:
                    change_agent = batch['non_blank_partition'][_]
                    texts = batch["texts_str"][_]
                    s = batch["shapes"][_]
                    c = batch["colors"][_]
                    debuglogger.info(f'i: {_}, caption: {batch["caption_str"][_]}, original target: {target[_]}, Correct? {correct_1[_]}/{correct_2[_]}')
                    debuglogger.info(f'i: {_}, texts: {texts}')
                    debuglogger.info(f'i: {_}, texts_shapes: {batch["texts_shapes"][_]}')
                    debuglogger.info(f'i: {_}, texts_colors: {batch["texts_colors"][_]}')
                    for _t, t in enumerate(texts):
                        if _t != target[_]:
                            st = batch["texts_shapes"][_][_t]
                            ct = batch["texts_colors"][_][_t]
                            if (st == s and ct != c) or (st != s and ct == c):
                                exchange_args["target"] = _t
                                exchange_args["change_agent"] = change_agent
                                exchange_args["agent_dict"] = agent_dicts[change_agent - 1]
                                if ct != c:
                                    exchange_args["subtract"] = c
                                    exchange_args["add"] = ct
                                else:
                                    exchange_args["subtract"] = s
                                    exchange_args["add"] = st
                                if exchange_args["subtract"] is None or exchange_args["add"] is None:
                                    debuglogger.info(f'Skipping example due to None add or subtract...')
                                    continue
                                debuglogger.info(f'i: {_}, subtracting: {exchange_args["subtract"]}, adding: {exchange_args["add"]}')
                                # Play game, corrupting message
                                _s, message_1, message_2, y_all, r = exchange(
                                    agent1, agent2, exchange_args)

                                s_masks_1, s_feats_1, s_probs_1 = _s[0]
                                s_masks_2, s_feats_2, s_probs_2 = _s[1]
                                feats_1, probs_1 = message_1
                                feats_2, probs_2 = message_2
                                y_nc = y_all[0]
                                y = y_all[1]

                                # Only care about after communication predictions
                                score = None
                                new_target = torch.zeros(1).fill_(_t).long()
                                debuglogger.info(f'Old target: {target[_]}')
                                na, argmax_y1 = torch.max(y[0][-1], 1)
                                na, argmax_y2 = torch.max(y[1][-1], 1)
                                debuglogger.info(f'y1: {argmax_y1.data[0]}, y2: {argmax_y2.data[0]}, new_target: {new_target[0]}')
                                if FLAGS.cuda:
                                    new_target = new_target.cuda()
                                if change_agent == 1:
                                    # Calculate score for agent 2
                                    (dist_2_change, na, na, na, na, na) = get_classification_loss_and_stats(y[1][-1], new_target)
                                    na, na, top_1_2_change = calculate_accuracy(
                                        dist_2_change, new_target, 1, FLAGS.top_k_dev)
                                    score = top_1_2_change
                                else:
                                    # Calculate score for agent 1
                                    (dist_1_change, na, na, na, na, na) = get_classification_loss_and_stats(y[0][-1], new_target)
                                    na, na, top_1_1_change = calculate_accuracy(
                                        dist_1_change, new_target, 1, FLAGS.top_k_dev)
                                    score = top_1_1_change
                                debuglogger.info(f'i: New caption: {t}, new target: {_t}, change_agent: {change_agent}, correct: {score[0]}, originally correct: {correct_1[_]}/{correct_2[_]}')

                                # Store results
                                test_compositionality["total"] += 1
                                test_compositionality["orig_shape"].append(s)
                                test_compositionality["orig_color"].append(c)

                                if score[0] == 1:
                                    test_compositionality["correct"].append(1)
                                else:
                                    test_compositionality["correct"].append(0)
                                if change_agent == 2:
                                    # The other agent had their message changed
                                    test_compositionality["originally_correct"].append(correct_1[_])
                                    test_compositionality["agent_w_changed_msg"].append(1)
                                else:
                                    test_compositionality["originally_correct"].append(correct_2[_])
                                    test_compositionality["agent_w_changed_msg"].append(2)
                                if ct != c:
                                    test_compositionality["shape"].append(None)
                                    test_compositionality["color"].append(ct)
                                else:
                                    test_compositionality["shape"].append(st)
                                    test_compositionality["color"].append(None)

    if agent_dicts is not None:
        debuglogger.info(f'Total msg changed: {test_compositionality["total"]}, Correct: {sum(test_compositionality["correct"])}')

    if store_examples:
        debuglogger.info(f'Finishing iterating through dev set, storing examples...')
        store_exemplar_batch(correct_to_analyze, "correct", logger, flogger)
        store_exemplar_batch(incorrect_to_analyze, "incorrect", logger, flogger)
    if analyze_messages:
        debuglogger.info(f'Analyzing messages...')
        run_analyze_messages(correct_to_analyze, "correct", logger, flogger, epoch, step, i_batch)
    if save_messages:
        debuglogger.info(f'Saving messages...')
        save_messages_and_stats(correct_to_analyze, incorrect_to_analyze, agent_tag)

    # Print confusion matrix
    true_labels = np.concatenate(true_labels).reshape(-1)
    pred_labels_1_nc = np.concatenate(pred_labels_1_nc).reshape(-1)
    pred_labels_1_com = np.concatenate(pred_labels_1_com).reshape(-1)
    pred_labels_2_nc = np.concatenate(pred_labels_2_nc).reshape(-1)
    pred_labels_2_com = np.concatenate(pred_labels_2_com).reshape(-1)

    np.savetxt(FLAGS.conf_mat + "_1_nc", confusion_matrix(
        true_labels, pred_labels_1_nc), delimiter=',', fmt='%d')
    np.savetxt(FLAGS.conf_mat + "_1_com", confusion_matrix(
        true_labels, pred_labels_1_com), delimiter=',', fmt='%d')
    np.savetxt(FLAGS.conf_mat + "_2_nc", confusion_matrix(
        true_labels, pred_labels_2_nc), delimiter=',', fmt='%d')
    np.savetxt(FLAGS.conf_mat + "_2_com", confusion_matrix(
        true_labels, pred_labels_2_com), delimiter=',', fmt='%d')

    # Compute statistics
    conversation_lengths_1 = np.array(conversation_lengths_1)
    conversation_lengths_2 = np.array(conversation_lengths_2)
    hamming_1 = np.array(hamming_1)
    hamming_2 = np.array(hamming_2)
    extra['conversation_lengths_1_mean'] = conversation_lengths_1.mean()
    extra['conversation_lengths_1_std'] = conversation_lengths_1.std()
    extra['conversation_lengths_2_mean'] = conversation_lengths_2.mean()
    extra['conversation_lengths_2_std'] = conversation_lengths_2.std()
    extra['hamming_1_mean'] = hamming_1.mean()
    extra['hamming_2_mean'] = hamming_2.mean()
    extra['shapes_accuracy'] = shapes_accuracy
    extra['colors_accuracy'] = colors_accuracy
    extra['agent1_performance'] = agent1_performance
    extra['agent2_performance'] = agent2_performance
    extra['test_compositionality'] = test_compositionality

    debuglogger.debug(f'Eval total size: {total}')
    total_accuracy_nc = total_correct_nc / total
    total_accuracy_com = total_correct_com / total
    atleast1_accuracy_nc = atleast1_correct_nc / total
    atleast1_accuracy_com = atleast1_correct_com / total

    # Return accuracy
    return total_accuracy_nc, total_accuracy_com, atleast1_accuracy_nc, atleast1_accuracy_com, extra


def get_and_log_dev_performance(agent1, agent2, dataset_path, in_domain_eval, dev_accuracy_log, logger, flogger, domain, epoch, step, i_batch, store_examples, analyze_messages, save_messages, agent_tag, agent_dicts=None):
    '''Logs performance on the dev set'''
    total_accuracy_nc, total_accuracy_com, atleast1_accuracy_nc, atleast1_accuracy_com, extra = eval_dev(
        dataset_path, FLAGS.top_k_dev, agent1, agent2, logger, flogger, epoch, step, i_batch, in_domain_eval=in_domain_eval, callback=None, store_examples=store_examples, analyze_messages=analyze_messages, save_messages=save_messages, agent_tag=agent_tag, agent_dicts=agent_dicts)
    dev_accuracy_log['total_acc_both_nc'].append(total_accuracy_nc)
    dev_accuracy_log['total_acc_both_com'].append(total_accuracy_com)
    dev_accuracy_log['total_acc_atl1_nc'].append(atleast1_accuracy_nc)
    dev_accuracy_log['total_acc_atl1_com'].append(atleast1_accuracy_com)
    logger.log(key=domain + " Development Accuracy, both right, no comms", val=dev_accuracy_log['total_acc_both_nc'][-1], step=step)
    logger.log(key=domain + "Development Accuracy, both right, after comms", val=dev_accuracy_log['total_acc_both_com'][-1], step=step)
    logger.log(key=domain + "Development Accuracy, at least 1 right, no comms", val=dev_accuracy_log['total_acc_atl1_nc'][-1], step=step)
    logger.log(key=domain + "Development Accuracy, at least 1 right, after comms", val=dev_accuracy_log['total_acc_atl1_com'][-1], step=step)
    logger.log(key=domain + "Conversation Length A1 (avg)",
               val=extra['conversation_lengths_1_mean'], step=step)
    logger.log(key=domain + "Conversation Length A1 (std)",
               val=extra['conversation_lengths_1_std'], step=step)
    logger.log(key=domain + "Conversation Length A2 (avg)",
               val=extra['conversation_lengths_2_mean'], step=step)
    logger.log(key=domain + "Conversation Length A2 (std)",
               val=extra['conversation_lengths_2_std'], step=step)
    logger.log(key=domain + "Hamming 1 (avg)",
               val=extra['hamming_1_mean'], step=step)
    logger.log(key=domain + "Hamming 2 (avg)",
               val=extra['hamming_2_mean'], step=step)
    if extra['agent1_performance']["total"] > 0:
        logger.log(key=domain + " Development Accuracy: Agent 1 given Agent 2 both right: 01: ",
                   val=extra['agent1_performance']["01"] / extra['agent1_performance']["total"], step=step)
        logger.log(key=domain + " Development Accuracy: Agent 1 given Agent 2 both right: 11: ",
                   val=extra['agent1_performance']["11"] / extra['agent1_performance']["total"], step=step)
        logger.log(key=domain + " Development Accuracy: Agent 1 given Agent 2 both right: 00: ",
                   val=extra['agent1_performance']["00"] / extra['agent1_performance']["total"], step=step)
        logger.log(key=domain + " Development Accuracy: Agent 1 given Agent 2 both right: 10: ",
                   val=extra['agent1_performance']["10"] / extra['agent1_performance']["total"], step=step)
    else:
        logger.log(key=domain + " Development Accuracy: Agent 1 given Agent 2 both right: 0 examples",
                   val=None, step=step)
    if extra['agent2_performance']["total"] > 0:
        logger.log(key=domain + " Development Accuracy: Agent 2 given Agent 1 both right: 01: ",
                   val=extra['agent2_performance']["01"] / extra['agent2_performance']["total"], step=step)
        logger.log(key=domain + " Development Accuracy: Agent 2 given Agent 1 both right: 11: ",
                   val=extra['agent2_performance']["11"] / extra['agent2_performance']["total"], step=step)
        logger.log(key=domain + " Development Accuracy: Agent 2 given Agent 1 both right: 00: ",
                   val=extra['agent2_performance']["00"] / extra['agent2_performance']["total"], step=step)
        logger.log(key=domain + " Development Accuracy: Agent 2 given Agent 1 both right: 10: ",
                   val=extra['agent2_performance']["10"] / extra['agent2_performance']["total"], step=step)
    else:
        logger.log(key=domain + " Development Accuracy: Agent 1 given Agent 2 both right: 0 examples",
                   val=None, step=step)
    for k in extra['shapes_accuracy']:
        if extra['shapes_accuracy'][k]['total'] > 0:
            logger.log(key=domain + " Development Accuracy: " + k + " ", val=extra['shapes_accuracy'][k]['correct'] / extra['shapes_accuracy'][k]['total'], step=step)
    for k in extra['colors_accuracy']:
        if extra['colors_accuracy'][k]['total'] > 0:
            logger.log(key=domain + " Development Accuracy: " + k + " ", val=extra['colors_accuracy'][k]['correct'] / extra['colors_accuracy'][k]['total'], step=step)

    flogger.Log("Epoch: {} Step: {} Batch: {} {} Development Accuracy, both right, no comms: {}".format(
        epoch, step, i_batch, domain, dev_accuracy_log['total_acc_both_nc'][-1]))
    flogger.Log("Epoch: {} Step: {} Batch: {} {} Development Accuracy, both right, after comms: {}".format(
        epoch, step, i_batch, domain, dev_accuracy_log['total_acc_both_com'][-1]))
    flogger.Log("Epoch: {} Step: {} Batch: {} {} Development Accuracy, at least right, no comms: {}".format(
        epoch, step, i_batch, domain, dev_accuracy_log['total_acc_atl1_nc'][-1]))
    flogger.Log("Epoch: {} Step: {} Batch: {} {} Development Accuracy, at least 1 right, after comms: {}".format(
        epoch, step, i_batch, domain, dev_accuracy_log['total_acc_atl1_com'][-1]))

    flogger.Log("Epoch: {} Step: {} Batch: {} {} Development Accuracy CHECK, both right, no comms: {}".format(
        epoch, step, i_batch, domain, total_accuracy_nc))
    flogger.Log("Epoch: {} Step: {} Batch: {} {} Development Accuracy CHECK, both right, after comms: {}".format(
        epoch, step, i_batch, domain, total_accuracy_com))
    flogger.Log("Epoch: {} Step: {} Batch: {} {} Development Accuracy CHECK, at least right, no comms: {}".format(
        epoch, step, i_batch, domain, atleast1_accuracy_nc))
    flogger.Log("Epoch: {} Step: {} Batch: {} {} Development Accuracy CHECK, at least 1 right, after comms: {}".format(
        epoch, step, i_batch, domain, atleast1_accuracy_com))

    flogger.Log("Epoch: {} Step: {} Batch: {} {} Conversation Length 1 (avg/std): {}/{}".format(
        epoch, step, i_batch, domain, extra['conversation_lengths_1_mean'], extra['conversation_lengths_1_std']))
    flogger.Log("Epoch: {} Step: {} Batch: {} {} Conversation Length 2 (avg/std): {}/{}".format(
        epoch, step, i_batch, domain, extra['conversation_lengths_2_mean'], extra['conversation_lengths_2_std']))

    flogger.Log("Epoch: {} Step: {} Batch: {} {} Mean Hamming Distance (1/2): {}/{}"
                .format(epoch, step, i_batch, domain, extra['hamming_1_mean'], extra['hamming_2_mean']))

    flogger.Log('Agent 1: total {}, 11: {}, 01: {} 00: {}, 10: {}'.format(
        extra["agent1_performance"]["total"],
        extra["agent1_performance"]["11"],
        extra["agent1_performance"]["01"],
        extra["agent1_performance"]["00"],
        extra["agent1_performance"]["10"]))
    if extra["agent1_performance"]["total"] > 0:
        flogger.Log('Agent 1: total {}, 11: {}, 01: {} 00: {}, 10: {}'.format(
            extra["agent1_performance"]["total"] / extra["agent1_performance"]["total"],
            extra["agent1_performance"]["11"] / extra["agent1_performance"]["total"],
            extra["agent1_performance"]["01"] / extra["agent1_performance"]["total"],
            extra["agent1_performance"]["00"] / extra["agent1_performance"]["total"],
            extra["agent1_performance"]["10"] / extra["agent1_performance"]["total"]))

    flogger.Log('Agent 2: total {}, 11: {}, 01: {} 00: {}, 10: {}'.format(
        extra["agent2_performance"]["total"],
        extra["agent2_performance"]["11"],
        extra["agent2_performance"]["01"],
        extra["agent2_performance"]["00"],
        extra["agent2_performance"]["10"]))
    if extra["agent2_performance"]["total"] > 0:
        flogger.Log('Agent 2: total {}, 11: {}, 01: {} 00: {}, 10: {}'.format(
            extra["agent2_performance"]["total"] / extra["agent2_performance"]["total"],
            extra["agent2_performance"]["11"] / extra["agent2_performance"]["total"],
            extra["agent2_performance"]["01"] / extra["agent2_performance"]["total"],
            extra["agent2_performance"]["00"] / extra["agent2_performance"]["total"],
            extra["agent2_performance"]["10"] / extra["agent2_performance"]["total"]))

    for k in extra['shapes_accuracy']:
        if extra['shapes_accuracy'][k]['total'] > 0:
            flogger.Log('{}: total: {}, correct: {}, accuracy: {}'.format(
                k,
                extra['shapes_accuracy'][k]['total'],
                extra['shapes_accuracy'][k]['correct'],
                extra['shapes_accuracy'][k]['correct'] / extra['shapes_accuracy'][k]['total']))
    for k in extra['colors_accuracy']:
        if extra['colors_accuracy'][k]['total'] > 0:
            flogger.Log('{}: total: {}, correct: {}, accuracy: {}'.format(
                k,
                extra['colors_accuracy'][k]['total'],
                extra['colors_accuracy'][k]['correct'],
                extra['colors_accuracy'][k]['correct'] / extra['colors_accuracy'][k]['total']))

    if agent_dicts is not None:
        flogger.Log('Test compositionality performance')
        comp_dict = {}
        comp_data = extra['test_compositionality']
        correct_orig_correct = 0
        correct_orig_incorrect = 0
        orig_correct = 0
        orig_incorrect = 0
        correct = 0
        for i in range(len(comp_data['orig_shape'])):
            if comp_data['shape'][i] is not None:
                transform = comp_data['orig_shape'][i] + '_' + comp_data['shape'][i]
            else:
                transform = comp_data['orig_color'][i] + '_' + comp_data['color'][i]
            if transform not in comp_dict:
                comp_dict[transform] = {'total': 0, 'correct': 0, 'p_correct': 0,
                                        'orig_correct': {'total': 0, 'correct': 0, 'p_correct': 0},
                                        'orig_incorrect': {'total': 0, 'correct': 0, 'p_correct': 0}}
            comp_dict[transform]['total'] += 1
            if comp_data['correct'][i] == 1:
                comp_dict[transform]['correct'] += 1
                correct += 1
            if comp_data['originally_correct'][i] == 1:
                comp_dict[transform]['orig_correct']['total'] += 1
                orig_correct += 1
                if comp_data['correct'][i]:
                    comp_dict[transform]['orig_correct']['correct'] += 1
                    correct_orig_correct += 1
            else:
                comp_dict[transform]['orig_incorrect']['total'] += 1
                orig_incorrect += 1
                if comp_data['correct'][i]:
                    comp_dict[transform]['orig_incorrect']['correct'] += 1
                    correct_orig_incorrect += 1
        flogger.Log(f'Test comp: total: {len(comp_data["orig_shape"])} orig correct: {orig_correct} orig incorrect: {orig_incorrect}')
        flogger.Log(f'Total correct: {correct}/{correct / len(comp_data["orig_shape"])}, orig_correct_correct {correct_orig_correct}/{correct_orig_correct / orig_correct}, orig_incorrect_correct {correct_orig_incorrect}/{correct_orig_incorrect / orig_incorrect}')
        for key in comp_dict:
            comp_dict[key]['p_correct'] = comp_dict[key]['correct'] / comp_dict[key]['total']
            if comp_dict[key]['orig_correct']['total'] > 0:
                comp_dict[key]['orig_correct']['p_correct'] = comp_dict[key]['orig_correct']['correct'] / comp_dict[key]['orig_correct']['total']
            if comp_dict[key]['orig_incorrect']['total'] > 0:
                comp_dict[key]['orig_incorrect']['p_correct'] = comp_dict[key]['orig_incorrect']['correct'] / comp_dict[key]['orig_incorrect']['total']
            flogger.Log(f'Transform {key}: total: {comp_dict[key]["total"]}:{comp_dict[key]["correct"]}/{comp_dict[key]["p_correct"]}')
            flogger.Log(f'Transform {key}: orig_correct: {comp_dict[key]["orig_correct"]["total"]}:{comp_dict[key]["orig_correct"]["correct"]}/{comp_dict[key]["orig_correct"]["p_correct"]}')
            flogger.Log(f'Transform {key}: orig_incorrect: {comp_dict[key]["orig_incorrect"]["total"]}:{comp_dict[key]["orig_incorrect"]["correct"]}/{comp_dict[key]["orig_incorrect"]["p_correct"]}')

    return dev_accuracy_log, total_accuracy_com


def eval_community(eval_list, models_dict, dev_accuracy_log, logger, flogger, epoch, step, i_batch, store_examples=False, analyze_messages=False, save_messages=False, agent_tag=""):
    eval_type = {1: "Self com, agent connected to multiple pools",
                 2: "Self com, agent connected to just one pool",
                 3: "Within pool com, different agents, trained together",
                 4: "Within pool com, different agents, never trained together",
                 5: "Cross pool com, different agents, trained together",
                 6: "Cross pool com, different agents, never trained together"}
    for num, items in enumerate(eval_list):
        logger.log(key="Dev accuracy: ",
                   val=eval_type[num + 1], step=step)
        flogger.Log(f'Dev accuracy: {eval_type[num + 1]}, step: {step}')
        print(items)
        if type(items[0]) is list:
            for nested in items:
                for elem in nested:
                    if elem is None:
                        pass
                    else:
                        (i, j) = elem
                        agent1 = models_dict["agent" + str(i + 1)]
                        agent2 = models_dict["agent" + str(j + 1)]
                        domain = f'In Domain Dev: Agent {i + 1} | Agent {j + 1}, ids [{id(agent1)}]/[{id(agent2)}]: '
                        _, _ = get_and_log_dev_performance(agent1, agent2, FLAGS.dataset_indomain_valid_path, True, dev_accuracy_log, logger, flogger, domain, epoch, step, i_batch, store_examples, analyze_messages, save_messages, agent_tag)
        else:
            for elem in items:
                if elem is None:
                    pass
                else:
                    (i, j) = elem
                    agent1 = models_dict["agent" + str(i + 1)]
                    agent2 = models_dict["agent" + str(j + 1)]
                    domain = f'In Domain Dev: Agent {i + 1} | Agent {j + 1}, ids [{id(agent1)}]/[{id(agent2)}]: '
                    _, _ = get_and_log_dev_performance(agent1, agent2, FLAGS.dataset_indomain_valid_path, True, dev_accuracy_log, logger, flogger, domain, epoch, step, i_batch, store_examples, analyze_messages, save_messages, agent_tag)


def corrupt_message(corrupt_region, agent, binary_message):
    # Obtain mask
    mask = Variable(build_mask(corrupt_region, agent.m_dim))
    mask_broadcast = mask.view(1, agent_1.m_dim).expand_as(binary_message)
    # Subtract the mask to change values, but need to get absolute value
    # to set -1 values to 1 to essentially "flip" all the bits.
    binary_message = (binary_message - mask_broadcast).abs()
    return binary_message


def exchange(a1, a2, exchange_args):
    """Run a batched conversation between two agents.

    There are two parts to an exchange:
        1. Each agent receives part of an image, and uses this to select the corresponding text from a selection of texts
        2. Agents communicate for a number of steps, then each select the corresponding text again from the same selection of texts

    Exchange Args:
        data: Image features
            - dict containing the image features for agent 1 and agent 2, and the percentage of the
              image each agent received
              e.g.  { "im_feats_1": im_feats_1,
                      "im_feats_2": im_feats_2,
                      "p": p}
        target: Class labels.
        desc: List of description vectors.
        train: Boolean value indicating training mode (True) or evaluation mode (False).
        break_early: Boolean value. If True, then terminate batched conversation if both agents are satisfied

    Function Args:
        a1: agent1
        a2: agent2
        exchange_args: Other useful arguments.

    Returns:
        s: Contains all stop bits [a1, a2]
            a1 = agent1 (Masks, Values, Probabilties)
            a2 = agent2 (Masks, Values, Probabilties)
        message_1: All agent_1 messages. (Values, Probabilities)
        message_2: All agent_2 messages. (Values, Probabilities)
        y_all = [y_nc, y]
            y_nc: (y_1_nc, y_2_nc)
                y_1_nc: Agent 1 predictions before communication
                y_2_nc: Agent 2 predictions before communication
            y = (y_1, y_2)
                y_1: All predictions that were made by agent 1 after communication
                y_2: All predictions that were made by agent 2 after communication
        r = (r_1, r_2)
            r_1: Estimated rewards of agent_1.
            r_2: Estimated rewards of agent_2.
    """

    # Randomly select which agent goes first
    who_goes_first = None
    if FLAGS.randomize_comms:
        if random.random() < 0.5:
            agent1 = a1
            agent2 = a2
            who_goes_first = 1
            debuglogger.debug(f'Agent 1 communicates first')
        else:
            agent1 = a2
            agent2 = a1
            who_goes_first = 2
            debuglogger.debug(f'Agent 2 communicates first')
    else:
        agent1 = a1
        agent2 = a2
        who_goes_first = 1
        debuglogger.debug(f'Agent 1 communicates first')

    data = exchange_args["data"]
    # TODO extend implementation to include data context
    data_context = None
    target = exchange_args["target"]
    desc = exchange_args["desc"]
    train = exchange_args["train"]
    break_early = exchange_args.get("break_early", False)
    corrupt = exchange_args.get("corrupt", False)
    corrupt_region = exchange_args.get("corrupt_region", None)
    test_compositionality = exchange_args.get("test_compositionality", False)
    batch_size = data["im_feats_1"].size(0)

    # Pad with one column of ones.
    stop_mask_1 = [Variable(torch.ones(batch_size, 1).byte())]
    stop_feat_1 = []
    stop_prob_1 = []
    stop_mask_2 = [Variable(torch.ones(batch_size, 1).byte())]
    stop_feat_2 = []
    stop_prob_2 = []
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

    # First message (default is 0)
    m_binary = Variable(torch.FloatTensor(batch_size, agent1.m_dim).fill_(
        FLAGS.first_msg), volatile=not train)
    if FLAGS.cuda:
        m_binary = m_binary.cuda()

    if train:
        agent1.train()
        agent2.train()
    else:
        agent1.eval()
        agent2.eval()

    agent1.reset_state()
    agent2.reset_state()

    # The message is ignored initially
    use_message = False
    # Run data through both agents
    if data_context is not None:
        # No data context at the moment - # TODO
        debuglogger.warning(f'Data context not supported currently')
        sys.exit()
    else:
        # debuglogger.info(f'Inside exchange: Train status: {train}, Message: {m_binary}')
        s_1e, m_1e, y_1e, r_1e = agent1(
            data['im_feats_1'],
            m_binary,
            0,
            desc,
            use_message,
            batch_size,
            train)

        s_2e, m_2e, y_2e, r_2e = agent2(
            data['im_feats_2'],
            m_binary,
            0,
            desc,
            use_message,
            batch_size,
            train)

    # Add no message selections to results
    # Need to be consistent about storing the a1 and a2's even though their roles are randomized during each exchange
    # agent1 and agent2 is a local name that refers to the order of communication. Storage refers to global labels a1 and a2
    if who_goes_first == 1:
        y_1_nc = y_1e
        y_2_nc = y_2e
    else:
        y_1_nc = y_2e
        y_2_nc = y_1e

    for i_exchange in range(FLAGS.max_exchange):
        debuglogger.debug(
            f' ================== EXCHANGE {i_exchange} ====================')
        # The messages are now used
        use_message = True

        # Agent 1's message
        m_1e_binary, m_1e_probs = m_1e
        # Store last but one communication from agent1 to train with
        # Last communication is "communication into the void" since the other
        # agent never gets it.
        m_binary_1, m_probs_1 = m_1e

        # Optionally corrupt agent 1's message
        if corrupt:
            m_1e_binary = corrupt_message(corrupt_region, agent1, m_1e_binary)

        # Optionally change agent 1's message
        if test_compositionality:
            if (who_goes_first == 1 and exchange_args["change_agent"] == 1) or (who_goes_first == 2 and exchange_args["change_agent"] == 2):
                debuglogger.info(f'Inside exchange: Changing agent 1 message...')
                a_dict = exchange_args["agent_dict"]
                add = a_dict[exchange_args["add"]].unsqueeze(0)
                sub = a_dict[exchange_args["subtract"]].unsqueeze(0)
                if FLAGS.cuda:
                    add = add.cuda()
                    sub = sub.cuda()
                new_m_prob = m_1e_probs.data - sub + add
                new_m_binary = torch.clamp(new_m_prob, 0, 1).round()
                m_1e_binary = _Variable(new_m_binary)
                debuglogger.info(f'Old msg: {m_binary_1}, old prob: {m_probs_1}')
                debuglogger.info(f'New msg: {new_m_binary}, new prob: {new_m_prob}')

        # Optionally mute communication channel
        if FLAGS.no_comms_channel:
            m_1e_binary = m_binary

        # Run data through agent 2
        if data_context is not None:
            # TODO
            debuglogger.warning(f'Data context not supported currently')
            sys.exit()
        else:
            # debuglogger.info(f'Inside exchange: Train status: {train}, Message to agent 2: {m_1e_binary}')
            s_2e, m_2e, y_2e, r_2e = agent2(
                data['im_feats_2'],
                m_1e_binary,
                i_exchange,
                desc,
                use_message,
                batch_size,
                train)

        # Agent 2's message
        m_2e_binary, m_2e_probs = m_2e

        # Optionally corrupt agent 2's message
        if corrupt:
            debuglogger.info(f'Corrupting message...')
            m_2e_binary = corrupt_message(corrupt_region, agent2, m_2e_binary)

        # Optionally change agent 2's message
        if test_compositionality:
            if (who_goes_first == 1 and exchange_args["change_agent"] == 2) or (who_goes_first == 2 and exchange_args["change_agent"] == 1):
                debuglogger.info(f'Inside exchange: Changing agent 2 message...')
                a_dict = exchange_args["agent_dict"]
                add = a_dict[exchange_args["add"]].unsqueeze(0)
                sub = a_dict[exchange_args["subtract"]].unsqueeze(0)
                if FLAGS.cuda:
                    add = add.cuda()
                    sub = sub.cuda()
                new_m_prob = m_2e_probs.data - sub + add
                new_m_binary = torch.clamp(new_m_prob, 0, 1).round()
                m_2e_binary = _Variable(new_m_binary)
                debuglogger.info(f'Old msg: {m_binary_1}, old prob: {m_probs_1}')
                debuglogger.info(f'New msg: {new_m_binary}, new prob: {new_m_prob}')

        # Optionally mute communication channel
        if FLAGS.no_comms_channel:
            m_2e_binary = m_binary

        # Run data through agent 1
        if data_context is not None:
            pass
        else:
            # debuglogger.info(f'Inside exchange: Train status: {train}, Message to agent 1: {m_2e_binary}')
            s_1e, m_1e, y_1e, r_1e = agent1(
                data['im_feats_1'],
                m_2e_binary,
                i_exchange,
                desc,
                use_message,
                batch_size,
                train)

        # Store rest of communication and stop information
        s_binary_1, s_prob_1 = s_1e
        s_binary_2, s_prob_2 = s_2e
        m_binary_2, m_probs_2 = m_2e

        # Save for later
        # TODO check stop mask
        # Need to be consistent about storing the a1 and a2's even though their roles are randomized during each exchange
        # agent1 and agent2 is a local name that refers to the order of communication. Storage refers to global labels a1 and a2
        if who_goes_first == 1:
            stop_mask_1.append(torch.min(stop_mask_1[-1], s_binary_1.byte()))
            stop_mask_2.append(torch.min(stop_mask_2[-1], s_binary_2.byte()))
            stop_feat_1.append(s_binary_1)
            stop_feat_2.append(s_binary_2)
            stop_prob_1.append(s_prob_1)
            stop_prob_2.append(s_prob_2)
            feats_1.append(m_binary_1)
            feats_2.append(m_binary_2)
            probs_1.append(m_probs_1)
            probs_2.append(m_probs_2)
            y_1.append(y_1e)
            y_2.append(y_2e)
            r_1.append(r_1e)
            r_2.append(r_2e)
        else:
            stop_mask_1.append(torch.min(stop_mask_2[-1], s_binary_2.byte()))
            stop_mask_2.append(torch.min(stop_mask_1[-1], s_binary_1.byte()))
            stop_feat_1.append(s_binary_2)
            stop_feat_2.append(s_binary_1)
            stop_prob_1.append(s_prob_2)
            stop_prob_2.append(s_prob_1)
            feats_1.append(m_binary_2)
            feats_2.append(m_binary_1)
            probs_1.append(m_probs_2)
            probs_2.append(m_probs_1)
            y_1.append(y_2e)
            y_2.append(y_1e)
            r_1.append(r_2e)
            r_2.append(r_1e)

        # Terminate exchange if everyone is done conversing
        if break_early and stop_mask_1[-1].float().sum().data[0] == 0 and stop_mask_2[-1].float().sum().data[0] == 0:
            break

    # The final mask must always be zero.
    stop_mask_1[-1].data.fill_(0)
    stop_mask_2[-1].data.fill_(0)

    s = [(stop_mask_1, stop_feat_1, stop_prob_1),
         (stop_mask_2, stop_feat_2, stop_prob_2)]
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
    '''Calculates the reinforcement learning loss on the agent communication vectors'''
    log_p_z = Variable(binary_features.data) * torch.log(binary_probs + 1e-8) + \
        (1 - Variable(binary_features.data)) * \
        torch.log(1 - binary_probs + 1e-8)
    log_p_z = log_p_z.sum(1)
    weight = Variable(rewards) - \
        Variable(baseline_rewards.clone().detach().data)
    if rewards.size(0) > 1:  # Ensures weights are not larger than 1
        weight = weight / np.maximum(1., torch.std(weight.data))
    loss = torch.mean(-1 * weight * log_p_z)

    # Must do both sides of negent, otherwise is skewed towards 0.
    initial_negent = (torch.log(binary_probs + 1e-8) * binary_probs).sum(1).mean()
    inverse_negent = (torch.log((1. - binary_probs) + 1e-8) * (1. - binary_probs)).sum(1).mean()
    negentropy = initial_negent + inverse_negent

    if entropy_penalty is not None:
        loss = (loss + entropy_penalty * negentropy)
    return loss, negentropy


def multistep_loss_binary(binary_features, binary_probs, rewards, baseline_rewards, masks, entropy_penalty):
    ''' Same as calculate loss binary but with multiple communications per exchange'''
    if masks is not None:
        # TODO - implement for new agents
        pass
    else:
        outp = list(map(lambda feat, prob, scores: calculate_loss_binary(feat, prob, rewards, scores, entropy_penalty), binary_features, binary_probs, baseline_rewards))
        losses = [o[0] for o in outp]
        entropies = [o[1] for o in outp]
        loss = sum(losses) / len(binary_features)
    return loss, entropies


def calculate_loss_bas(baseline_scores, rewards):
    loss_bas = nn.MSELoss()(baseline_scores, Variable(rewards))
    return loss_bas


def multistep_loss_bas(baseline_scores, rewards, masks):
    if masks is not None:
        # TODO - check for new agents
        pass
    else:
        losses = list(map(lambda scores: calculate_loss_bas(scores, rewards), baseline_scores))
        loss = sum(losses) / len(baseline_scores)
    return loss


def calculate_accuracy(prediction_dist, target, batch_size, top_k):
    '''Calculates the prediction accuracy using correct@top_k
       Returns:
        - accuracy: float
        - correct: boolean vector of batch_size elements.
                   1 indicates prediction was correct@top_k
        - top_1: boolean vector of batch_size elements.
                   1 indicates prediction was correct (top 1)
    '''
    assert batch_size == target.size(0)
    target_exp = target.view(-1, 1).expand(batch_size, top_k)
    top_k_ind = torch.from_numpy(
        prediction_dist.data.cpu().numpy().argsort()[:, -top_k:]).long()
    correct = (top_k_ind == target_exp.cpu()).sum(dim=1)
    top_1_ind = torch.from_numpy(
        prediction_dist.data.cpu().numpy().argsort()[:, -1:]).long()
    top_1 = (top_1_ind == target.view(-1, 1).cpu()).sum(dim=1)
    accuracy = correct.sum() / float(batch_size)
    return accuracy, correct, top_1


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
    debuglogger.debug(f'Mean entropy: {-ent.data[0]}')
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
    agents = []
    optimizers_dict = {}
    models_dict = {}
    frozen_agents = {}
    # Only used for training agent communities
    num_agents_per_community = None
    intra_pool_connect_p = None
    train_vec_prob = None  # Probability distribution over agent pairs, used to sample pairs per batch
    agent_idx_list = None  # Mapping from train_vec indices to agent pairs
    eval_agent_list = None  # List of agent pairs to evaluate each dev eval

    # Check agent setup
    if FLAGS.num_agents > 2 and not (FLAGS.agent_pools or FLAGS.agent_communities):
        flogger.Log("{} is too many agents. There can only be two agents if FLAGS.agent_pools and FLAGS.agent_communities are false".format(FLAGS.num_agents))
        sys.exit()
    if FLAGS.agent_pools and FLAGS.agent_communities:
        flogger.Log("You cannot train an agent pool and community at the same time. Please select one of the other")
        sys.exit()
    if not FLAGS.use_binary:
        flogger.Log("Non binary agent communication not implemented yet. Set FLAGS.use_binary to True")
        sys.exit()
    if FLAGS.agent_communities:
        if FLAGS.num_communities != len(FLAGS.num_agents_per_community):
            flogger.Log("Number of communities doesn't match length of community size list")
            sys.exit()
        if FLAGS.num_communities != len(FLAGS.community_checkpoints):
            flogger.Log("Number of communities doesn't match length of community checkpoint list. Use None to train from scratch.")
            sys.exit()
        if FLAGS.num_communities != len(FLAGS.intra_pool_connect_p):
            flogger.Log("Number of communities doesn't match length of intra community connectivity list")
            sys.exit()
        if FLAGS.community_type == "chain" and FLAGS.num_communities <= 2:
            flogger.Log("There must be at least 3 communities to train in a chain")
            sys.exit()
        num_agents_per_community = [int(x) for x in FLAGS.num_agents_per_community]
        intra_pool_connect_p = [float(x) for x in FLAGS.intra_pool_connect_p]
        if FLAGS.num_agents != sum(num_agents_per_community):
            flogger.Log("Total number of agents does not match sum of agents in each community")
            sys.exit()
        flogger.Log(f'Total number of agents: {FLAGS.num_agents}, Agents per community: {num_agents_per_community}')
    if FLAGS.test_compositionality and not FLAGS.eval_only:
        flogger.Log("Can only test compositionality in eval only mode")
        sys.exit()
    if FLAGS.test_compositionality:
        if len(FLAGS.agent_dicts) != FLAGS.num_agents:
            flogger.Log("There must be a code dictionary per agent")
            sys.exit()

    # Create agents
    for _ in range(FLAGS.num_agents):
        agent = Agent(im_feature_type=FLAGS.img_feat,
                      im_feat_dim=FLAGS.img_feat_dim,
                      h_dim=FLAGS.h_dim,
                      m_dim=FLAGS.m_dim,
                      desc_dim=FLAGS.desc_dim,
                      num_classes=FLAGS.num_classes,
                      s_dim=FLAGS.s_dim,
                      use_binary=FLAGS.use_binary,
                      use_attn=FLAGS.visual_attn,
                      attn_dim=FLAGS.attn_dim,
                      use_MLP=FLAGS.use_MLP,
                      cuda=FLAGS.cuda,
                      im_from_scratch=FLAGS.improc_from_scratch,
                      dropout=FLAGS.dropout)

        flogger.Log("Agent {} id: {} Architecture: {}".format(_ + 1, id(agent), agent))
        total_params = sum([functools.reduce(lambda x, y: x * y, p.size(), 1.0)
                            for p in agent.parameters()])
        flogger.Log("Total Parameters: {}".format(total_params))
        agents.append(agent)

        # Optimizer
        if FLAGS.optim_type == "SGD":
            optimizer_agent = optim.SGD(
                agent.parameters(), lr=FLAGS.learning_rate)
        elif FLAGS.optim_type == "Adam":
            optimizer_agent = optim.Adam(
                agent.parameters(), lr=FLAGS.learning_rate)
        elif FLAGS.optim_type == "RMSprop":
            optimizer_agent = optim.RMSprop(
                agent.parameters(), lr=FLAGS.learning_rate)
        else:
            raise NotImplementedError

        optim_name = "optimizer_agent" + str(_ + 1)
        agent_name = "agent" + str(_ + 1)
        optimizers_dict[optim_name] = optimizer_agent
        models_dict[agent_name] = agent

    if FLAGS.agent_communities:
        flogger.Log(f'Training {FLAGS.num_communities} communities of agents, type: {FLAGS.community_type}')
    flogger.Log("Number of agents: {}".format(len(agents)))
    for k in optimizers_dict:
        flogger.Log("Optimizer {}: {}".format(k, optimizers_dict[k]))

    # Create additional data structures if training with agent communites
    # These handle to sampling of agents during training, and selection of agent pairs when evaluating of the dev set
    if FLAGS.agent_communities:
        (train_vec_prob, agent_idx_list) = build_train_matrix(num_agents_per_community, FLAGS.community_type, intra_pool_connect_p, FLAGS.inter_pool_connect_p, FLAGS.ratio_adjust, FLAGS.intra_inter_ratio)
        eval_agent_list = build_eval_list(num_agents_per_community, FLAGS.community_type, train_vec_prob)
        flogger.Log(f'Train matrix: {torch.from_numpy(np.reshape(train_vec_prob, (FLAGS.num_agents, FLAGS.num_agents)))}')
        flogger.Log(f'Eval agent list: {eval_agent_list}')

    # Training metrics
    epoch = 0
    step = 0
    best_dev_acc = 0

    # Optionally load previously saved models
    if os.path.exists(FLAGS.checkpoint):
        flogger.Log("Loading from: " + FLAGS.checkpoint)
        data = torch_load(FLAGS.checkpoint, models_dict, optimizers_dict)
        flogger.Log("Loaded at step: {} and best dev acc: {}".format(
            data['step'], data['best_dev_acc']))
        step = data['step']
        best_dev_acc = data['best_dev_acc']
        if FLAGS.agent_communities:
            # Load train and eval matrices
            train_vec_prob = pickle.load(open(FLAGS.checkpoint + '_train_vec.pkl', 'rb'))
            agent_idx_list = pickle.load(open(FLAGS.checkpoint + '_agent_idx_list.pkl', 'rb'))
            eval_agent_list = pickle.load(open(FLAGS.checkpoint + '_eval_agent_list.pkl', 'rb'))
            flogger.Log(f'Train matrix: {torch.from_numpy(np.reshape(train_vec_prob, (FLAGS.num_agents, FLAGS.num_agents)))}')
            flogger.Log(f'Eval agent list: {eval_agent_list}')
    elif FLAGS.agent_communities:
        torch_load_communities(FLAGS.community_checkpoints, num_agents_per_community, models_dict, optimizers_dict)

    if FLAGS.agent_communities:
        # Save additional community data
        pickle.dump(train_vec_prob, open(FLAGS.checkpoint + '_train_vec.pkl', 'wb'))
        pickle.dump(agent_idx_list, open(FLAGS.checkpoint + '_agent_idx_list.pkl', 'wb'))
        pickle.dump(eval_agent_list, open(FLAGS.checkpoint + '_eval_agent_list.pkl', 'wb'))

    # Copy agents to measure how much the language has changed
    frozen_agents = copy.deepcopy(models_dict)

    # GPU support
    if FLAGS.cuda:
        for m in models_dict.values():
            m.cuda()
        for m in frozen_agents.values():
            m.cuda()
        for o in optimizers_dict.values():
            recursively_set_device(o.state_dict(), gpu=0)

    # If training / evaluating with pools of agents or communities of agents sample with each batch
    if FLAGS.agent_pools or FLAGS.agent_communities:
        agent1 = None
        agent2 = None
        optimizer_agent1 = None
        optimizer_agent2 = None
        agent_idxs = [None, None]
    # Otherwise keep agents fixed for each batch
    else:
        if FLAGS.num_agents == 1:
            agent1 = agents[0]
            agent2 = agents[0]
            optimizer_agent1 = optimizers_dict["optimizer_agent1"]
            optimizer_agent2 = optimizers_dict["optimizer_agent1"]
            agent_idxs = [1, 1]
        else:
            agent1 = agents[0]
            agent2 = agents[1]
            optimizer_agent1 = optimizers_dict["optimizer_agent1"]
            optimizer_agent2 = optimizers_dict["optimizer_agent2"]
            agent_idxs = [1, 2]

    # Alternatives to training.
    if FLAGS.eval_only:
        if not os.path.exists(FLAGS.checkpoint):
            raise Exception("Must provide valid checkpoint.")

        debuglogger.info("Evaluating on in domain validation set")
        step = i_batch = epoch = 0

        # Storage for results
        dev_accuracy_id = []
        dev_accuracy_ood = []
        dev_accuracy_self_com = []
        for i in range(FLAGS.num_agents):
            dev_accuracy_id.append({'total_acc_both_nc': [],  # % both agents right before comms
                                    'total_acc_both_com': [],  # % both agents right after comms
                                    'total_acc_atl1_nc': [],  # % at least 1 agent right before comms
                                    'total_acc_atl1_com': []  # % at least 1 agent right after comms
                                    })

            dev_accuracy_ood.append({'total_acc_both_nc': [],  # % both agents right before comms
                                     'total_acc_both_com': [],  # % both agents right after comms
                                     'total_acc_atl1_nc': [],  # % at least 1 agent right before comms
                                     'total_acc_atl1_com': []  # % at least 1 agent right after comms
                                     })
            dev_accuracy_self_com.append({'total_acc_both_nc': [],  # % both agents right before comms
                                          'total_acc_both_com': [],  # % both agents right after comms
                                          'total_acc_atl1_nc': [],  # % at least 1 agent right before comms
                                          'total_acc_atl1_com': []  # % at least 1 agent right after comms
                                          })
        if FLAGS.eval_xproduct:
            # Complete evaluation on all possible pairs of agents
            for i in range(FLAGS.num_agents):
                for j in range(FLAGS.num_agents):
                    flogger.Log("Agent 1: {}".format(i + 1))
                    logger.log(key="Agent 1: ", val=i + 1, step=step)
                    agent1 = models_dict["agent" + str(i + 1)]
                    flogger.Log("Agent 2: {}".format(j + 1))
                    logger.log(key="Agent 2: ", val=j + 1, step=step)
                    agent2 = models_dict["agent" + str(j + 1)]
                    if i == 0 and j == 0:
                        # Report in domain development accuracy and store examples
                        dev_accuracy_id[i], total_accuracy_com = get_and_log_dev_performance(agent1, agent2, FLAGS.dataset_indomain_valid_path, True, dev_accuracy_id[i], logger, flogger, f'In Domain Agents {i + 1},{j + 1}', epoch, step, i_batch, store_examples=True, analyze_messages=False, save_messages=False, agent_tag=f'eval_only_A_{i + 1}_{j + 1}')
                    else:
                        # Report in domain development accuracy
                        dev_accuracy_id[i], total_accuracy_com = get_and_log_dev_performance(agent1, agent2, FLAGS.dataset_indomain_valid_path, True, dev_accuracy_id[i], logger, flogger, f'In Domain Agents {i + 1},{j + 1}', epoch, step, i_batch, store_examples=False, analyze_messages=False, save_messages=False, agent_tag=f'eval_only_A_{i + 1}_{j + 1}')
        elif FLAGS.test_compositionality:
            # Load agent dictionaries
            code_dicts = []
            for i in range(FLAGS.num_agents):
                _ = pickle.load(open(FLAGS.agent_dicts[i], 'rb'))
                code_dicts.append(_)
            for i in range(FLAGS.num_agents - 1):
                flogger.Log("Agent 1: {}".format(i + 1))
                logger.log(key="Agent 1: ", val=i + 1, step=step)
                agent1 = models_dict["agent" + str(i + 1)]
                flogger.Log("Agent 2: {}".format(i + 2))
                logger.log(key="Agent 2: ", val=i + 2, step=step)
                agent2 = models_dict["agent" + str(i + 2)]
                dev_accuracy_id[i], total_accuracy_com = get_and_log_dev_performance(agent1, agent2, FLAGS.dataset_indomain_valid_path, True, dev_accuracy_id[i], logger, flogger, f'In Domain Agents {i + 1},{i + 2}', epoch, step, i_batch, store_examples=True, analyze_messages=False, save_messages=False, agent_tag=f'eval_only_A_{i + 1}_{i + 2}', agent_dicts=(code_dicts[i], code_dicts[i + 1]))
                # dev_accuracy_id[i], total_accuracy_com = get_and_log_dev_performance(agent1, agent1, FLAGS.dataset_indomain_valid_path, True, dev_accuracy_id[i], logger, flogger, f'In Domain Agents {i + 1},{i + 1}', epoch, step, i_batch, store_examples=True, analyze_messages=False, save_messages=False, agent_tag=f'eval_only_A_{i + 1}_{i + 1}', agent_dicts=(code_dicts[i], code_dicts[i]))
        elif FLAGS.agent_communities:
            eval_community(eval_agent_list, models_dict, dev_accuracy_id[0], logger, flogger, epoch, step, i_batch, store_examples=False, analyze_messages=False, save_messages=False, agent_tag="no_tag")
        else:
            # For the pairs of agents calculate results
            # Applies to both pools of agents and an agent pair
            # for i in range(FLAGS.num_agents - 1):
            #     flogger.Log("Agent 1: {}".format(i + 1))
            #     logger.log(key="Agent 1: ", val=i + 1, step=step)
            #     agent1 = models_dict["agent" + str(i + 1)]
            #     flogger.Log("Agent 2: {}".format(i + 2))
            #     logger.log(key="Agent 2: ", val=i + 2, step=step)
            #     agent2 = models_dict["agent" + str(i + 2)]
            #     if i == 0:
            #         # Report in domain development accuracy and analyze messages and store examples
            #         dev_accuracy_id[i], total_accuracy_com = get_and_log_dev_performance(agent1, agent2, FLAGS.dataset_indomain_valid_path, True, dev_accuracy_id[i], logger, flogger, f'In Domain Agents {i + 1},{i + 2}', epoch, step, i_batch, store_examples=True, analyze_messages=False, save_messages=True, agent_tag=f'eval_only_A_{i + 1}_{i + 2}')
            #     else:
            #         # Report in domain development accuracy
            #         dev_accuracy_id[i], total_accuracy_com = get_and_log_dev_performance(
            #             agent1, agent2, FLAGS.dataset_indomain_valid_path, True, dev_accuracy_id[i], logger, flogger, f'In Domain Agents {i + 1},{i + 2}', epoch, step, i_batch, store_examples=False, analyze_messages=False, save_messages=True, agent_tag=f'eval_only_A_{i + 1}_{i + 2}')
            #
            #     # Report out of domain development accuracy
            #     dev_accuracy_ood[i], total_accuracy_com = get_and_log_dev_performance(agent1, agent2, FLAGS.dataset_path, False, dev_accuracy_ood[i], logger, flogger, f'Out of Domain Agents {i + 1},{i + 2}', epoch, step, i_batch, store_examples=False, analyze_messages=False, save_messages=False, agent_tag="")

            # Report in domain development accuracy when agents communicate with themselves
            for i in range(FLAGS.num_agents):
                agent1 = models_dict["agent" + str(i + 1)]
                agent2 = copy.deepcopy(agent1)
                flogger.Log("Agent {} self communication: id {}".format(i + 1, id(agent)))
                dev_accuracy_self_com[i], total_accuracy_com = get_and_log_dev_performance(
                    agent1, agent2, FLAGS.dataset_indomain_valid_path, True, dev_accuracy_self_com[i], logger, flogger, "Agent " + str(i + 1) + " self communication: In Domain", epoch, step, i_batch, store_examples=False, analyze_messages=False, save_messages=True, agent_tag=f'eval_only_self_com_A_{i + 1}')
        sys.exit()

    # Training loop
    while epoch < FLAGS.max_epoch:

        flogger.Log("Starting epoch: {}".format(epoch))

        # Read dataset randomly into batches
        if FLAGS.dataset == "shapeworld":
            dataloader = load_shapeworld_dataset(FLAGS.dataset_path, FLAGS.glove_path, FLAGS.dataset_mode, FLAGS.dataset_size_train, FLAGS.dataset_type,
                                                 FLAGS.dataset_name, FLAGS.batch_size, FLAGS.random_seed, FLAGS.shuffle_train, FLAGS.img_feat, FLAGS.cuda, truncate_final_batch=False)
        else:
            raise NotImplementedError

        # Keep track of metrics
        batch_accuracy = {'total_nc': [],  # no communicaton
                          'total_com': [],  # after communication
                          'rewards_1': [],  # agent 1 rewards
                          'rewards_2': [],  # agent 2 rewards
                          'total_acc_both_nc': [],  # % both agents right before comms
                          'total_acc_both_com': [],  # % both agents right after comms
                          'total_acc_atl1_nc': [],  # % at least 1 agent right before comms
                          'total_acc_atl1_com': [],  # % at least 1 agent right after comms
                          'agent1_nc': [],  # no communicaton
                          'agent2_nc': [],  # no communicaton
                          'agent1_com': [],  # after communicaton
                          'agent2_com': []  # after communicaton
                          }

        dev_accuracy_id = {'total_acc_both_nc': [],  # % both agents right before comms
                           'total_acc_both_com': [],  # % both agents right after comms
                           'total_acc_atl1_nc': [],  # % at least 1 agent right before comms
                           'total_acc_atl1_com': []  # % at least 1 agent right after comms
                           }

        dev_accuracy_ood = {'total_acc_both_nc': [],  # % both agents right before comms
                            'total_acc_both_com': [],  # % both agents right after comms
                            'total_acc_atl1_nc': [],  # % at least 1 agent right before comms
                            'total_acc_atl1_com': []  # % at least 1 agent right after comms
                            }
        dev_accuracy_id_pairs = []
        dev_accuracy_self_com = []
        for i in range(FLAGS.num_agents):
            dev_accuracy_id_pairs.append({'total_acc_both_nc': [],  # % both agents right before comms
                                          'total_acc_both_com': [],  # % both agents right after comms
                                          'total_acc_atl1_nc': [],  # % at least 1 agent right before comms
                                          'total_acc_atl1_com': []  # % at least 1 agent right after comms
                                          })
            dev_accuracy_self_com.append({'total_acc_both_nc': [],  # % both agents right before comms
                                          'total_acc_both_com': [],  # % both agents right after comms
                                          'total_acc_atl1_nc': [],  # % at least 1 agent right before comms
                                          'total_acc_atl1_com': []  # % at least 1 agent right after comms
                                          })

        # Iterate through batches
        for i_batch, batch in enumerate(dataloader):
            debuglogger.debug(f'Batch {i_batch}')

            # Select agents if training with pools or communities
            if FLAGS.agent_pools:
                idx = random.randint(0, len(agents) - 1)
                agent1 = agents[idx]
                optimizer_agent1 = optimizers_dict["optimizer_agent" + str(idx + 1)]
                agent_idxs[0] = idx + 1
                old_idx = idx
                if FLAGS.with_replacement:
                    # Sampling second agent with replacement
                    idx = random.randint(0, len(agents) - 1)
                else:
                    # Sampling second agent without replacement
                    while idx == old_idx:
                        idx = random.randint(0, len(agents) - 1)
                agent2 = agents[idx]
                optimizer_agent2 = optimizers_dict["optimizer_agent" + str(idx + 1)]
                agent_idxs[1] = idx + 1
            elif FLAGS.agent_communities:
                (idx1, idx2) = sample_agents(train_vec_prob, agent_idx_list)
                agent1 = agents[idx1]
                optimizer_agent1 = optimizers_dict["optimizer_agent" + str(idx1 + 1)]
                agent_idxs[0] = idx1 + 1
                agent2 = agents[idx2]
                optimizer_agent2 = optimizers_dict["optimizer_agent" + str(idx2 + 1)]
                agent_idxs[1] = idx2 + 1
            debuglogger.info(f'Agent 1: {agent_idxs[0]}, Agent 1: {agent_idxs[1]}')

            # Converted to Variable in get_classification_loss_and_stats
            target = batch["target"]
            im_feats_1 = batch["im_feats_1"]  # Already Variable
            im_feats_2 = batch["im_feats_2"]  # Already Variable
            p = batch["p"]
            desc = Variable(batch["texts_vec"])

            # GPU support
            if FLAGS.cuda:
                im_feats_1 = im_feats_1.cuda()
                im_feats_2 = im_feats_2.cuda()
                target = target.cuda()
                desc = desc.cuda()

            data = {"im_feats_1": im_feats_1,
                    "im_feats_2": im_feats_2,
                    "p": p}

            exchange_args = dict()
            exchange_args["data"] = data
            exchange_args["target"] = target
            exchange_args["desc"] = desc
            exchange_args["train"] = True
            exchange_args["break_early"] = not FLAGS.fixed_exchange

            s, message_1, message_2, y_all, r = exchange(
                agent1, agent2, exchange_args)

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
                pass

            # Before communication predictions
            # Obtain predictions, loss and stats agent 1
            (dist_1_nc, maxdist_1_nc, argmax_1_nc, ent_1_nc, nll_loss_1_nc,
             logs_1_nc) = get_classification_loss_and_stats(y_nc[0], target)
            # Obtain predictions, loss and stats agent 2
            (dist_2_nc, maxdist_2_nc, argmax_2_nc, ent_2_nc, nll_loss_2_nc,
             logs_2_nc) = get_classification_loss_and_stats(y_nc[1], target)
            # After communication predictions
            # Obtain predictions, loss and stats agent 1
            (dist_1, maxdist_1, argmax_1, ent_1, nll_loss_1_com,
             logs_1) = get_classification_loss_and_stats(outp_1, target)
            # Obtain predictions, loss and stats agent 2
            (dist_2, maxdist_2, argmax_2, ent_2, nll_loss_2_com,
             logs_2) = get_classification_loss_and_stats(outp_2, target)

            # Store prediction entropies
            if FLAGS.fixed_exchange:
                ent_agent1_y = [ent_1]
                ent_agent2_y = [ent_2]
            else:
                # TODO - not implemented yet
                ent_agent1_y = []
                ent_agent2_y = []

            # Calculate accuracy
            accuracy_1_nc, correct_1_nc, top_1_1_nc = calculate_accuracy(
                dist_1_nc, target, FLAGS.batch_size, FLAGS.top_k_train)
            accuracy_1, correct_1, top_1_1 = calculate_accuracy(
                dist_1, target, FLAGS.batch_size, FLAGS.top_k_train)
            accuracy_2_nc, correct_2_nc, top_1_2_nc = calculate_accuracy(
                dist_2_nc, target, FLAGS.batch_size, FLAGS.top_k_train)
            accuracy_2, correct_2, top_1_2 = calculate_accuracy(
                dist_2, target, FLAGS.batch_size, FLAGS.top_k_train)

            # Calculate accuracy
            total_correct_nc = correct_1_nc.float() + correct_2_nc.float()
            total_correct_com = correct_1.float() + correct_2.float()
            total_accuracy_nc = (total_correct_nc ==
                                 2).sum() / float(FLAGS.batch_size)
            total_accuracy_com = (total_correct_com ==
                                  2).sum() / float(FLAGS.batch_size)
            atleast1_accuracy_nc = (
                total_correct_nc > 0).sum() / float(FLAGS.batch_size)
            atleast1_accuracy_com = (
                total_correct_com > 0).sum() / float(FLAGS.batch_size)
            # Calculate rewards
            # rewards = difference between performance before and after communication
            # Only use top 1
            total_correct_top_1_nc = top_1_1_nc.float() + top_1_2_nc.float()
            total_correct_top_1_com = top_1_1.float() + top_1_2.float()
            if FLAGS.cooperative_reward:
                rewards_1 = (total_correct_top_1_com.float() - total_correct_top_1_nc.float())
                rewards_2 = rewards_1
            else:
                rewards_1 = top_1_1.float()
                rewards_2 = top_1_2.float()
            debuglogger.debug(
                f'total correct top 1 com: {total_correct_top_1_com}')
            debuglogger.debug(
                f'total correct top 1 nc: {total_correct_top_1_nc}')
            debuglogger.debug(f'total correct com: {total_correct_com}')
            debuglogger.debug(f'total correct nc: {total_correct_nc}')
            debuglogger.debug(f'rewards_1: {rewards_1}')
            debuglogger.debug(f'rewards_2: {rewards_2}')
            # Store results
            batch_accuracy['agent1_nc'].append(accuracy_1_nc)
            batch_accuracy['agent2_nc'].append(accuracy_2_nc)
            batch_accuracy['agent1_com'].append(accuracy_1)
            batch_accuracy['agent2_com'].append(accuracy_2)
            batch_accuracy['total_nc'].append(total_correct_nc)
            batch_accuracy['total_com'].append(total_correct_com)
            batch_accuracy['rewards_1'].append(rewards_1)
            batch_accuracy['rewards_1'].append(rewards_1)
            batch_accuracy['total_acc_both_nc'].append(total_accuracy_nc)
            batch_accuracy['total_acc_both_com'].append(total_accuracy_com)
            batch_accuracy['total_acc_atl1_nc'].append(atleast1_accuracy_nc)
            batch_accuracy['total_acc_atl1_com'].append(atleast1_accuracy_com)

            # Cross entropy loss for each agent
            nll_loss_1 = FLAGS.nll_loss_weight_nc * nll_loss_1_nc + \
                FLAGS.nll_loss_weight_com * nll_loss_1_com
            nll_loss_2 = FLAGS.nll_loss_weight_nc * nll_loss_2_nc + \
                FLAGS.nll_loss_weight_com * nll_loss_2_com
            loss_agent1 = nll_loss_1
            loss_agent2 = nll_loss_2

            # If training communication channel
            if FLAGS.use_binary:
                if not FLAGS.fixed_exchange:
                    # TODO - fix
                    # Stop loss
                    # TODO - check old use of entropy_s
                    # The receiver might have no z-loss if we stop after first message from sender.
                    debuglogger.warning(
                        f'Error: multistep adaptive exchange not implemented yet')
                    sys.exit()
                elif FLAGS.max_exchange == 1:
                    loss_binary_1, ent_bin_1 = calculate_loss_binary(
                        feats_1[0], probs_1[0], rewards_1, r[0][0], FLAGS.entropy_agent1)
                    loss_binary_2, ent_bin_2 = calculate_loss_binary(
                        feats_2[0], probs_2[0], rewards_2, r[1][0], FLAGS.entropy_agent2)
                    loss_baseline_1 = calculate_loss_bas(r[0][0], rewards_1)
                    loss_baseline_2 = calculate_loss_bas(r[1][0], rewards_2)
                    ent_agent1_bin = [ent_bin_1]
                    ent_agent2_bin = [ent_bin_2]
                elif FLAGS.max_exchange > 1:
                    loss_binary_1, ent_bin_1 = multistep_loss_binary(
                        feats_1, probs_1, rewards_1, r[0], binary_agent1_masks, FLAGS.entropy_agent1)
                    loss_binary_2, ent_bin_2 = multistep_loss_binary(
                        feats_2, probs_2, rewards_2, r[1], binary_agent2_masks, FLAGS.entropy_agent2)
                    loss_baseline_1 = multistep_loss_bas(r[0], rewards_1, bas_agent1_masks)
                    loss_baseline_2 = multistep_loss_bas(r[1], rewards_2, bas_agent2_masks)
                    ent_agent1_bin = ent_bin_1
                    ent_agent2_bin = ent_bin_2

            debuglogger.debug(f'Loss bin 1: {loss_binary_1} bin 2: {loss_binary_2}')
            debuglogger.debug(f'Loss baseline 1: {loss_baseline_1} baseline 2: {loss_baseline_2}')
            debuglogger.debug(f'Entropy bin 1: {ent_agent1_bin} Entropy bin 2: {ent_agent1_bin}')

            if FLAGS.use_binary:
                loss_agent1 += FLAGS.rl_loss_weight * loss_binary_1
                loss_agent2 += FLAGS.rl_loss_weight * loss_binary_2
                if not FLAGS.fixed_exchange:
                    # TODO
                    pass
            else:
                loss_baseline_1 = Variable(torch.zeros(1))
                loss_baseline_2 = Variable(torch.zeros(1))

            loss_agent1 += FLAGS.baseline_loss_weight * loss_baseline_1
            loss_agent2 += FLAGS.baseline_loss_weight * loss_baseline_2

            if FLAGS.num_agents == 1 or (agent_idxs[0] == agent_idxs[1]):
                debuglogger.debug(f'Agent 1 and 2 are the same')
                optimizer_agent1.zero_grad()
                # Get gradients from both sides of the exchange
                loss_agent1.backward(retain_graph=True)
                nn.utils.clip_grad_norm(agent1.parameters(), max_norm=1.)
                loss_agent2.backward()
                nn.utils.clip_grad_norm(agent2.parameters(), max_norm=1.)
                # Only need to update agent1
                optimizer_agent1.step()
            else:
                debuglogger.debug(f'Different agents')
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
                log_acc = "Epoch: {} Step: {} Batch: {} Agent 1: {} Agent 2: {} Training Accuracy:\nBefore comms: Both correct: {} At least 1 correct: {}\nAfter comms: Both correct: {} At least 1 correct: {}".format(epoch, step, i_batch, agent_idxs[0], agent_idxs[1], avg_batch_acc_total_nc, avg_batch_acc_atl1_nc, avg_batch_acc_total_com, avg_batch_acc_atl1_com)
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
                    log_ent_agent1_y = "Entropy Agent1 Predictions\n"
                    log_ent_agent1_y += "No comms entropy {}\n Comms entropy\n".format(
                        -ent_1_nc.data[0])
                    for i, ent in enumerate(ent_agent1_y):
                        log_ent_agent1_y += "\n{}. {}".format(i, -ent.data[0])
                    log_ent_agent1_y += "\n"
                    flogger.Log(log_ent_agent1_y)

                if len(ent_agent2_y) > 0:
                    log_ent_agent2_y = "Entropy Agent2 Predictions\n"
                    log_ent_agent2_y += "No comms entropy {}\n Comms entropy\n".format(
                        -ent_2_nc.data[0])
                    for i, ent in enumerate(ent_agent2_y):
                        log_ent_agent2_y += "\n{}. {}".format(i, -ent.data[0])
                    log_ent_agent2_y += "\n"
                    flogger.Log(log_ent_agent2_y)

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
            if step % FLAGS.log_dev == 0:
                # Report in domain development accuracy and checkpoint if best result
                log_agents = "Epoch: {} Step: {} Batch: {} Agent 1: {} Agent 2: {}".format(
                    epoch, step, i_batch, agent_idxs[0], agent_idxs[1])
                flogger.Log(log_agents)
                dev_accuracy_id, total_accuracy_com = get_and_log_dev_performance(
                    agent1, agent2, FLAGS.dataset_indomain_valid_path, True, dev_accuracy_id, logger, flogger, "In Domain", epoch, step, i_batch, store_examples=False, analyze_messages=False, save_messages=False, agent_tag="no_tag")

                if step >= FLAGS.save_after and total_accuracy_com > best_dev_acc:
                    best_dev_acc = total_accuracy_com
                    flogger.Log(
                        "Checkpointing with best In Domain Development Accuracy (both right after comms): {}".format(best_dev_acc))
                    # Optionally store additional information
                    data = dict(step=step, best_dev_acc=best_dev_acc)
                    torch_save(FLAGS.checkpoint + "_best", data, models_dict,
                               optimizers_dict, gpu=0 if FLAGS.cuda else -1)

                    # Re-run in domain dev performance and log examples and analyze messages for a number of pairs of agents
                    # Time consuming to run, only run if dev_acc high enough
                    if best_dev_acc > 0.75 and FLAGS.agent_pools:
                        for i in range(FLAGS.num_agents - 1):
                            flogger.Log("Agent 1: {}".format(i + 1))
                            logger.log(key="Agent 1: ", val=i + 1, step=step)
                            _agent1 = models_dict["agent" + str(i + 1)]
                            flogger.Log("Agent 2: {}".format(i + 2))
                            logger.log(key="Agent 2: ", val=i + 2, step=step)
                            _agent2 = models_dict["agent" + str(i + 2)]
                            if i == 0:
                                # Report in domain development accuracy and store examples
                                dev_accuracy_id_pairs[i], total_accuracy_com = get_and_log_dev_performance(
                                    _agent1, _agent2, FLAGS.dataset_indomain_valid_path, True, dev_accuracy_id_pairs[i], logger, flogger, f'In Domain: Agents {i + 1},{i + 2}', epoch, step, i_batch, store_examples=True, analyze_messages=False, save_messages=False, agent_tag=f'A_{i + 1}_{i + 2}')
                            else:
                                # Report in domain development accuracy
                                dev_accuracy_id_pairs[i], total_accuracy_com = get_and_log_dev_performance(
                                    agent1, agent2, FLAGS.dataset_indomain_valid_path, True, dev_accuracy_id_pairs[i], logger, flogger, f'In Domain: Agents {i + 1},{i + 2}', epoch, step, i_batch, store_examples=False, analyze_messages=False, save_messages=False, agent_tag=f'A_{i + 1}_{i + 2}')

                # Report out of domain development accuracy
                dev_accuracy_ood, total_accuracy_com = get_and_log_dev_performance(
                    agent1, agent2, FLAGS.dataset_path, False, dev_accuracy_ood, logger, flogger, f'Out of Domain:', epoch, step, i_batch, store_examples=False, analyze_messages=False, save_messages=False, agent_tag="no_tag")

            # Report in domain development accuracy when training agent communities
            if FLAGS.agent_communities and step % FLAGS.log_community_eval == 0:
                eval_community(eval_agent_list, models_dict, dev_accuracy_id_pairs[0], logger, flogger, epoch, step, i_batch, store_examples=False, analyze_messages=False, save_messages=False, agent_tag="no_tag")

                # Log how much the language has changed by measuring the performance of one agent from each pool playing a frozen version of itself.
                offset = 0
                for _, p in enumerate(num_agents_per_community):
                    idx = offset + p - 1  # Select last agent from each community to play with itself
                    key = "agent" + str(idx + 1)
                    dev_accuracy_id, total_accuracy_com = get_and_log_dev_performance(
                        models_dict[key], frozen_agents[key], FLAGS.dataset_indomain_valid_path, True, dev_accuracy_id, logger, flogger, f'In Domain, Pool {_ + 1}, agent {idx + 1} playing with frozen version of itself, ids: {id(models_dict[key])}/{id(frozen_agents[key])}', epoch, step, i_batch, store_examples=False, analyze_messages=False, save_messages=False, agent_tag="no_tag")
                    offset += p

            # Report in domain development accuracy when agents communicate with themselves
            if (not FLAGS.agent_communities) and step % FLAGS.log_self_com == 0:
                for i in range(FLAGS.num_agents):
                    agent1 = models_dict["agent" + str(i + 1)]
                    agent2 = copy.deepcopy(agent1)
                    flogger.Log("Agent {} self communication: id {}".format(i + 1, id(agent)))
                    dev_accuracy_self_com[i], total_accuracy_com = get_and_log_dev_performance(
                        agent1, agent2, FLAGS.dataset_indomain_valid_path, True, dev_accuracy_self_com[i], logger, flogger, "Agent " + str(i + 1) + " self communication: In Domain", epoch, step, i_batch, store_examples=False, analyze_messages=False, save_messages=False, agent_tag=f'self_com_A_{i + 1}')

            # Save model periodically
            if step >= FLAGS.save_after and step % FLAGS.save_interval == 0:
                flogger.Log("Checkpointing.")
                # Optionally store additional information
                data = dict(step=step, best_dev_acc=best_dev_acc)
                torch_save(FLAGS.checkpoint, data, models_dict,
                           optimizers_dict, gpu=0 if FLAGS.cuda else -1)

            # Increment batch step
            step += 1

        # Increment epoch
        epoch += 1

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
    FLAGS.attn_extra_context = False
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
    gflags.DEFINE_string("debug_log_level", 'INFO', "")

    # Convenience settings
    gflags.DEFINE_integer("save_after", 1000,
                          "Min step (num batches) after which to save")
    gflags.DEFINE_integer(
        "save_interval", 1000, "How often to save after min batches have been reached")
    gflags.DEFINE_string("checkpoint", None, "Path to save data")
    gflags.DEFINE_string("conf_mat", None, "Path to save confusion matrix")
    gflags.DEFINE_string("log_path", "./logs", "Path to save logs")
    gflags.DEFINE_string("log_file", None, "")
    gflags.DEFINE_string("id_eval_csv_file", None, "Path to in domain eval log file")
    gflags.DEFINE_string("ood_eval_csv_file", None, "Path to out of domain  eval log file")
    gflags.DEFINE_string(
        "json_file", None, "Where to store all flags for an experiment")
    gflags.DEFINE_string("log_load", None, "")
    gflags.DEFINE_boolean("eval_only", False, "")
    gflags.DEFINE_boolean("eval_xproduct", False, "Whether to evaluate the full cross product of agent pairs")

    # Extract Settings
    gflags.DEFINE_boolean("binary_only", False,
                          "Only extract binary data (no training)")
    gflags.DEFINE_string("binary_output", None, "Where to store binary data")

    # Performance settings
    gflags.DEFINE_boolean("cuda", False, "")

    # Display settings
    gflags.DEFINE_string("env", "main", "")
    gflags.DEFINE_boolean("visdom", False, "")
    gflags.DEFINE_string("experiment_name", None, "")
    gflags.DEFINE_integer("log_interval", 50, "")
    gflags.DEFINE_integer("log_dev", 1000, "")
    gflags.DEFINE_integer("log_self_com", 10000, "")
    gflags.DEFINE_integer("log_community_eval", 10000, "")

    # Data settings
    gflags.DEFINE_integer("wv_dim", 100, "Dimension of the word vectors")
    gflags.DEFINE_string("dataset", "shapeworld",
                         "What type of dataset to use")
    gflags.DEFINE_string(
        "dataset_path", "./Shapeworld/data/oneshape_simple_textselect", "Root directory of the dataset")
    gflags.DEFINE_string(
        "dataset_indomain_valid_path", "./Shapeworld/data/oneshape_valid/oneshape_simple_textselect", "Root directory of the in domain validation dataset")
    gflags.DEFINE_string("dataset_mode", "train", "")
    gflags.DEFINE_enum("dataset_eval_mode", "validation",
                       ["validation", "test"], "")
    gflags.DEFINE_string("dataset_type", "agreement", "Task type")
    gflags.DEFINE_string("dataset_name", "oneshape_simple_textselect",
                         "Name of dataset (should correspond to the root directory name automatically generated using ShapeWorld generate.py)")
    gflags.DEFINE_integer("dataset_size_train", 100,
                          "How many examples to use")
    gflags.DEFINE_integer("dataset_size_dev", 100, "How many examples to use")
    gflags.DEFINE_string(
        "glove_path", "./glove.6B/glove.6B.100d.txt", "")
    gflags.DEFINE_boolean("shuffle_train", True, "")
    gflags.DEFINE_boolean("shuffle_dev", False, "")
    gflags.DEFINE_integer("random_seed", 7, "")
    gflags.DEFINE_enum(
        "resnet", "34", ["18", "34", "50", "101", "152"], "Specify Resnet variant.")
    gflags.DEFINE_boolean("improc_from_scratch", False, "Whether to train the image processor from scratch")
    gflags.DEFINE_integer("image_size", 128, "Width and height in pixels of the images to give to the agents")
    gflags.DEFINE_boolean("vertical_mask", False, "Whether to just use a vertical mask on images. Otherwise the mask is random")

    # Model settings
    gflags.DEFINE_enum("model_type", None, [
                       "Fixed", "Adaptive", "FixedAttention", "AdaptiveAttention"], "Preset model configurations.")
    gflags.DEFINE_enum("img_feat", "avgpool_512", [
                       "layer4_2", "avgpool_512", "fc"], "Specify which layer output to use as image")
    gflags.DEFINE_enum("data_context", "fc", [
                       "fc"], "Specify which layer output to use as context for attention")
    gflags.DEFINE_integer("img_feat_dim", 512,
                          "Dimension of the image features")
    gflags.DEFINE_integer(
        "h_dim", 100, "Hidden dimension for all hidden representations in the network")
    gflags.DEFINE_integer("m_dim", 64, "Dimension of the messages")
    gflags.DEFINE_integer(
        "desc_dim", 100, "Dimension of the input description vectors")
    gflags.DEFINE_integer(
        "num_classes", 10, "How many texts the agents have to choose from")
    gflags.DEFINE_integer("s_dim", 1, "Stop probability output dim")
    gflags.DEFINE_boolean("use_binary", True,
                          "Encoding whether agents uses binary features")
    gflags.DEFINE_boolean("no_comms_channel", False,
                          "Whether to mute the communications channel")
    gflags.DEFINE_boolean("test_compositionality", False,
                          "Whether to test compositionality by changing trained agent messages, eval only option")
    gflags.DEFINE_list("agent_dicts", ['None', 'None'], "list of paths to average message code dictionaries")
    gflags.DEFINE_boolean("randomize_comms", False,
                          "Whether to randomize the order in which agents communicate")
    gflags.DEFINE_boolean("cooperative_reward", False,
                          "Whether to have a cooperative or individual reward structure")
    gflags.DEFINE_boolean("agent_pools", False,
                          "Whether to have a pool of agents to train instead of two fixed agents")
    gflags.DEFINE_integer("num_agents", 2, "How many agents total (single pool or community)")
    gflags.DEFINE_boolean("with_replacement", False, "Whether to sample agents from pool with replacement")
    gflags.DEFINE_boolean("agent_communities", False,
                          "Whether to have a community of agents to train instead of two fixed agents")
    gflags.DEFINE_integer("num_communities", 2, "How many communities of agents")
    gflags.DEFINE_string("community_type", "dense", "Type of agent community: dense or hub_spoke")
    gflags.DEFINE_list("community_checkpoints", ['None', 'None'], "list of checkpoints per community")
    gflags.DEFINE_list("num_agents_per_community", [5, 5], "Number of agents per community. Specify one per community, can be different")
    gflags.DEFINE_list("intra_pool_connect_p", [1.0, 1.0], "Percentage of intra pool connectivity. Specify one value per community, can be different")
    gflags.DEFINE_float("inter_pool_connect_p", 0.1, "Percentage of inter pool connectvity. Specify one value")
    gflags.DEFINE_integer("ratio_adjust", 1, "Whether to adjust the inter:intra training ratio")
    gflags.DEFINE_float("intra_inter_ratio", 1.0, "Ratio of intra to inter pool training. Default is 1.0. Agents across pools and within pools will be trained equally.")
    gflags.DEFINE_float("first_msg", 0, "Value to fill the first message with")
    gflags.DEFINE_boolean("visual_attn", False, "agents attends over image")
    gflags.DEFINE_boolean(
        "use_MLP", False, "use MLP to generate prediction scores")
    gflags.DEFINE_integer("attn_dim", 256, "")
    gflags.DEFINE_boolean("attn_extra_context", False, "")
    gflags.DEFINE_integer("attn_context_dim", 4096, "")
    gflags.DEFINE_float("dropout", 0.3, "How much dropout to apply when training using the original image")
    gflags.DEFINE_integer("top_k_dev", 3, "Top-k error in development")
    gflags.DEFINE_integer("top_k_train", 3, "Top-k error in training")

    # Optimization settings
    gflags.DEFINE_enum("optim_type", "RMSprop", ["Adam", "SGD", "RMSprop"], "")
    gflags.DEFINE_integer("batch_size", 32, "Minibatch size for train set.")
    gflags.DEFINE_integer("batch_size_dev", 50, "Minibatch size for dev set.")
    gflags.DEFINE_float("learning_rate", 1e-4, "Used in optimizer.")
    gflags.DEFINE_integer("max_epoch", 500, "")
    gflags.DEFINE_float("entropy_s", None, "")
    gflags.DEFINE_float("entropy_agent1", None, "")
    gflags.DEFINE_float("entropy_agent2", None, "")
    gflags.DEFINE_float("nll_loss_weight_nc", 1.0, "")
    gflags.DEFINE_float("nll_loss_weight_com", 1.0, "")
    gflags.DEFINE_float("rl_loss_weight", 1.0, "")
    gflags.DEFINE_float("baseline_loss_weight", 1.0, "")

    # Conversation settings
    gflags.DEFINE_integer("max_exchange", 1, "")
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

    if not FLAGS.use_binary:
        FLAGS.exchange_samples = 0

    if not FLAGS.experiment_name:
        timestamp = str(int(time.time()))
        FLAGS.experiment_name = "{}-so_{}-wv_{}-bs_{}-{}".format(
            FLAGS.dataset,
            FLAGS.m_dim,
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

    if not FLAGS.id_eval_csv_file:
        FLAGS.id_eval_csv_file = os.path.join(
            FLAGS.log_path, FLAGS.experiment_name + ".id_eval.csv")

    if not FLAGS.ood_eval_csv_file:
        FLAGS.ood_eval_csv_file = os.path.join(
            FLAGS.log_path, FLAGS.experiment_name + ".ood_eval.csv")

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

    FORMAT = '[%(asctime)s %(levelname)s] %(message)s'
    logging.basicConfig(format=FORMAT)
    debuglogger = logging.getLogger('main_logger')
    debuglogger.setLevel(FLAGS.debug_log_level)
    if FLAGS.random_seed != -1:
        random.seed(FLAGS.random_seed)
        np.random.seed(FLAGS.random_seed)
        torch.manual_seed(FLAGS.random_seed)
    else:
        random.seed()
        np.random.seed()
    run()
