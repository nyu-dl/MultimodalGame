import os
import sys
import json
import time
import numpy as np
import random
import h5py
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


def calc_message_mean_and_std(m_store):
    # TODO comments and check
    for k in m_store:
        msgs = m_store[k]["message"]
        msgs = torch.stack(msgs, dim=0)
        debuglogger.debug(
            f'Key: {k}, Count: {m_store[k]["count"]}, Messages: {msgs.size()}')
        mean = torch.mean(msgs, dim=0).cpu()
        std = torch.std(msgs, dim=0).cpu()
        m_store[k]["mean"] = mean
        m_store[k]["std"] = std
    return m_store


def log_message_stats(message_stats, logger, flogger, data_type, epoch, step, i_batch):
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
        # debuglogger.debug(f'Means: {means}')
        # debuglogger.debug(f'Std: {stds}')
        # debuglogger.debug(f'Distances: {dists}')
        logger.log(key=data_type + ": " + s + " message stats: count: ",
                   val=num, step=step)
        for i in range(len(means)):
            logger.log(key=data_type + ": " + s + " message stats: Agent " + str(i) + ": mean: ",
                       val=means[i], step=step)
            logger.log(key=data_type + ": " + s + " message stats: Agent " + str(i) + ": std: ",
                       val=stds[i], step=step)
            flogger.Log("Epoch: {} Step: {} Batch: {} {} message stats: shape {}: count: {}, agent {}: mean: {}, std: {}".format(
                epoch, step, i_batch, data_type, s, num, i, means[i], stds[i]))
        for i in range(len(dists)):
            logger.log(key=data_type + ": " + s + " message stats: distances: [" + str(
                dists[i][0]) + ":" + str(dists[i][1]) + "]: ", val=dists[i][2], step=step)
        flogger.Log("Epoch: {} Step: {} Batch: {} {} message stats: shape {}: dists: {}".format(
            epoch, step, i_batch, data_type, s, dists))

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
        logger.log(key=data_type + ": " + s + " message stats: count: ",
                   val=num, step=step)
        for i in range(len(means)):
            logger.log(key=data_type + ": " + s + " message stats: Agent " + str(i) + ": mean: ",
                       val=means[i], step=step)
            logger.log(key=data_type + ": " + s + " message stats: Agent " + str(i) + ": std: ",
                       val=stds[i], step=step)
            flogger.Log("Epoch: {} Step: {} Batch: {} {} message stats: color {}: count: {}, agent {}: mean: {}, std: {}".format(
                epoch, step, i_batch, data_type, s, num, i, means[i], stds[i]))
        for i in range(len(dists)):
            logger.log(key=data_type + ": " + s + " message stats: distances: [" + str(
                dists[i][0]) + ":" + str(dists[i][1]) + "]: ", val=dists[i][2], step=step)
        flogger.Log("Epoch: {} Step: {} Batch: {} {} message stats: color {}: dists: {}".format(
            epoch, step, i_batch, data_type, s, dists))

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
        logger.log(key=data_type + ": " + s + " message stats: count: ",
                   val=num, step=step)
        for i in range(len(means)):
            logger.log(key=data_type + ": " + s + " message stats: Agent " + str(i) + ": mean: ",
                       val=means[i], step=step)
            logger.log(key=data_type + ": " + s + " message stats: Agent " + str(i) + ": std: ",
                       val=stds[i], step=step)
            flogger.Log("Epoch: {} Step: {} Batch: {} {} message stats: shape_color {}: count: {}, agent {}: mean: {}, std: {}".format(
                epoch, step, i_batch, data_type, s, num, i, means[i], stds[i]))
        for i in range(len(dists)):
            logger.log(key=data_type + ": " + s + " message stats: distances: [" + str(
                dists[i][0]) + ":" + str(dists[i][1]) + "]: ", val=dists[i][2], step=step)
        flogger.Log("Epoch: {} Step: {} Batch: {} {} message stats: shape_color {}: dists: {}".format(
            epoch, step, i_batch, data_type, s, dists))
    path = FLAGS.log_path + "/" + FLAGS.experiment_name + \
        "_" + data_type + "_message_stats.pkl"
    pickle.dump(message_stats, open(path, "wb"))
    debuglogger.info(f'Saved message stats to log file')


def run_analyze_messages(data, data_type, logger, flogger, epoch, step, i_batch):
    '''Logs the mean and std deviation per set of messages per shape, per color and per shape-color for each message set.
      Additionally logs the distances between the mean message for each agent type per shape, color and shape-color

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
                # sys.exit()
            j += 1
        # Calculate and log mean and std_dev
        s_store = calc_message_mean_and_std(s_store)
        c_store = calc_message_mean_and_std(c_store)
        s_c_store = calc_message_mean_and_std(s_c_store)
    log_message_stats(message_stats, logger, flogger,
                      data_type, epoch, step, i_batch)


def add_data_point(batch, i, data_store, messages_1, messages_2):
    '''Adds the relevant data from a batch to a data store to analyze later'''
    data_store["masked_im_1"].append(batch["masked_im_1"][i])
    data_store["masked_im_2"].append(batch["masked_im_2"][i])
    data_store["p"].append(batch["p"][i])
    data_store["target"].append(batch["target"][i])
    data_store["caption"].append(batch["caption_str"][i])
    data_store["shapes"].append(batch["shapes"][i])
    data_store["colors"].append(batch["colors"][i])
    data_store["texts"].append(batch["texts_str"][i])
    # Add messages from each exchange
    m_1 = []
    for exchange in messages_1:
        # debuglogger.debug(f'Exchange agent 1: {exchange[i]}')
        m_1.append(exchange[i])
    data_store["msg_1"].append(m_1)
    m_2 = []
    for exchange in messages_2:
        # debuglogger.debug(f'Exchange agent 2: {exchange[i]}')
        m_2.append(exchange[i])
    data_store["msg_2"].append(m_2)
    # debuglogger.debug(f'Data store: {data_store}')
    return data_store


def eval_dev(dataset_path, top_k, agent1, agent2, logger, flogger, epoch, step, i_batch, in_domain_eval=True, callback=None, store_examples=False, analyze_messages=True):
    """
    Function computing development accuracy
    """

    extra = dict()
    correct_to_analyze = {"masked_im_1": [],
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
    incorrect_to_analyze = {"masked_im_1": [],
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

    # Load development images
    if in_domain_eval:
        eval_mode = "train"
        debuglogger.info("Evaluating on in domain validation set")
    else:
        eval_mode = FLAGS.dataset_eval_mode
        debuglogger.info("Evaluating on out of domain validation set")
    dev_loader = load_shapeworld_dataset(dataset_path, FLAGS.glove_path, eval_mode, FLAGS.dataset_size_dev, FLAGS.dataset_type, FLAGS.dataset_name,
                                         FLAGS.batch_size_dev, FLAGS.random_seed, FLAGS.shuffle_dev, FLAGS.img_feat, FLAGS.cuda, truncate_final_batch=False)

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
            # outp_1, ent_y1 = get_outp(y[0], y1_masks)
            # outp_2, ent_y2 = get_outp(y[1], y2_masks)
            pass

        # Obtain predictions, loss and stats agent 1
        # Before communication predictions
        (dist_1_nc, maxdist_1_nc, argmax_1_nc, ent_1_nc, nll_loss_1_nc,
         logs_1_nc) = get_classification_loss_and_stats(y_nc[0], target)
        # After communication predictions
        (dist_2_nc, maxdist_2_nc, argmax_2_nc, ent_2_nc, nll_loss_2_nc,
         logs_2_nc) = get_classification_loss_and_stats(y_nc[1], target)
        # Obtain predictions, loss and stats agent 1
        # Before communication predictions
        (dist_1, maxdist_1, argmax_1, ent_1, nll_loss_1_com,
         logs_1) = get_classification_loss_and_stats(outp_1, target)
        # After communication predictions
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

        debuglogger.debug(f'eval batch correct com: {batch_correct_com}')
        debuglogger.debug(f'eval batch correct nc: {batch_correct_nc}')
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
        a1_10 = (a2_idx & ((correct_1_nc.float() +
                            (1 - correct_1.float()) == 2))).sum()
        a1_01 = (
            a2_idx & (((1 - correct_1_nc.float()) + correct_1.float()) == 2)).sum()
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
        a2_10 = (a1_idx & ((correct_2_nc.float() +
                            (1 - correct_2.float()) == 2))).sum()
        a2_01 = (
            a1_idx & (((1 - correct_2_nc.float()) + correct_2.float()) == 2)).sum()
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
            # Store batch data to analyze
            if correct_indices_com[_i]:
                correct_to_analyze = add_data_point(
                    batch, _i, correct_to_analyze, feats_1, feats_2)
            else:
                incorrect_to_analyze = add_data_point(
                    batch, _i, incorrect_to_analyze, feats_1, feats_2)

        # debuglogger.debug(f'shapes dict: {shapes_accuracy}')
        # debuglogger.debug(f'colors dict: {colors_accuracy}')

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
        # break

    if store_examples:
        store_exemplar_batch(correct_to_analyze, "correct", logger, flogger)
        store_exemplar_batch(incorrect_to_analyze,
                             "incorrect", logger, flogger)
    if analyze_messages:
        run_analyze_messages(correct_to_analyze, "correct",
                             logger, flogger, epoch, step, i_batch)
        # run_analyze_messages(incorrect_to_analyze, "incorrect", logger, flogger, epoch, step, i_batch)

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

    debuglogger.debug(f'Eval total size: {total}')
    total_accuracy_nc = total_correct_nc / total
    total_accuracy_com = total_correct_com / total
    atleast1_accuracy_nc = atleast1_correct_nc / total
    atleast1_accuracy_com = atleast1_correct_com / total

    # Return accuracy
    return total_accuracy_nc, total_accuracy_com, atleast1_accuracy_nc, atleast1_accuracy_com, extra


def get_and_log_dev_performance(agent1, agent2, dataset_path, in_domain_eval, dev_accuracy_log, logger, flogger, domain, epoch, step, i_batch, store_examples, analyze_messages):
    total_accuracy_nc, total_accuracy_com, atleast1_accuracy_nc, atleast1_accuracy_com, extra = eval_dev(
        dataset_path, FLAGS.top_k_dev, agent1, agent2, logger, flogger, epoch, step, i_batch, in_domain_eval=in_domain_eval, callback=None, store_examples=store_examples, analyze_messages=analyze_messages)
    dev_accuracy_log['total_acc_both_nc'].append(total_accuracy_nc)
    dev_accuracy_log['total_acc_both_com'].append(total_accuracy_com)
    dev_accuracy_log['total_acc_atl1_nc'].append(atleast1_accuracy_nc)
    dev_accuracy_log['total_acc_atl1_com'].append(atleast1_accuracy_com)
    logger.log(key=domain + " Development Accuracy, both right, no comms",
               val=dev_accuracy_log['total_acc_both_nc'][-1], step=step)
    logger.log(key=domain + "Development Accuracy, both right, after comms",
               val=dev_accuracy_log['total_acc_both_com'][-1], step=step)
    logger.log(key=domain + "Development Accuracy, at least 1 right, no comms",
               val=dev_accuracy_log['total_acc_atl1_nc'][-1], step=step)
    logger.log(key=domain + "Development Accuracy, at least 1 right, after comms",
               val=dev_accuracy_log['total_acc_atl1_com'][-1], step=step)
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
            logger.log(key=domain + " Development Accuracy: " + k + " ",
                       val=extra['shapes_accuracy'][k]['correct'] / extra['shapes_accuracy'][k]['total'], step=step)
    for k in extra['colors_accuracy']:
        if extra['colors_accuracy'][k]['total'] > 0:
            logger.log(key=domain + " Development Accuracy: " + k + " ",
                       val=extra['colors_accuracy'][k]['correct'] / extra['colors_accuracy'][k]['total'], step=step)

    flogger.Log("Epoch: {} Step: {} Batch: {} {} Development Accuracy, both right, no comms: {}".format(
        epoch, step, i_batch, domain, dev_accuracy_log['total_acc_both_nc'][-1]))
    flogger.Log("Epoch: {} Step: {} Batch: {} {} Development Accuracy, both right, after comms: {}".format(
        epoch, step, i_batch, domain, dev_accuracy_log['total_acc_both_com'][-1]))
    flogger.Log("Epoch: {} Step: {} Batch: {} {} Development Accuracy, at least right, no comms: {}".format(
        epoch, step, i_batch, domain, dev_accuracy_log['total_acc_atl1_nc'][-1]))
    flogger.Log("Epoch: {} Step: {} Batch: {} {} Development Accuracy, at least 1 right, after comms: {}".format(
        epoch, step, i_batch, domain, dev_accuracy_log['total_acc_atl1_com'][-1]))

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
            extra["agent1_performance"]["total"] /
            extra["agent1_performance"]["total"],
            extra["agent1_performance"]["11"] /
            extra["agent1_performance"]["total"],
            extra["agent1_performance"]["01"] /
            extra["agent1_performance"]["total"],
            extra["agent1_performance"]["00"] /
            extra["agent1_performance"]["total"],
            extra["agent1_performance"]["10"] / extra["agent1_performance"]["total"]))

    flogger.Log('Agent 2: total {}, 11: {}, 01: {} 00: {}, 10: {}'.format(
        extra["agent2_performance"]["total"],
        extra["agent2_performance"]["11"],
        extra["agent2_performance"]["01"],
        extra["agent2_performance"]["00"],
        extra["agent2_performance"]["10"]))
    if extra["agent2_performance"]["total"] > 0:
        flogger.Log('Agent 2: total {}, 11: {}, 01: {} 00: {}, 10: {}'.format(
            extra["agent2_performance"]["total"] /
            extra["agent2_performance"]["total"],
            extra["agent2_performance"]["11"] /
            extra["agent2_performance"]["total"],
            extra["agent2_performance"]["01"] /
            extra["agent2_performance"]["total"],
            extra["agent2_performance"]["00"] /
            extra["agent2_performance"]["total"],
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

    return dev_accuracy_log, total_accuracy_com
