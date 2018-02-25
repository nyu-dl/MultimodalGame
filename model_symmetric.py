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
from misc import xavier_normal
from misc import build_mask

from agents import Agent
from dataset_loader import load_shapeworld_dataset
from binary_vectors import extract_binary
from sparks import sparks

import gflags
FLAGS = gflags.FLAGS

SHAPES = ['circle', 'cross', 'ellipse', 'pentagon', 'rectangle', 'semicircle', 'square', 'triangle']
COLORS = ['blue', 'cyan', 'gray', 'green', 'magenta', 'red', 'yellow']


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


def eval_dev(dataset_path, top_k, agent1, agent2, in_domain_eval=True, callback=None):
    """
    Function computing development accuracy
    """

    extra = dict()
    correct_to_save = {"masked_im_1": [],
                       "masked_im_2": [],
                       "p": [],
                       "target": [],
                       "caption": [],
                       "texts": []
                       }
    incorrect_to_save = {"masked_im_1": [],
                         "masked_im_2": [],
                         "p": [],
                         "target": [],
                         "caption": [],
                         "texts": []
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


def get_and_log_dev_performance(agent1, agent2, dataset_path, in_domain_eval, dev_accuracy_log, logger, flogger, domain, epoch, step, i_batch):
    total_accuracy_nc, total_accuracy_com, atleast1_accuracy_nc, atleast1_accuracy_com, extra = eval_dev(
        dataset_path, FLAGS.top_k_dev, agent1, agent2, in_domain_eval, callback=None)
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

    return dev_accuracy_log, total_accuracy_com


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
        s: All STOP bits. (Masks, Values, Probabilities)
        w_1: All agent_1 messages. (Values, Probabilities)
        w_2: All agent_2 messages. (Values, Probabilities)
        y_1: All predictions that were made by agent 1 (Before comms, after comms)
        y_2: All predictions that were made by agent 2 (Before comms, after comms)
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

        # Optionally corrupt agent 1's message
        if corrupt:
            m_1e_binary = corrupt_message(corrupt_region, agent1, m_1e_binary)

        # Run data through agent 2
        if data_context is not None:
            # TODO
            debuglogger.warning(f'Data context not supported currently')
            sys.exit()
        else:
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
            m_2e_binary = corrupt_message(corrupt_region, agent2, m_2e_binary)

        # Run data through agent 1
        if data_context is not None:
            pass
        else:
            s_1e, m_1e, y_1e, r_1e = agent1(
                data['im_feats_1'],
                m_2e_binary,
                i_exchange,
                desc,
                use_message,
                batch_size,
                train)

        s_binary_1, s_prob_1 = s_1e
        s_binary_2, s_prob_2 = s_2e
        m_binary_1, m_probs_1 = m_1e
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
    weight = Variable(rewards) - \
        Variable(baseline_rewards.clone().detach().data)
    # debuglogger.debug(f'Reinforcement weight: {weight.data}')
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
    loss_bas = nn.MSELoss()(baseline_scores, Variable(rewards))
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


def calculate_accuracy(prediction_dist, target, batch_size, top_k):
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
    agent1 = Agent(im_feature_type=FLAGS.img_feat,
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
                   cuda=FLAGS.cuda)

    flogger.Log("Agent 1 Architecture: {}".format(agent1))
    total_params = sum([functools.reduce(lambda x, y: x * y, p.size(), 1.0)
                        for p in agent1.parameters()])
    flogger.Log("Total Parameters: {}".format(total_params))

    agent2 = Agent(im_feature_type=FLAGS.img_feat,
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
                   cuda=FLAGS.cuda)

    flogger.Log("Agent 2 Architecture: {}".format(agent2))
    total_params = sum([functools.reduce(lambda x, y: x * y, p.size(), 1.0)
                        for p in agent2.parameters()])
    flogger.Log("Total Parameters: {}".format(total_params))

    # Optimizer
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

        debuglogger.info("Evaluating on in domain validation set")
        total_accuracy_nc, total_accuracy_com, atleast1_accuracy_nc, atleast1_accuracy_com, extra = eval_dev(
            FLAGS.dataset_indomain_valid_path, FLAGS.top_k_dev, agent1, agent2, in_domain_eval=True, callback=None)
        flogger.Log(
            "In domain dev accuracy no comms, both right: " + str(total_accuracy_nc))
        flogger.Log(
            "In domain dev accuracy no comms, at least 1 right: " + str(atleast1_accuracy_nc))
        flogger.Log(
            "In domain dev accuracy after comms, both right: " + str(total_accuracy_com))
        flogger.Log(
            "In domain dev accuracy after comms, at least 1 right: " + str(atleast1_accuracy_com))
        with open(FLAGS.id_eval_csv_file, 'w') as f:
            f.write(
                "checkpoint,eval_file,topk,step,best_dev_acc,eval_acc_nc_2,eval_acc_nc_1,eval_acc_com_2,eval_acc_com_1,convlen_1_mean,convlen_1_std,convlen_2_mean,convlen_2_std\n")
            f.write("{},{},{},{},{},{},{},{}\n".format(
                FLAGS.checkpoint, FLAGS.dataset_indomain_valid_path, FLAGS.top_k_dev,
                step, best_dev_acc, total_accuracy_nc, atleast1_accuracy_nc, total_accuracy_com, atleast1_accuracy_com,
                extra['conversation_lengths_1_mean'], extra['conversation_lengths_1_std'],
                extra['conversation_lengths_2_mean'], extra['conversation_lengths_2_std']))

        debuglogger.info("Evaluating on out of  domain validation set")
        total_accuracy_nc, total_accuracy_com, atleast1_accuracy_nc, atleast1_accuracy_com, extra = eval_dev(
            FLAGS.dataset_path, FLAGS.top_k_dev, agent1, agent2, in_domain_eval=False, callback=None)
        flogger.Log(
            "Out of domain dev accuracy no comms, both right: " + str(total_accuracy_nc))
        flogger.Log(
            "Out of domain dev accuracy no comms, at least 1 right: " + str(atleast1_accuracy_nc))
        flogger.Log(
            "Out of domain dev accuracy after comms, both right: " + str(total_accuracy_com))
        flogger.Log(
            "Out of domain dev accuracy after comms, at least 1 right: " + str(atleast1_accuracy_com))
        with open(FLAGS.ood_eval_csv_file, 'w') as f:
            f.write(
                "checkpoint,eval_file,topk,step,best_dev_acc,eval_acc_nc_2,eval_acc_nc_1,eval_acc_com_2,eval_acc_com_1,convlen_1_mean,convlen_1_std,convlen_2_mean,convlen_2_std\n")
            f.write("{},{},{},{},{},{},{},{}\n".format(
                FLAGS.checkpoint, FLAGS.dataset_path, FLAGS.top_k_dev,
                step, best_dev_acc, total_accuracy_nc, atleast1_accuracy_nc, total_accuracy_com, atleast1_accuracy_com,
                extra['conversation_lengths_1_mean'], extra['conversation_lengths_1_std'],
                extra['conversation_lengths_2_mean'], extra['conversation_lengths_2_std']))
        sys.exit()
    elif FLAGS.binary_only:
        if not os.path.exists(FLAGS.checkpoint):
            raise Exception("Must provide valid checkpoint.")
        # TODO fix for new agents
        debuglogger.warning(f'Extract binary not updated for new agents yet')
        sys.exit()
        extract_binary(FLAGS, load_hdf5, exchange, FLAGS.dev_file, FLAGS.batch_size_dev, epoch,
                       FLAGS.shuffle_dev, FLAGS.cuda, FLAGS.top_k_dev,
                       sender, receiver, desc_dev_dict, map_labels_dev, FLAGS.experiment_name)
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

        # Iterate through batches
        for i_batch, batch in enumerate(dataloader):
            debuglogger.debug(f'Batch {i_batch}')

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
                        f'Error: multistep fixed exchange not implemented yet')
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
                    # TODO
                    ent_agent1_bin = []
                    ent_agent2_bin = []
                    debuglogger.warning(
                        f'Error: multistep fixed exchange not implemented yet')
                    sys.exit()

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
                log_acc = "Epoch: {} Step: {} Batch: {} Training Accuracy:\nBefore comms: Both correct: {} At least 1 correct: {}\nAfter comms: Both correct: {} At least 1 correct: {}".format(
                    epoch, step, i_batch, avg_batch_acc_total_nc, avg_batch_acc_atl1_nc, avg_batch_acc_total_com, avg_batch_acc_atl1_com)
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

                # Optionally print sampled and inferred binary vectors from
                # most recent exchange.
                if FLAGS.exchange_samples > 0:

                    log_train = log_exchange(
                        s, message_1, message_2, current_exchange, log_type="Train:")
                    flogger.Log(log_train)

                    exchange_args["train"] = False
                    s, message_1, message_2, y_all, r = exchange(
                        agent1, agent2, exchange_args)

                    log_train = log_exchange(
                        s, message_1, message_2, current_exchange, log_type="Eval:")
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
            if step % FLAGS.log_dev == 0:
                # Report in domain development accuracy and checkpoint if best result
                dev_accuracy_id, total_accuracy_com = get_and_log_dev_performance(
                    agent1, agent2, FLAGS.dataset_indomain_valid_path, True, dev_accuracy_id, logger, flogger, "In Domain", epoch, step, i_batch)
                if step >= FLAGS.save_after and total_accuracy_com > best_dev_acc:
                    best_dev_acc = total_accuracy_com
                    flogger.Log(
                        "Checkpointing with best In Domain Development Accuracy (both right after comms): {}".format(best_dev_acc))
                    # Optionally store additional information
                    data = dict(step=step, best_dev_acc=best_dev_acc)
                    torch_save(FLAGS.checkpoint + "_best", data, models_dict,
                               optimizers_dict, gpu=0 if FLAGS.cuda else -1)
                # Report out of domain development accuracy
                dev_accuracy_id, total_accuracy_com = get_and_log_dev_performance(
                    agent1, agent2, FLAGS.dataset_path, False, dev_accuracy_ood, logger, flogger, "Out of Domain", epoch, step, i_batch)

            # Report in domain development accuracy when agents communicate with themselves
            if step % FLAGS.log_self_com == 0:
                dev_accuracy_id, total_accuracy_com = get_and_log_dev_performance(
                    agent1, agent1, FLAGS.dataset_indomain_valid_path, True, dev_accuracy_id, logger, flogger, "Agent 1 self communication: In Domain", epoch, step, i_batch)
                dev_accuracy_id, total_accuracy_com = get_and_log_dev_performance(
                    agent2, agent2, FLAGS.dataset_indomain_valid_path, True, dev_accuracy_id, logger, flogger, "Agent 2 self communication: In Domain", epoch, step, i_batch)

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
        "save_interval", 100, "How often to save after min batches have been reached")
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

    # Extract Settings
    gflags.DEFINE_boolean("binary_only", False,
                          "Only extract binary data (no training)")
    gflags.DEFINE_string("binary_output", None, "Where to store binary data")

    # Performance settings
    gflags.DEFINE_boolean("cuda", False, "")

    # Display settings
    gflags.DEFINE_string("env", "main", "")
    gflags.DEFINE_boolean("visdom", False, "")
    gflags.DEFINE_boolean("use_alpha", False, "")
    gflags.DEFINE_string("experiment_name", None, "")
    gflags.DEFINE_integer("log_interval", 50, "")
    gflags.DEFINE_integer("log_dev", 1000, "")
    gflags.DEFINE_integer("log_self_com", 10000, "")

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
    gflags.DEFINE_boolean("shuffle_dev", True, "")
    gflags.DEFINE_integer("random_seed", 7, "")
    gflags.DEFINE_enum(
        "resnet", "34", ["18", "34", "50", "101", "152"], "Specify Resnet variant.")

    # Model settings
    gflags.DEFINE_enum("model_type", None, [
                       "Fixed", "Adaptive", "FixedAttention", "AdaptiveAttention"], "Preset model configurations.")
    gflags.DEFINE_enum("img_feat", "avgpool_512", [
                       "layer4_2", "avgpool_512", "fc"], "Specify which layer output to use as image")
    gflags.DEFINE_enum("data_context", "fc", [
                       "fc"], "Specify which layer output to use as context for attention")
    # gflags.DEFINE_enum("sender_mix", "sum", ["sum", "prod", "mou"], "")
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
    gflags.DEFINE_boolean("randomize_comms", False,
                          "Whether to randomize the order in which agents communicate")
    gflags.DEFINE_boolean("cooperative_reward", False,
                          "Whether to have a cooperative or individual reward structure")
    # gflags.DEFINE_boolean("ignore_2", False,
    #                       "Agent 1 ignores messages from Agent 2")
    # gflags.DEFINE_boolean("ignore_1", False,
    #                       "Agent 2 ignores messages from Agent 1")
    # gflags.DEFINE_boolean("block_y", True, "Halt gradient flow through description scores")
    gflags.DEFINE_float("first_msg", 0, "Value to fill the first message with")
    # gflags.DEFINE_float("flipout_1", None, "Dropout for bit flipping")
    # gflags.DEFINE_float("flipout_2", None, "Dropout for bit flipping")
    # gflags.DEFINE_boolean("flipout_dev", False, "Dropout for bit flipping")
    # gflags.DEFINE_boolean("s_prob_prod", True, "Simulate sampling during test time")
    gflags.DEFINE_boolean("visual_attn", False, "agents attends over image")
    gflags.DEFINE_boolean(
        "use_MLP", False, "use MLP to generate prediction scores")
    gflags.DEFINE_integer("attn_dim", 256, "")
    gflags.DEFINE_boolean("attn_extra_context", False, "")
    gflags.DEFINE_integer("attn_context_dim", 4096, "")
    gflags.DEFINE_boolean("desc_attn", False, "agents attend over text")
    gflags.DEFINE_integer("desc_attn_dim", 64, "text attention dim")
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
    gflags.DEFINE_integer("exchange_samples", 1, "")
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

    run()
