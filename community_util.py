import os
import sys
import numpy as np
import random
import functools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Variable

FORMAT = '[%(asctime)s %(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT)
debuglogger = logging.getLogger('main_logger')
debuglogger.setLevel('DEBUG')


def sample_agents(train_probs, agent_idx_list):
    ''' Takes in a probability distribution over agent pairs and a mapping from from flattened index to agent pairs, and returns a pair of agent indexes'''
    idx = np.random.choice(list(range(len(agent_idx_list))), p=train_probs)
    (agent1, agent2) = agent_idx_list[idx]
    return (agent1, agent2)


def build_train_matrix(pools_num, community_type, intra_pool_connect_p, inter_pool_connect_p, intra_inter_ratio=1.0):
    '''Builds a connectivity matrix over multiple pools of agents. Returns a probability distribution over all possible agent connections in the population (across multiple pools) and a mapping from ints to agent pairs

    Params:
    pools_num: list containing the number of agents in each pool
    community_type: community structure - "hub_spoke" or "dense"
    intra_pool_connect_p: list containing percentages of agents in a pool that should be connected with other agents in the same pool
    inter_pool_connect_p: percentage of agents in a pool that should be connected with agents in the other pools
    inter_intra_ratio: ratio of within pool to between pool training samples. Default is 1.0

    Returns:
    train_vec_prob: probability distribution over all possible agent pairs. Size = total_agents ^2
    agent_idx_list: corresponding agent pairs.
        e.g if train_vec_prob[1] = 0.1 and agent_idx_list[1] = (0, 1) then agents 0 and 1 will be selected to train 10% of the time
    '''
    total_agents = sum(pools_num)
    agent_idx_list = []
    intra_train_matrix = np.zeros((total_agents, total_agents))
    inter_train_matrix = np.zeros((total_agents, total_agents))
    train_matrix = np.zeros((total_agents, total_agents))
    if community_type == "hub_spoke":
        debuglogger.warn(f'Hub and spoke model not implemented yet, select "dense"')
        sys.exit()
    elif community_type == "dense":
        # fill intra_pool connectivity
        offset = 0
        for p, prob in zip(pools_num, intra_pool_connect_p):
            for i in range(p):
                for j in range(p):
                    # Agents cannot be sampled with themselves
                    if i != j:
                        i_0 = i + offset
                        j_0 = j + offset
                        if np.random.rand() < prob:
                            intra_train_matrix[i_0][j_0] = 1
            offset += p
        debuglogger.info(f'Intra pool connect: {torch.from_numpy(intra_train_matrix)}')
        total_intrapool = np.sum(intra_train_matrix)
        debuglogger.info(f'Total intrapool: {total_intrapool}')
        # fill inter_pool connectivity
        prev_p = 0
        curr_p = 0
        for p in pools_num:
            curr_p += p
            debuglogger.debug(f'prev p: {prev_p}, current p: {curr_p}')
            for i in range(prev_p, curr_p):
                for j in range(prev_p):
                    if np.random.rand() < inter_pool_connect_p:
                        inter_train_matrix[i][j] = 1
                for j in range(curr_p, total_agents):
                    if np.random.rand() < inter_pool_connect_p:
                        inter_train_matrix[i][j] = 1
            prev_p = curr_p
        debuglogger.info(f'Inter pool connect: {torch.from_numpy(inter_train_matrix)}')
        total_interpool = np.sum(inter_train_matrix)
        debuglogger.info(f'Total interpool: {total_interpool}')
        if total_interpool == 0:
            debuglogger.warn(f'No inter pool connections, exiting (comment out this line in community_util.py if this is what you are after)...')
            sys.exit()
        # Adjust to correct training ratio between intra and inter pool
        cur_ratio = total_intrapool / total_interpool
        if total_interpool > 0 and total_intrapool > 0:
            inter_train_matrix *= cur_ratio
            inter_train_matrix /= intra_inter_ratio
            cur_ratio = total_intrapool / np.sum(inter_train_matrix)
        debuglogger.info(f'Desired inter-intra sampling ratio: {intra_inter_ratio}, current: {cur_ratio}')
        debuglogger.info(f'Adjusted inter pool connect: {torch.from_numpy(inter_train_matrix)}')
        # Combine the matrices
        train_matrix = intra_train_matrix + inter_train_matrix
        debuglogger.info(f'Agent connectivity: {torch.from_numpy(train_matrix)}')
        # Convert to probability distribution
        train_vec = np.reshape(train_matrix, total_agents * total_agents)
        total_connections = np.sum(train_vec)
        debuglogger.info(f'Total connections: {total_connections}')
        train_vec_prob = train_vec / total_connections
        assert np.abs(np.sum(train_vec_prob) - 1.0) < 0.00001
        debuglogger.info(f'Train vec: \n{train_vec}\ntrain prob: \n{train_vec_prob}')
        # Build agent list
        for i in range(total_agents):
            for j in range(total_agents):
                agent_idx_list.append((i, j))
        debuglogger.info(f'Agent idx list: {agent_idx_list}')
    else:
        debuglogger.warn(f'Invalid community type, please select "dense" or "hub_spoke"')
        sys.exit()
    return (train_vec_prob, agent_idx_list)


def build_eval_list(pools_num, community_type, train_vec_prob):
    '''Builds a list of list of agent pairs to select for evaluation from all possible pairs of agents.
    Considers 6 types of connections.
        1. self communication, agent connected to multiple pools
        2. self communication, agent connected to only one pool
        3. within pool communication, different agents, trained together
        4. within pool communication, different agents, never trained together
        5. cross pool communication, different agents, trained together
        6. cross pool communication, different agents, never trained together
    '''
    total_agents = sum(pools_num)
    eval_connections = np.zeros((total_agents, total_agents))
    train_matrix = np.reshape(train_vec_prob, (total_agents, total_agents))
    total_combinations = 0
    c_1 = []  # self communication, agent connected to multiple pools
    c_2 = []  # self communication, agent connected to only one pool
    c_3 = []  # within pool communication, different agents, trained together
    c_4 = []  # within pool communication, different agents, never trained together
    c_5 = []  # cross pool communication, different agents, trained together
    c_6 = []  # cross pool communication, different agents, never trained together
    offset = 0
    for p in pools_num:
        c_p_3 = []
        c_p_4 = []
        for i in range(p):
            for j in range(p):
                if i != j:
                    i_0 = i + offset
                    j_0 = j + offset
                    if train_matrix[i_0][j_0] > 0:
                        c_p_3.append((i_0, j_0))
                        total_combinations += 1
                    else:
                        c_p_4.append((i_0, j_0))
                        total_combinations += 1
        offset += p
        c_3.append(c_p_3)
        c_4.append(c_p_4)
    prev_p = 0
    curr_p = 0
    for p in pools_num:
        c_p_1 = []
        c_p_2 = []
        c_p_5 = []
        c_p_6 = []
        curr_p += p
        debuglogger.debug(f'prev p: {prev_p}, current p: {curr_p}')
        for i in range(prev_p, curr_p):
            cross_connected = False
            for j in range(prev_p):
                if train_matrix[i][j] > 0:
                    cross_connected = True
                    c_p_5.append((i, j))
                    total_combinations += 1
                else:
                    c_p_6.append((i, j))
                    total_combinations += 1
            for j in range(curr_p, total_agents):
                if train_matrix[i][j] > 0:
                    cross_connected = True
                    c_p_5.append((i, j))
                    total_combinations += 1
                else:
                    c_p_6.append((i, j))
                    total_combinations += 1
            if cross_connected:
                c_p_1.append((i, i))
                total_combinations += 1
            else:
                c_p_2.append((i, i))
                total_combinations += 1
        c_1.append(c_p_1)
        c_2.append(c_p_2)
        c_5.append(c_p_5)
        c_6.append(c_p_6)
        prev_p = curr_p
    debuglogger.info(f'Total combinations: {total_combinations}')
    debuglogger.info(f'Self communication: multiple pools: {sum([len(x) for x in c_1])}, one pool: {sum([len(x) for x in c_2])}')
    debuglogger.info(f'Within pool comms: trained together: {sum([len(x) for x in c_3])}, not trained together: {sum([len(x) for x in c_4])}')
    debuglogger.info(f'Cross pool comms: trained together: {sum([len(x) for x in c_5])}, not trained together: {sum([len(x) for x in c_6])}')
    choices = [c_1, c_2, c_3, c_4, c_5, c_6]
    agent_idxs = []
    for i, c in enumerate(choices):
        debuglogger.debug(f'Type {i + 1}: {c}')
        temp = []
        # To ensure one example from each pool for each type
        for pool in c:
            if len(pool) > 0:
                idx = np.random.choice(list(range(len(pool))))
                temp.append(pool[idx])
        agent_idxs.append(temp)
    debuglogger.info(f'Train matrix: {torch.from_numpy(train_matrix)}')
    debuglogger.info(f'Eval agent combinations: {agent_idxs}')
    return agent_idxs


if __name__ == "__main__":
    pools_num = [3, 3]
    community_type = "dense"
    intra_pool_connect_p = [1.0, 1.0]
    inter_pool_connect_p = 0.1
    intra_inter_ratio = 1.0
    (train_vec_prob, agent_idx_list) = build_train_matrix(pools_num, community_type, intra_pool_connect_p, inter_pool_connect_p, intra_inter_ratio)
    eval_agent_idxs = build_eval_list(pools_num, community_type, train_vec_prob)
    for i in range(20):
        (agent1, agent2) = sample_agents(train_vec_prob, agent_idx_list)
        debuglogger.info(f'Agent1: {agent1}, Agent2: {agent2}')
