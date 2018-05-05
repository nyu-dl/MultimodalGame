import argparse
import numpy as np
import torch
import math
import pickle
import pprint
import math
import sys


SHAPES = ['circle', 'cross', 'ellipse', 'pentagon', 'rectangle', 'semicircle', 'square', 'triangle']
COLORS = ['blue', 'cyan', 'gray', 'green', 'magenta', 'red', 'yellow']
OUTOFDOMAIN = [('square', 'red'), ('triangle', 'green'), ('circle', 'blue'), ('rectangle', 'yellow'), ('cross', 'magenta'), ('ellipse', 'cyan')]


def convert_tensor_to_string(message):
    '''Converts binary message stored in a pytorch tensor to a string'''
    assert message.dim() == 1
    m_len = message.size(0)
    m_str = ""
    for i in range(m_len):
        m_str += "0" if message[i] == 0 else "1"
    return m_str


def add_elem(m_dict, message, shape, color, answer_type):
    '''General'''
    m_dict['total'] += 1
    m_dict[answer_type] += 1
    if message not in m_dict:
        m_dict[message] = {'shape': {'count': 0},
                           'color': {'count': 0},
                           'shape_color': {'count': 0},
                           'total': 0,
                           'correct': 0, 'correct_p': 0,
                           'incorrect': 0, 'incorrect_p': 0}
    m_dict[message]['total'] += 1
    m_dict[message][answer_type] += 1

    '''Shapes'''
    shape_dict = m_dict[message]['shape']
    if shape is not None:
        shape_dict['count'] += 1
        if shape not in shape_dict:
            shape_dict[shape] = 1
        else:
            shape_dict[shape] += 1

    '''Colors'''
    color_dict = m_dict[message]['color']
    if color is not None:
        color_dict['count'] += 1
        if color not in color_dict:
            color_dict[color] = 1
        else:
            color_dict[color] += 1

    '''Shapes and colors'''
    shape_color_dict = m_dict[message]['shape_color']
    if shape is not None and color is not None:
        sc = shape + '_' + color
        shape_color_dict['count'] += 1
        if sc not in shape_color_dict:
            shape_color_dict[sc] = 1
        else:
            shape_color_dict[sc] += 1

    return m_dict


def calc_ratios(m_dict):
    for m, stats in m_dict.items():
        if m == 'total' or m == 'correct' or m == 'incorrect':
            pass
        else:
            stats['correct_p'] = stats['correct'] / stats['total']
            stats['incorrect_p'] = stats['incorrect'] / stats['total']
            shape_dict = stats['shape']
            for s in SHAPES:
                if s in shape_dict:
                    shape_dict[s + '_p'] = shape_dict[s] / stats['total']
            color_dict = stats['color']
            for c in COLORS:
                if c in color_dict:
                    color_dict[c + '_p'] = color_dict[c] / stats['total']
            shape_color_dict = stats['shape_color']
            for s in SHAPES:
                for c in COLORS:
                    sc = s + '_' + c
                    if sc in shape_color_dict:
                        shape_color_dict[sc + '_p'] = shape_color_dict[sc] / stats['total']
                        shape_color_dict[sc + '_pratio'] = \
                            shape_color_dict[sc + '_p'] / (stats['shape'][s + '_p'] * stats['color'][c + '_p'])
                        shape_color_dict[sc + '_log_pratio'] = math.log(shape_color_dict[sc + '_pratio'])

    m_dict['correct_p'] = m_dict['correct'] / m_dict['total']
    m_dict['incorrect_p'] = m_dict['incorrect'] / m_dict['total']
    return m_dict


def count_pratios(m_dict):
    counts = []
    for m, stats in m_dict.items():
        if m == 'total' or m == 'correct' or m == 'incorrect' or m == 'correct_p' or m == 'incorrect_p':
            pass
        else:
            pratios = []
            shape_color_dict = stats['shape_color']
            for s in SHAPES:
                for c in COLORS:
                    sc = s + '_' + c
                    if sc in shape_color_dict:
                        pratios.append((shape_color_dict[sc], shape_color_dict[sc + '_log_pratio']))
            if len(pratios) > 0:
                counts.append((m, stats['total'], pratios, len(pratios)))
    counts = sorted(counts, key=lambda x: x[3])
    return counts


def get_pratio_stats(counts):
    mean_log_pratio = {'total': {'count': 0, 'log_pratio': 0},
                       '1': {'count': 0, 'log_pratio': 0},
                       '<=2': {'count': 0, 'log_pratio': 0},
                       '<=3': {'count': 0, 'log_pratio': 0},
                       '<=4': {'count': 0, 'log_pratio': 0},
                       '5+': {'count': 0, 'log_pratio': 0}}
    for _, elem in enumerate(counts):
        for prt in elem[2]:
            if elem[3] == 1:
                mean_log_pratio['1']['log_pratio'] += prt[0] * prt[1]
                mean_log_pratio['1']['count'] += prt[0]
            if elem[3] <= 2:
                mean_log_pratio['<=2']['log_pratio'] += prt[0] * prt[1]
                mean_log_pratio['<=2']['count'] += prt[0]
            if elem[3] <= 3:
                mean_log_pratio['<=3']['log_pratio'] += prt[0] * prt[1]
                mean_log_pratio['<=3']['count'] += prt[0]
            if elem[3] <= 4:
                mean_log_pratio['<=4']['log_pratio'] += prt[0] * prt[1]
                mean_log_pratio['<=4']['count'] += prt[0]
            if elem[3] >= 5:
                mean_log_pratio['5+']['log_pratio'] += prt[0] * prt[1]
                mean_log_pratio['5+']['count'] += prt[0]
            mean_log_pratio['total']['log_pratio'] += prt[0] * prt[1]
            mean_log_pratio['total']['count'] += prt[0]
    '''Normalize'''
    mean_log_pratio['total']['log_pratio'] /= mean_log_pratio['total']['count']
    mean_log_pratio['1']['log_pratio'] /= mean_log_pratio['1']['count']
    mean_log_pratio['<=2']['log_pratio'] /= mean_log_pratio['<=2']['count']
    mean_log_pratio['<=3']['log_pratio'] /= mean_log_pratio['<=3']['count']
    mean_log_pratio['<=4']['log_pratio'] /= mean_log_pratio['<=4']['count']
    mean_log_pratio['5+']['log_pratio'] /= mean_log_pratio['5+']['count']
    mean_log_pratio['total']['pratio'] = math.exp(mean_log_pratio['total']['log_pratio'])
    mean_log_pratio['1']['pratio'] = math.exp(mean_log_pratio['1']['log_pratio'])
    mean_log_pratio['<=2']['pratio'] = math.exp(mean_log_pratio['<=2']['log_pratio'])
    mean_log_pratio['<=3']['pratio'] = math.exp(mean_log_pratio['<=3']['log_pratio'])
    mean_log_pratio['<=4']['pratio'] = math.exp(mean_log_pratio['<=4']['log_pratio'])
    mean_log_pratio['5+']['pratio'] = math.exp(mean_log_pratio['5+']['log_pratio'])
    pprint.pprint(mean_log_pratio)
    return mean_log_pratio


def build_message_dict(data, agents="both"):
    m_dict = {'total': 0, 'correct': 0, 'incorrect': 0}
    for _, d in enumerate(data):
        if agents == "both":
            for msg, m_prob in zip(d["msg_1"], d["probs_1"]):
                answer_type = 'correct' if d["correct"] else 'incorrect'
                m_dict = add_elem(m_dict, convert_tensor_to_string(msg), d["shape"], d["color"], answer_type)
            for msg, m_prob in zip(d["msg_2"], d["probs_2"]):
                answer_type = 'correct' if d["correct"] else 'incorrect'
                m_dict = add_elem(m_dict, convert_tensor_to_string(msg), d["shape"], d["color"], answer_type)
        elif agents == "one":
            for msg, m_prob in zip(d["msg_1"], d["probs_1"]):
                answer_type = 'correct' if d["correct"] else 'incorrect'
                m_dict = add_elem(m_dict, convert_tensor_to_string(msg), d["shape"], d["color"], answer_type)
        elif agents == "two":
            for msg, m_prob in zip(d["msg_2"], d["probs_2"]):
                answer_type = 'correct' if d["correct"] else 'incorrect'
                m_dict = add_elem(m_dict, convert_tensor_to_string(msg), d["shape"], d["color"], answer_type)
        else:
            print("Error, please select 'both', 'one' or 'two' agents")
            break
    m_dict = calc_ratios(m_dict)
    print(f'Total messages: {m_dict["total"]}')
    exclude = ['total', 'correct', 'correct_p', 'incorrect', 'incorrect_p']
    list_dict = [(key, val) for key, val in m_dict.items() if key not in exclude]
    list_dict = sorted(list_dict, key=lambda x: x[1]['total'], reverse=True)
    return m_dict, list_dict


def count_blanks(data, blank_m1, blank_m2):
    counts = {'01': {'correct': 0, 'incorrect': 0},
              '10': {'correct': 0, 'incorrect': 0},
              '11': {'correct': 0, 'incorrect': 0},
              '00': {'correct': 0, 'incorrect': 0}}
    for _, d in enumerate(data):
        for msg_1, msg_2 in zip(d["msg_1"], d["msg_2"]):
            str_m1 = convert_tensor_to_string(msg_1)
            str_m2 = convert_tensor_to_string(msg_2)
            if str_m1 == blank_m1 and str_m2 == blank_m2:
                if d['correct']:
                    counts['11']['correct'] += 1
                else:
                    counts['11']['incorrect'] += 1
            elif str_m1 == blank_m1 and str_m2 != blank_m2:
                if d['correct']:
                    counts['10']['correct'] += 1
                else:
                    counts['10']['incorrect'] += 1
            elif str_m1 != blank_m1 and str_m2 == blank_m2:
                if d['correct']:
                    counts['01']['correct'] += 1
                else:
                    counts['01']['incorrect'] += 1
            elif str_m1 != blank_m1 and str_m2 != blank_m2:
                if d['correct']:
                    counts['00']['correct'] += 1
                else:
                    counts['00']['incorrect'] += 1
    total = counts['11']['correct'] + counts['11']['incorrect'] + counts['00']['correct'] + counts['00']['incorrect'] + counts['01']['correct'] + counts['01']['incorrect'] + counts['10']['correct'] + counts['10']['incorrect']
    counts['11']['p'] = (counts['11']['correct'] + counts['11']['incorrect']) / total
    counts['00']['p'] = (counts['00']['correct'] + counts['00']['incorrect']) / total
    counts['10']['p'] = (counts['10']['correct'] + counts['10']['incorrect']) / total
    counts['01']['p'] = (counts['01']['correct'] + counts['01']['incorrect']) / total
    pprint.pprint(counts)
    return counts


def calc_entropy_ratio(data, agents, answer_type):
    entropies = []
    messages = []
    message_probs = []
    convert = torch.log(torch.zeros(1).fill_(2))
    nans = 0
    for _, d in enumerate(data):
        include = None
        if answer_type == "both":
            include = True
        elif answer_type == "correct":
            include = d["correct"]
        else:
            include = not d["correct"]
        if include:
            if agents == "both":
                for msg, m_prob in zip(d["msg_1"], d["probs_1"]):
                    messages.append(msg.numpy())
                    message_probs.append(m_prob.numpy())
                    H = - torch.mul(m_prob, torch.log(m_prob) / convert).sum() - torch.mul(1 - m_prob, torch.log(1 - m_prob) / convert).sum()
                    if np.isnan(H):
                        # print("ARRRGHHH NAN!")
                        nans += 1
                    else:
                        entropies.append(H)
                for msg, m_prob in zip(d["msg_2"], d["probs_2"]):
                    messages.append(msg.numpy())
                    message_probs.append(m_prob.numpy())
                    H = - torch.mul(m_prob, torch.log(m_prob) / convert).sum() - torch.mul(1 - m_prob, torch.log(1 - m_prob) / convert).sum()
                    if np.isnan(H):
                        # print("ARRRGHHH NAN!")
                        nans += 1
                    else:
                        entropies.append(H)
            elif agents == "one":
                for msg, m_prob in zip(d["msg_1"], d["probs_1"]):
                    messages.append(msg.numpy())
                    message_probs.append(m_prob.numpy())
                    H = - torch.mul(m_prob, torch.log(m_prob) / convert).sum() - torch.mul(1 - m_prob, torch.log(1 - m_prob) / convert).sum()
                    if np.isnan(H):
                        # print("ARRRGHHH NAN!")
                        nans += 1
                    else:
                        entropies.append(H)
            elif agents == "two":
                for msg, m_prob in zip(d["msg_2"], d["probs_2"]):
                    messages.append(msg.numpy())
                    message_probs.append(m_prob.numpy())
                    H = - torch.mul(m_prob, torch.log(m_prob) / convert).sum() - torch.mul(1 - m_prob, torch.log(1 - m_prob) / convert).sum()
                    if np.isnan(H):
                        # print("ARRRGHHH NAN!")
                        nans += 1
                    else:
                        entropies.append(H)
    print(f'Number of messages skipped due to nan: {nans}')
    # print(f'Entropies: {entropies[:10]}')
    mean_e = sum(entropies) / len(entropies)
    messages = np.stack(messages)
    message_probs = np.stack(message_probs)
    mean_m = torch.from_numpy(np.mean(messages, axis=0)).float()
    mean_m_prob = torch.from_numpy(np.mean(message_probs, axis=0)).float()
    # Convert to base 2
    ent_mean_m = - torch.mul(mean_m, torch.log(mean_m) / convert).sum() - torch.mul(1 - mean_m, torch.log(1 - mean_m) / convert).sum()
    ent_mean_m_prob = - torch.mul(mean_m_prob, torch.log(mean_m_prob) / convert).sum() - torch.mul(1 - mean_m_prob, torch.log(1 - mean_m_prob) / convert).sum()
    # print(f'E(m|x) = {mean_m}')
    print(f'E[H(m|x)] = {mean_e}, H[E(m|x)] = {ent_mean_m}/{ent_mean_m_prob}')
    print(f'Ratio: {mean_e / ent_mean_m}/{mean_e / ent_mean_m_prob}')


def add_shape_color_elem(d, m_dict, shape=None, color=None, blanks=[]):
    for msg_1, msg_2 in zip(d["msg_1"], d["msg_2"]):
        str_m1 = convert_tensor_to_string(msg_1)
        str_m2 = convert_tensor_to_string(msg_2)
        inc_1 = False if str_m1 in blanks else True
        inc_2 = False if str_m2 in blanks else True
        if shape is not None and color is not None:
            m_dict[shape + '_' + color]['count'] += 1
            if inc_1:
                m_dict[shape + '_' + color]['1']['all'].append(msg_1.numpy())
                if str_m1 in m_dict[shape + '_' + color]['1']:
                    m_dict[shape + '_' + color]['1'][str_m1] += 1
                else:
                    m_dict[shape + '_' + color]['1'][str_m1] = 1
            if inc_2:
                m_dict[shape + '_' + color]['2']['all'].append(msg_2.numpy())
                if str_m2 in m_dict[shape + '_' + color]['2']:
                    m_dict[shape + '_' + color]['2'][str_m2] += 1
                else:
                    m_dict[shape + '_' + color]['2'][str_m2] = 1
        if shape is not None:
            m_dict[shape]['count'] += 1
            if inc_1:
                m_dict[shape]['1']['all'].append(msg_1.numpy())
                if str_m1 in m_dict[shape]['1']:
                    m_dict[shape]['1'][str_m1] += 1
                else:
                    m_dict[shape]['1'][str_m1] = 1
            if inc_2:
                m_dict[shape]['2']['all'].append(msg_2.numpy())
                if str_m2 in m_dict[shape]['2']:
                    m_dict[shape]['2'][str_m2] += 1
                else:
                    m_dict[shape]['2'][str_m2] = 1
        if color is not None:
            m_dict[color]['count'] += 1
            if inc_1:
                m_dict[color]['1']['all'].append(msg_1.numpy())
                if str_m1 in m_dict[color]['1']:
                    m_dict[color]['1'][str_m1] += 1
                else:
                    m_dict[color]['1'][str_m1] = 1
            if inc_2:
                m_dict[color]['2']['all'].append(msg_2.numpy())
                if str_m2 in m_dict[color]['2']:
                    m_dict[color]['2'][str_m2] += 1
                else:
                    m_dict[color]['2'][str_m2] = 1
    return m_dict


def study_shape_color(data, shape, color, blanks=[]):
    m_dict = {shape: {'1': {'all': []}, '2': {'all': []}, 'count': 0},
              color: {'1': {'all': []}, '2': {'all': []}, 'count': 0},
              shape + '_' + str(color): {'1': {'all': []}, '2': {'all': []}, 'count': 0}}
    skip_1 = 0
    skip_2 = 0
    for _, d in enumerate(data):
        if d['shape'] == shape and d['color'] == color:
            if convert_tensor_to_string(d['msg_1'][0]) in blanks:
                skip_1 += 1
            if convert_tensor_to_string(d['msg_2'][0]) in blanks:
                skip_2 += 1
            m_dict = add_shape_color_elem(d, m_dict, shape=shape, color=color, blanks=blanks)
        elif d['shape'] == shape:
            if convert_tensor_to_string(d['msg_1'][0]) in blanks:
                skip_1 += 1
            if convert_tensor_to_string(d['msg_2'][0]) in blanks:
                skip_2 += 1
            m_dict = add_shape_color_elem(d, m_dict, shape=shape, color=None, blanks=blanks)
        elif d['color'] == color:
            if convert_tensor_to_string(d['msg_1'][0]) in blanks:
                skip_1 += 1
            if convert_tensor_to_string(d['msg_2'][0]) in blanks:
                skip_2 += 1
            m_dict = add_shape_color_elem(d, m_dict, shape=None, color=color, blanks=blanks)
    # print(f'Skipped {skip_1}/{skip_2} messages for {shape} and / or {color}')
    '''Calc mean and std of messages per shape, color and shape color'''
    m_dict[color]['1']['all'] = np.stack(m_dict[color]['1']['all'])
    m_dict[color]['2']['all'] = np.stack(m_dict[color]['2']['all'])
    m_dict[shape]['1']['all'] = np.stack(m_dict[shape]['1']['all'])
    m_dict[shape]['2']['all'] = np.stack(m_dict[shape]['2']['all'])
    m_dict[shape + '_' + color]['1']['all'] = np.stack(m_dict[shape + '_' + color]['1']['all'])
    m_dict[shape + '_' + color]['2']['all'] = np.stack(m_dict[shape + '_' + color]['2']['all'])

    # print(f"Number {color}: A1: {m_dict[color]['1']['all'].shape}, A2: {m_dict[color]['2']['all'].shape}")
    m_dict[color]['1']['mean'] = np.mean(m_dict[color]['1']['all'], axis=0)
    m_dict[color]['1']['std'] = np.std(m_dict[color]['1']['all'], axis=0)
    m_dict[color]['1']['count'] = m_dict[color]['1']['all'].shape[0]
    del m_dict[color]['1']['all']
    m_dict[color]['2']['mean'] = np.mean(m_dict[color]['2']['all'], axis=0)
    m_dict[color]['2']['std'] = np.std(m_dict[color]['2']['all'], axis=0)
    m_dict[color]['2']['count'] = m_dict[color]['2']['all'].shape[0]
    del m_dict[color]['2']['all']
    # print(f"Number {color}: A1: {m_dict[color]['1']['count']}, A2: {m_dict[color]['2']['count']}")

    # print(f"Number {shape}: A1: {m_dict[shape]['1']['all'].shape}, A2: {m_dict[shape]['2']['all'].shape}")
    m_dict[shape]['1']['mean'] = np.mean(m_dict[shape]['1']['all'], axis=0)
    m_dict[shape]['1']['std'] = np.std(m_dict[shape]['1']['all'], axis=0)
    m_dict[shape]['1']['count'] = m_dict[shape]['1']['all'].shape[0]
    del m_dict[shape]['1']['all']
    m_dict[shape]['2']['mean'] = np.mean(m_dict[shape]['2']['all'], axis=0)
    m_dict[shape]['2']['std'] = np.std(m_dict[shape]['2']['all'], axis=0)
    m_dict[shape]['2']['count'] = m_dict[shape]['2']['all'].shape[0]
    del m_dict[shape]['2']['all']
    # print(f"Number {shape}: A1: {m_dict[shape]['1']['count']}, A2: {m_dict[shape]['2']['count']}")

    # print(f"Number {shape}_{color}: A1: {m_dict[shape + '_' + color]['1']['all'].shape}, A2: {m_dict[shape + '_' + color]['2']['all'].shape}")
    m_dict[shape + '_' + color]['1']['mean'] = np.mean(m_dict[shape + '_' + color]['1']['all'], axis=0)
    m_dict[shape + '_' + color]['1']['std'] = np.std(m_dict[shape + '_' + color]['1']['all'], axis=0)
    m_dict[shape + '_' + color]['1']['count'] = m_dict[shape + '_' + color]['1']['all'].shape[0]
    del m_dict[shape + '_' + color]['1']['all']
    m_dict[shape + '_' + color]['2']['mean'] = np.mean(m_dict[shape + '_' + color]['2']['all'], axis=0)
    m_dict[shape + '_' + color]['2']['std'] = np.std(m_dict[shape + '_' + color]['2']['all'], axis=0)
    m_dict[shape + '_' + color]['2']['count'] = m_dict[shape + '_' + color]['2']['all'].shape[0]
    del m_dict[shape + '_' + color]['2']['all']
    # print(f"Number {shape}_{color}: A1: {m_dict[shape + '_' + color]['1']['count']}, A2: {m_dict[shape + '_' + color]['2']['count']}")
    return m_dict


def quantize_codes(codes):
    codes = codes.numpy()
    codes[np.where(codes > 0.95)] = 11
    codes[np.where(codes >= 0.8, np.where(codes < 0.95, True, False), False)] = 1
    codes[np.where(codes <= 0.05)] = -1
    codes[np.where(codes <= 0.2, np.where(codes > 0.05, True, False), False)] = 0
    codes[np.where(codes > 0.2, np.where(codes < 0.8, True, False), False)] = np.nan
    codes = torch.from_numpy(codes)
    return codes


def create_shape_color_combos(data, blanks=[]):
    print(f'Blanks: {blanks}')
    print('================================== SHAPES ==================================')
    codes_c1 = []
    codes_c2 = []
    codes_s1 = []
    codes_s2 = []
    for s in SHAPES:
        mean_1 = []
        mean_2 = []
        cols = []
        for c in COLORS:
            if (s, c) in OUTOFDOMAIN:
                continue
            sc = s + '_' + c
            cols.append(c)
            result = study_shape_color(data, s, c, blanks=blanks)
            mean_1.append(result[sc]['1']['mean'])
            mean_2.append(result[sc]['2']['mean'])
        if len(mean_1) > 0:
            assert len(mean_1) == len(mean_2)
            print(f'Mean messages for {s}: colors: {cols}')
            mean_1 = torch.from_numpy(np.stack(mean_1))
            mean_2 = torch.from_numpy(np.stack(mean_2))
            print(f'Agent 1: {mean_1}')
            print(f'Agent 2: {mean_2}')
        result = study_shape_color(data, s, 'gray', blanks=blanks)
        print(f'Num messages for {s}, A1: {result[s]["1"]["count"]}, A2: {result[s]["2"]["count"]}')
        c_1 = result[s]['1']['mean']
        c_2 = result[s]['2']['mean']
        codes_s1.append((s, c_1))
        codes_s2.append((s, c_2))
    print('================================== COLORS ==================================')
    for c in COLORS:
        mean_1 = []
        mean_2 = []
        sh = []
        for s in SHAPES:
            if (s, c) in OUTOFDOMAIN:
                continue
            sc = s + '_' + c
            sh.append(s)
            result = study_shape_color(data, s, c, blanks=blanks)
            mean_1.append(result[sc]['1']['mean'])
            mean_2.append(result[sc]['2']['mean'])
        if len(mean_1) > 0:
            assert len(mean_1) == len(mean_2)
            print(f'Mean messages for {c}: shapes: {sh}')
            mean_1 = torch.from_numpy(np.stack(mean_1))
            mean_2 = torch.from_numpy(np.stack(mean_2))
            print(f'Agent 1: {mean_1}')
            print(f'Agent 2: {mean_2}')
        result = study_shape_color(data, 'semicircle', c, blanks=blanks)
        print(f'Num messages for {c}, A1: {result[c]["1"]["count"]}, A2: {result[c]["2"]["count"]}')
        c_1 = result[c]['1']['mean']
        c_2 = result[c]['2']['mean']
        codes_c1.append((c, c_1))
        codes_c2.append((c, c_2))
    shapes = [x[0] for x in codes_s1]
    print(f'Shapes: {shapes}')
    shape_codes_1 = [x[1] for x in codes_s1]
    shape_codes_mean_1 = torch.from_numpy(np.stack(shape_codes_1))
    print(f'Mean shape codes, agent 1: {shape_codes_mean_1}')
    shape_codes_2 = [x[1] for x in codes_s2]
    shape_codes_mean_2 = torch.from_numpy(np.stack(shape_codes_2))
    print(f'Mean shape codes, agent 2: {shape_codes_mean_2}')
    colors = [x[0] for x in codes_c1]
    print(f'Colors: {colors}')
    color_codes_1 = [x[1] for x in codes_c1]
    color_codes_mean_1 = torch.from_numpy(np.stack(color_codes_1))
    print(f'Mean color codes, agent 1: {color_codes_mean_1}')
    color_codes_2 = [x[1] for x in codes_c2]
    color_codes_mean_2 = torch.from_numpy(np.stack(color_codes_2))
    print(f'Mean color codes, agent 2: {color_codes_mean_2}')
    shape_codes_mean_1 = quantize_codes(shape_codes_mean_1)
    print(f'Shapes: {shapes}')
    print(f'Quantized shape codes, agent 1: {shape_codes_mean_1}')
    shape_codes_mean_2 = quantize_codes(shape_codes_mean_2)
    print(f'Quantized shape codes, agent 2: {shape_codes_mean_2}')
    color_codes_mean_1 = quantize_codes(color_codes_mean_1)
    print(f'Colors: {colors}')
    print(f'Quantized color codes, agent 1: {color_codes_mean_1}')
    color_codes_mean_2 = quantize_codes(color_codes_mean_2)
    print(f'Quantized color codes, agent 2: {color_codes_mean_2}')
    return codes_c1, codes_c2, codes_s1, codes_s2


def print_shape_color_results(result, shape, color, sc):
    print(f"{shape}: {1}:\n mean:{result[shape]['1']['mean']}\nstd: {result[shape]['1']['std']}")
    print(f"{color}: {1}:\nmean: {result[color]['1']['mean']}\nstd: {result[color]['1']['std']}")
    print(f"{sc}: {1}:\nmean: {result[sc]['1']['mean']}\nstd: {result[sc]['1']['std']}\n")
    print(f"{shape}: {2}:\nmean: {result[shape]['2']['mean']}\nstd: {result[shape]['2']['std']}")
    print(f"{color}: {2}:\nmean: {result[color]['2']['mean']}\nstd: {result[color]['2']['std']}")
    print(f"{sc}: {2}:\nmean: {result[sc]['2']['mean']}\nstd: {result[sc]['2']['std']}")


def convert_codes_to_dict(c1, c2):
    d = {}
    for elem in c1:
        d[elem[0]] = torch.from_numpy(elem[1])
    for elem in c2:
        d[elem[0]] = torch.from_numpy(elem[1])
    return d


def analyze_messages(data, path):
    print(f'========================== Analyzing messages from {args.path} ==========================')
    m_dict, list_dict = build_message_dict(data, agents="one")
    m_dict_2, list_dict_2 = build_message_dict(data, agents="two")
    num_messages = len(list(m_dict.keys())) - 5
    num_messages_2 = len(list(m_dict_2.keys())) - 5
    print(f'Num distinct messages: {num_messages}/{num_messages_2}')
    print(f'Correct: {m_dict["correct_p"]}, incorrect: {m_dict["incorrect_p"]}')
    print(f'Correct: {m_dict_2["correct_p"]}, incorrect: {m_dict_2["incorrect_p"]}')
    print(f'================================== Distribution of messages agent 1: ==================================')
    for i, l in enumerate(list_dict):
        print(f'{i}: Message: {l[0]}, count: {l[1]["total"]}')
    print(f'================================== Distribution of messages agent 2: ==================================')
    for i, l in enumerate(list_dict_2):
        print(f'{i}: Message: {l[0]}, count: {l[1]["total"]}')
    counts = count_pratios(m_dict)
    counts_2 = count_pratios(m_dict_2)
    print('================================== pratio stats ==================================')
    mean_pratio = get_pratio_stats(counts)
    print('================================== pratio stats 2 ==================================')
    mean_pratio_2 = get_pratio_stats(counts_2)
    print('================================== I cant see anything message ==================================')
    for i in range(5):
        blank_m1 = list_dict[i][0]
        blank_m2 = list_dict_2[i][0]
        p_blanks_m1 = list_dict[i][1]['total'] / m_dict['total']
        p_blanks_m2 = list_dict_2[i][1]['total'] / m_dict_2['total']
        print(f'Blank_m1: {blank_m1}, blanks_m2: {blank_m2}, probs: {p_blanks_m1}/{p_blanks_m2}, joint prob {p_blanks_m1 * p_blanks_m2}')
        _ = count_blanks(data, blank_m1, blank_m2)
    print('================================== Entropies ==================================')
    print(f'Overall entropy ratio')
    print(f'Overall')
    calc_entropy_ratio(data, "both", "both")
    print(f'Correct')
    calc_entropy_ratio(data, "both", "correct")
    print(f'Incorrect')
    calc_entropy_ratio(data, "both", "incorrect")
    print(f'Entropy ratio agent 1')
    print(f'Overall')
    calc_entropy_ratio(data, "one", "both")
    print(f'Correct')
    calc_entropy_ratio(data, "one", "correct")
    print(f'Incorrect')
    calc_entropy_ratio(data, "one", "incorrect")
    print(f'Entropy ratio agent 2')
    print(f'Overall')
    calc_entropy_ratio(data, "two", "both")
    print(f'Correct')
    calc_entropy_ratio(data, "two", "correct")
    print(f'Incorrect')
    calc_entropy_ratio(data, "two", "incorrect")
    print('================================== Shape color studies ==================================')
    shape = "triangle"
    color = "blue"
    print(f'shape: {shape}, color: {color}')
    sc = shape + '_' + color
    result = study_shape_color(data, shape, color)
    pprint.pprint(result)
    shape = "square"
    color = "blue"
    print(f'shape: {shape}, color: {color}')
    sc = shape + '_' + color
    result = study_shape_color(data, shape, color)
    pprint.pprint(result)
    shape = "square"
    color = "yellow"
    print(f'shape: {shape}, color: {color}')
    sc = shape + '_' + color
    result = study_shape_color(data, shape, color)
    pprint.pprint(result)
    _, _, _, _ = create_shape_color_combos(data)
    print('================================== Shape color studies, no blank msg ==================================')
    codes_c1, codes_c2, codes_s1, codes_s2 = create_shape_color_combos(data, blanks=[list_dict[0][0], list_dict_2[0][0]])
    dict_c1 = convert_codes_to_dict(codes_c1, codes_s1)
    dict_c2 = convert_codes_to_dict(codes_c2, codes_s2)
    print("Agent 1 dictionary")
    pprint.pprint(dict_c1)
    print("Agent 2 dictionary")
    pprint.pprint(dict_c2)
    print("Saving average shape and color code dictionaries to file...")
    pickle.dump(dict_c1, open(path + '_mean_codes_A1.pkl', 'wb'))
    pickle.dump(dict_c2, open(path + '_mean_codes_A2.pkl', 'wb'))


def check_distribution(data):
    counts = {}
    for _, d in enumerate(data):
        if d['shape'] not in counts:
            counts[d['shape']] = 1
        else:
            counts[d['shape']] += 1
        if d['color'] not in counts:
            counts[d['color']] = 1
        else:
            counts[d['color']] += 1
        if d['shape'] is not None and d['color'] is not None:
            sc = d['shape'] + '_' + d['color']
            if sc not in counts:
                counts[sc] = 1
            else:
                counts[sc] += 1
    pprint.pprint(counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze messages')
    parser.add_argument('--path', type=str, default="./logs/experiments_030718/big_valid_msg_eval_only_A_1_2_message_stats.pkl",
                        help='Path to messages')
    args = parser.parse_args()
    data = pickle.load(open(args.path, 'rb'))
    check_distribution(data)
    analyze_messages(data, args.path[:-4])
