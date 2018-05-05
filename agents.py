import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as _Variable
import torch.optim as optim
from torch.nn.parameter import Parameter
import math
import numpy as np
import logging
import functools

from misc import xavier_normal

FORMAT = '[%(asctime)s %(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT)
debuglogger = logging.getLogger('main_logger')
debuglogger.setLevel('INFO')


def reset_parameters_util(model):
    for m in model.modules():
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


class ImageProcessor(nn.Module):
    '''Processes an agent's image, with or without attention'''

    def __init__(self, im_feat_dim, hid_dim, use_attn, attn_dim):
        super(ImageProcessor, self).__init__()
        self.im_feat_dim = im_feat_dim
        self.hid_dim = hid_dim
        self.use_attn = use_attn
        self.attn_dim = attn_dim
        self.im_transform = nn.Linear(self.im_feat_dim, self.hid_dim)
        self.attn_W_x = nn.Linear(self.im_feat_dim, self.attn_dim)
        self.attn_W_w = nn.Linear(self.hid_dim, self.attn_dim)
        self.attn_U = nn.Linear(self.attn_dim, 1)
        self.attn_scores = []
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_util(self)

    def reset_state(self):
        # Used for debugging.
        self.attn_scores = []

    def get_attn_scores(self, x, h_z):
        batch_size, n_feats, channels = x.size()
        # Process hidden state
        h_w_attn = self.attn_W_w(h_z)
        debuglogger.debug(f'h_w_attn: {h_w_attn.size()}')
        h_w_attn_broadcast = h_w_attn.contiguous().unsqueeze(
            1).expand(batch_size, n_feats, self.attn_dim)
        debuglogger.debug(f'h_w_broadcast: {h_w_broadcast.size()}')
        h_w_attn_flat = h_w_attn_broadcast.contiguous().view(
            batch_size * n_feats, self.attn_dim)
        debuglogger.debug(f'h_w_flat: {h_w_flat.size()}')
        # Process image
        x_flat = x.contiguous().view(batch_size * n_feats, channels)
        debuglogger.debug(f'x_flat: {x_flat.size()}')
        h_x_attn_flat = self.attn_W_x(x_flat)
        debuglogger.debug(f'h_x_attn_flat: {h_x_attn_flat.size()}')
        # Calculate attention scores
        attn_U_inp = nn.Tanh()(h_w_attn_flat + h_x_attn_flat)
        attn_scores_flat = self.attn_U(attn_U_inp)
        debuglogger.debug(f'attn_scores_flat: {attn_scores_flat.size()}')
        attn_scores = attn_scores_flat.view(batch_size, n_feats)
        debuglogger.debug(f'attn_scores: {attn_scores.size()}')

        return attn_scores

    def forward(self, x, h_z, t):
        '''
        x = x or image_attn(x)
            Image Attention (https://arxiv.org/pdf/1502.03044.pdf):
                \beta_i = U tanh(W_r h_z + W_x x_i)
                \alpha = 1 / |x|        if t == 0
                \alpha = softmax(\beta) otherwise
                x = \sum_i \alpha x_i
        Returns
            h_i = im_transform(x)
        '''
        debuglogger.debug(f'Inside image processing...')
        if self.use_attn:
            batch_size, channels, height, width = x.size()
            n_feats = height * width
            debuglogger.debug(f'x: {x.size()}')
            x = x.view(batch_size, channels, n_feats)
            debuglogger.debug(f'x: {x.size()}')
            x = x.transpose(1, 2)
            debuglogger.debug(f'x: {x.size()}')
            attn_scores = self.get_attn_scores(x, h_z)
            # attention scores
            if t == 0:
                attn_scores = Variable(torch.FloatTensor(
                    batch_size, n_feats).fill_(1), volatile=not self.training)
                attn_scores = attn_scores / n_feats
            else:
                attn_scores = F.softmax(attn_scores, dim=1)
            debuglogger.debug(f'attn_scores: {attn_scores.size()}')
            debuglogger.debug(f'attn_scores: {attn_scores}')
            x_attn = torch.bmm(attn_scores.unsqueeze(1), x).squeeze()
            debuglogger.debug(f'x with attn: {x_attn.size()}')
            # Cache values for inspection
            self.attn_scores.append(attn_scores)

            _x = x_attn
        else:
            _x = x
        # Transform image to hid_dim shape
        h_i = F.relu(self.im_transform(_x))
        return h_i


class ImageProcessorFromScratch(nn.Module):
    '''Processes an agent's image, with or without attention'''

    def __init__(self, im_dim, hid_dim, use_attn, attn_dim, dropout):
        super(ImageProcessorFromScratch, self).__init__()
        self.im_dim = (3, im_dim, im_dim)
        self.hid_dim = hid_dim
        self.use_attn = use_attn
        self.attn_dim = attn_dim
        self.dropout = dropout
        self.model = self.build_model()
        self.attn_scores = []
        self.reset_parameters()

    def build_model(self):
        layers = []
        layers += [nn.Conv2d(3, 16, kernel_size=3, stride=2)]
        layers += [nn.BatchNorm2d(16)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(16, 32, kernel_size=3, stride=2)]
        layers += [nn.BatchNorm2d(32)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout2d(p=self.dropout)]
        layers += [nn.Conv2d(32, 32, kernel_size=3, stride=2)]
        layers += [nn.BatchNorm2d(32)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(32, 64, kernel_size=3, stride=2)]
        layers += [nn.BatchNorm2d(64)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout2d(p=self.dropout)]
        layers += [nn.Conv2d(64, self.hid_dim, kernel_size=3, stride=2)]
        return nn.Sequential(*layers)

    def reset_parameters(self):
        reset_parameters_util(self)

    def reset_state(self):
        # Used for debugging.
        self.attn_scores = []

    def forward(self, x, h_z, t):
        '''
        x = x or image_attn(x)
            Image Attention (https://arxiv.org/pdf/1502.03044.pdf):
                \beta_i = U tanh(W_r h_z + W_x x_i)
                \alpha = 1 / |x|        if t == 0
                \alpha = softmax(\beta) otherwise
                x = \sum_i \alpha x_i
        Returns
            h_i = im_transform(x)
        '''
        debuglogger.debug(f'Inside image processing...')
        batch_size = x.size(0)
        if self.use_attn:
            debuglogger.warn(f'Not implemented yet')
            sys.exit()
        else:
            _x = x
        _x = self.model(_x)
        h, w = _x.size(2), _x.size(3)
        _x = nn.functional.avg_pool2d(_x, (h, w))
        _x = _x.view(batch_size, -1)
        h_i = F.relu(_x)
        return h_i


class TextProcessor(nn.Module):
    '''Processes sentence representations to the correct hidden dimension'''

    def __init__(self, desc_dim, hid_dim):
        super(TextProcessor, self).__init__()
        self.desc_dim = desc_dim
        self.hid_dim = hid_dim
        self.transform = nn.Linear(desc_dim, hid_dim)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_util(self)

    def forward(self, desc):
        bs, num_classes, desc_dim = desc.size()
        desc = desc.view(-1, desc_dim)
        out = self.transform(desc)
        out = out.view(bs, num_classes, -1)
        return F.relu(out)


class MessageProcessor(nn.Module):
    '''Processes a received message from an agent'''

    def __init__(self, m_dim, hid_dim, cuda, identify_agents, num_agents):
        super(MessageProcessor, self).__init__()
        self.m_dim = m_dim
        self.hid_dim = hid_dim
        self.use_cuda = cuda
        self.identify_agents = identify_agents
        self.num_agents = num_agents
        if self.identify_agents:
            self.rnn = nn.GRUCell(self.m_dim + self.num_agents, self.hid_dim)
        else:
            self.rnn = nn.GRUCell(self.m_dim, self.hid_dim)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_util(self)

    def concat_identifier(self, message, m_identifier):
        # Concatenate agent identifier with message
        bs = message.size(0)
        agent_identity = torch.zeros(bs, self.num_agents)
        # -1 is the "blank message" option
        if m_identifier != -1:
            agent_identity[:, m_identifier] = 1
        if self.use_cuda:
            agent_identity = agent_identity.cuda()
        # debuglogger.info(f'Agent identifier: {agent_identity}')
        # debuglogger.info(f'Message: {message}')
        message = torch.cat([agent_identity, message.data], dim=1)
        # debuglogger.info(f'Combined shape: {message.shape}')
        # debuglogger.info(f'Combined: {message}')
        message = _Variable(message)
        return message

    def forward(self, m, h, use_message, m_identifier):
        if self.identify_agents:
            m = self.concat_identifier(m, m_identifier)
        if use_message:
            debuglogger.debug(f'Using message')
            return self.rnn(m, h)
        else:
            debuglogger.debug(f'Ignoring message, using blank instead...')
            blank_msg = _Variable(torch.zeros_like(m.data))
            if self.identify_agents:
                blank_msg = concat_identifier(blank_msg, m_identifier)
            if self.use_cuda:
                blank_msg = blank_msg.cuda()
            return self.rnn(blank_msg, h)


class MessageGenerator(nn.Module):
    '''Generates a message for an agent
    TODO MAKE RECURRENT? - later'''

    def __init__(self, m_dim, hid_dim, use_binary):
        super(MessageGenerator, self).__init__()
        self.m_dim = m_dim
        self.hid_dim = hid_dim
        self.use_binary = use_binary
        # Why different biases?
        self.w_h = nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        self.w_d = nn.Linear(self.hid_dim, self.hid_dim, bias=False)
        self.w = nn.Linear(self.hid_dim, self.m_dim)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_util(self)

    def forward(self, y_scores, h_c, desc, training):
        '''
            desc = \sum_i y_scores desc_i
            w_hat = tanh(W_h h_c + W_d desc)
            w = bernoulli(sig(w_hat)) or round(sig(w_hat))
        '''
        # y_scores: batch_size x num_classes
        # desc: batch_size x num_classes x hid_dim
        # h_c: batch_size x hid_dim
        batch_size, num_classes = y_scores.size()
        y_broadcast = y_scores.unsqueeze(2).expand(
            batch_size, num_classes, self.hid_dim)
        debuglogger.debug(f'y_broadcast: {y_broadcast.size()}')
        # debuglogger.debug(f'y_broadcast: {y_broadcast}')
        debuglogger.debug(f'desc: {desc.size()}')
        # Weight descriptions based on current predictions
        desc = torch.mul(y_broadcast, desc).sum(1).squeeze(1)
        debuglogger.debug(f'desc: {desc.size()}')
        # desc: batch_size x hid_dim
        h_w = F.tanh(self.w_h(h_c) + self.w_d(desc))
        w_scores = self.w(h_w)
        if self.use_binary:
            w_probs = F.sigmoid(w_scores)
            if training:
                # debuglogger.info(f"Training...")
                probs_ = w_probs.data.cpu().numpy()
                rand_num = np.random.rand(*probs_.shape)
                # debuglogger.debug(f'rand_num: {rand_num}')
                # debuglogger.info(f'probs: {probs_}')
                w_binary = _Variable(torch.from_numpy(
                    (rand_num < probs_).astype('float32')))
            else:
                # debuglogger.info(f"Eval mode, rounding...")
                w_binary = torch.round(w_probs).detach()
            if w_probs.is_cuda:
                w_binary = w_binary.cuda()
            w_feats = w_binary
            # debuglogger.debug(f'w_binary: {w_binary}')
        else:
            debuglogger.warn(f'Error: Training loop with real valued messages not implemented yet. Please set FLAGS.use_binary to true')
            sys.exit()
            w_feats = w_scores
            w_probs = None
        # debuglogger.info(f'Message : {w_feats}')
        return w_feats, w_probs


class RewardEstimator(nn.Module):
    '''Estimates the reward the agent will receieved. Value used as a baseline in REINFORCE loss'''

    def __init__(self, hid_dim):
        super(RewardEstimator, self).__init__()
        self.hid_dim = hid_dim
        self.v1 = nn.Linear(hid_dim, math.ceil(hid_dim / 2))
        self.v2 = nn.Linear(math.ceil(hid_dim / 2), 1)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_util(self)

    def forward(self, x):
        # Detach input from rest of graph - only want gradients to flow through the RewardEstimator and no future
        x = x.detach()
        x = F.relu(self.v1(x))
        x = self.v2(x)
        return x


class Agent(nn.Module):
    def __init__(self,
                 im_feature_type,
                 im_feat_dim,
                 h_dim,
                 m_dim,
                 desc_dim,
                 num_classes,
                 s_dim,
                 use_binary,
                 use_attn,
                 attn_dim,
                 use_MLP,
                 cuda,
                 im_from_scratch,
                 dropout,
                 identify_agents,
                 num_agents):
        super(Agent, self).__init__()
        self.im_feature_type = im_feature_type
        self.im_feat_dim = im_feat_dim
        self.h_dim = h_dim
        self.m_dim = m_dim
        self.desc_dim = desc_dim
        self.num_classes = num_classes
        self.s_dim = s_dim
        self.use_binary = use_binary
        self.use_attn = use_attn
        self.use_MLP = use_MLP
        self.attn_dim = attn_dim
        self.use_cuda = cuda
        self.identify_agents = identify_agents
        self.num_agents = num_agents
        if im_from_scratch:
            self.image_processor = ImageProcessorFromScratch(
                im_feat_dim, h_dim, use_attn, attn_dim, dropout)
        else:
            self.image_processor = ImageProcessor(
                im_feat_dim, h_dim, use_attn, attn_dim)
        self.text_processor = TextProcessor(desc_dim, h_dim)
        self.message_processor = MessageProcessor(m_dim, h_dim, cuda, identify_agents, num_agents)
        self.message_generator = MessageGenerator(m_dim, h_dim, use_binary)
        self.reward_estimator = RewardEstimator(h_dim)
        # Network for combining processed image and message representations
        self.text_im_combine = nn.Linear(h_dim * 2, h_dim)
        # Network for making predicitons
        self.y1 = nn.Linear(self.h_dim * 2, self.h_dim)
        self.y2 = nn.Linear(self.h_dim, 1)
        # Network for making stop decision decisions
        self.s = nn.Linear(self.h_dim, self.s_dim)
        self.h_z = None
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.set_(xavier_normal(m.weight.data))
                if m.bias is not None:
                    m.bias.data.zero_()
        self.image_processor.reset_parameters()
        self.message_processor.reset_parameters()
        self.text_processor.reset_parameters()
        self.message_generator.reset_parameters()
        self.reward_estimator.reset_parameters()

    def reset_state(self):
        """Initialize state for the agent.

        The agent is stateful, keeping tracking of previous messages it
        has sent and received.

        """
        self.h_z = None
        self.image_processor.reset_state()

    def initial_state(self, batch_size):
        h = _Variable(torch.zeros(batch_size, self.h_dim))
        if self.use_cuda:
            h = h.cuda()
        return h

    def predict_classes(self, h_c, desc_proc, batch_size):
        '''
        Scores each class using an MLP or simple dot product
        desc_proc:     bs x num_classes x hid_dim
        h_c:           bs x hid_dim
        h_c:           bs x hid_dim x 1
        h_c_expand:    bs x num_classes x hid_dim
        hid_cat_desc:  (bs x num_classes) x (hid_dim * 2)
        y:             bs x num_classes
        '''
        if self.use_MLP:
            h_c_expand = torch.unsqueeze(
                h_c, dim=1).expand(-1, self.num_classes, -1)
            debuglogger.debug(f'h_c_expand: {h_c_expand.size()}')
            # debuglogger.debug(f'h_c: {h_c}')
            # debuglogger.debug(f'h_c_expand: {h_c_expand}')
            hid_cat_desc = torch.cat([h_c_expand, desc_proc], dim=2)
            debuglogger.debug(f'hid_cat_desc: {hid_cat_desc.size()}')
            hid_cat_desc = hid_cat_desc.view(-1, self.h_dim * 2)
            debuglogger.debug(f'hid_cat_desc: {hid_cat_desc.size()}')
            y = F.relu(self.y1(hid_cat_desc))
            debuglogger.debug(f'y: {y.size()}')
            y = self.y2(y).view(batch_size, -1)
        else:
            h_c_unsqueezed = h_c.unsqueeze(dim=2)
            y = torch.bmm(desc_proc, h_c_unsqueezed).squeeze(dim=2)
        debuglogger.debug(f'y: {y.size()}')
        return y

    def forward(self, x, m, t, desc, use_message, batch_size, training, m_identifier):
        """
        Update State:
            h_z = message_processor(m, h_z, m_identifier)

        Image processing
            h_i = image_processor(x, h_z)
            Image Attention (https://arxiv.org/pdf/1502.03044.pdf):
                \beta_i = U tanh(W_r h_z + W_x x_i)
                \alpha = 1 / |x|        if t == 0
                \alpha = softmax(\beta) otherwise
                x = \sum_i \alpha x_i

        Combine Image and Message information
            h_c = text_im_combine(h_z, h_i)

        Text processing
            desc_proc = text_processor(desc)

        STOP Bit:
            s_hat = W_s h_c
            s = bernoulli(sig(s_hat)) or round(sig(s_hat))

        Predictions:
            y_i = f_y(h_c, desc_proc_i)

        Generate message:
            m_out = message_generator(y, h_c, desc_proc)
            Communication:
                desc = \sum_i y_i t_i
                w_hat = tanh(W_h h_c + W_d t)
                w = bernoulli(sig(w_hat)) or round(sig(w_hat))

        Args:
            x: Image features.
            m: communication from other agent
            t: (attention) Timestep. Used to change attention equation in first iteration.
            desc: List of description vectors used in communication and predictions.
            batch_size: size of batch
            training: whether agent is training or not
            m_identifier: identity of agent that sent the message
        Output:
            s, s_probs: A STOP bit and its associated probability, indicating whether the agent has decided to make a selection. If the exchange length is not set to FIXED then the conversation will continue until both agents have selected STOP.
            w, w_probs: A binary message and the probability of each bit in the message being ``1``.
            y: A prediction for each class described in the descriptions.
            r: An estimate of the reward the agent will receive
        """
        debuglogger.debug(f'Input sizes...')
        debuglogger.debug(f'x: {x.size()}')
        debuglogger.debug(f'm: {m.size()}')
        debuglogger.debug(f'm: {m}')
        debuglogger.debug(f'desc: {desc.size()}')

        # Initialize hidden state if necessary
        if self.h_z is None:
            self.h_z = self.initial_state(batch_size)

        # Process message sent from the other agent
        self.h_z = self.message_processor(m, self.h_z, use_message, m_identifier)
        debuglogger.debug(f'h_z: {self.h_z.size()}')

        # Process the image
        h_i = self.image_processor(x, self.h_z, t)
        debuglogger.debug(f'h_i: {h_i.size()}')

        # Combine the image and message info to a single vector
        h_c = self.text_im_combine(torch.cat([self.h_z, h_i], dim=1))
        debuglogger.debug(f'h_c: {h_c.size()}')

        # Process the texts
        # desc: bs x num_classes x desc_dim
        # desc_proc:    bs x num_classes x hid_dim
        desc_proc = self.text_processor(desc)
        debuglogger.debug(f'desc_proc: {desc_proc.size()}')

        # Estimate the reward
        r = self.reward_estimator(h_c)
        debuglogger.debug(f'r: {r.size()}')

        # Calculate stop bits
        s_score = self.s(h_c)
        s_prob = F.sigmoid(s_score)
        debuglogger.debug(f's_score: {s_score.size()}')
        debuglogger.debug(f's_prob: {s_prob.size()}')
        if training:
            # Sample decisions
            prob_ = s_prob.data.cpu().numpy()
            rand_num = np.random.rand(*prob_.shape)
            # debuglogger.debug(f'rand_num: {rand_num}')
            # debuglogger.debug(f'prob: {prob_}')
            s_binary = _Variable(torch.from_numpy(
                (rand_num < prob_).astype('float32')))
            if self.use_cuda:
                s_binary = s_binary.cuda()
        else:
            # Infer decisions
            s_binary = torch.round(s_prob).detach()
        debuglogger.debug(f'stop decisions: {s_binary.size()}')
        # debuglogger.debug(f'stop decisions: {s_binary}')

        # Predict classes
        # y: batch_size * num_classes
        y = self.predict_classes(h_c, desc_proc, batch_size)
        y_scores = F.softmax(y, dim=1).detach()
        debuglogger.debug(f'y_scores: {y_scores.size()}')
        # debuglogger.debug(f'y_scores: {y_scores}')

        # Generate message
        w, w_probs = self.message_generator(y_scores, h_c, desc_proc, training)
        debuglogger.debug(f'w: {w.size()}')
        debuglogger.debug(f'w_probs: {w_probs.size()}')

        return (s_binary, s_prob), (w, w_probs), y, r


if __name__ == "__main__":
    print("Testing agent init and forward pass...")
    im_feature_type = ""
    im_feat_dim = 128
    h_dim = 64
    m_dim = 6
    desc_dim = 100
    num_classes = 3
    s_dim = 1
    use_binary = True
    use_message = True
    use_attn = False
    attn_dim = 128
    batch_size = 8
    training = True
    dropout = 0.3
    use_MLP = False
    cuda = False
    im_from_scratch = False
    identify_agents = True
    num_agents = 7
    agent = Agent(im_feature_type,
                  im_feat_dim,
                  h_dim,
                  m_dim,
                  desc_dim,
                  num_classes,
                  s_dim,
                  use_binary,
                  use_attn,
                  attn_dim,
                  use_MLP,
                  cuda,
                  im_from_scratch,
                  dropout,
                  identify_agents,
                  num_agents)
    print(agent)
    total_params = sum([functools.reduce(lambda x, y: x * y, p.size(), 1.0)
                        for p in agent.parameters()])
    image_proc_params = sum([functools.reduce(lambda x, y: x * y, p.size(), 1.0)
                            for p in agent.image_processor.parameters()])
    print(f'Total params: {total_params}, image proc params: {image_proc_params}')
    if im_from_scratch:
        x = _Variable(torch.ones(batch_size, 3, im_feat_dim, im_feat_dim))
    else:
        x = _Variable(torch.ones(batch_size, im_feat_dim))
    m = _Variable(torch.ones(batch_size, m_dim))
    desc = _Variable(torch.ones(batch_size, num_classes, desc_dim))

    agent_identifier = -1
    for i in range(2):
        s, w, y, r = agent(x, m, i, desc, use_message, batch_size, training, agent_identifier)
        # print(f's_binary: {s[0]}')
        # print(f's_probs: {s[1]}')
        # print(f'w_binary: {w[0]}')
        # print(f'w_probs: {w[1]}')
        # print(f's_binary: {s[0]}')
        # print(f's_probs: {s[1]}')
        # print(f'y: {y}')
        # print(f'r: {r}')
