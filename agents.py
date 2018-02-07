import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as _Variable
import torch.optim as optim
from torch.nn.parameter import Parameter
import math
import numpy as np
import logging

from misc import xavier_normal

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
debuglogger = logging.getLogger('main_logger')
debuglogger.setLevel(10)


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

    def __init__(self, m_dim, hid_dim):
        super(MessageProcessor, self).__init__()
        self.m_dim = m_dim
        self.hid_dim = hid_dim
        self.rnn = nn.GRUCell(self.m_dim, self.hid_dim)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_util(self)

    def forward(self, m, h, use_message):
        if use_message:
            debuglogger.debug(f'Using message')
            return self.rnn(m, h)
        else:
            debuglogger.debug(f'Ignoring message, using blank instead...')
            blank_msg = torch.zeros_like(m)
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
        debuglogger.debug(f'y_broadcast: {y_broadcast}')
        debuglogger.debug(f'desc: {desc.size()}')
        desc = torch.mul(y_broadcast, desc).sum(1).squeeze(1)
        debuglogger.debug(f'desc: {desc.size()}')
        # desc: batch_size x hid_dim
        h_w = F.tanh(self.w_h(h_c) + self.w_d(desc))
        w_scores = self.w(h_w)
        if self.use_binary:
            w_probs = F.sigmoid(w_scores)
            if training:
                probs_ = w_probs.data.cpu().numpy()
                rand_num = np.random.rand(*probs_.shape)
                debuglogger.debug(f'rand_num: {rand_num}')
                debuglogger.debug(f'probs: {probs_}')
                w_binary = _Variable(torch.from_numpy(
                    (rand_num < probs_).astype('float32')))
            else:
                w_binary = torch.round(w_probs).detach()
            if w_probs.is_cuda:
                w_binary = w_binary.cuda()
            w_feats = w_binary
            debuglogger.debug(f'w_binary: {w_binary}')
        else:
            w_feats = w_scores
            w_probs = None

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
                 attn_dim):
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
        self.attn_dim = attn_dim
        self.image_processor = ImageProcessor(
            im_feat_dim, h_dim, use_attn, attn_dim)
        self.text_processor = TextProcessor(desc_dim, h_dim)
        self.message_processor = MessageProcessor(m_dim, h_dim)
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

    def initial_state(self, batch_size):
        return _Variable(torch.zeros(batch_size, self.h_dim))

    def forward(self, x, m, t, desc, use_message, batch_size, training):
        """
        Update State:
            h_z = message_processor(m, h_z)

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
        Output:
            s, s_probs: A STOP bit and its associated probability, indicating whether the agent has decided to make a selection. The conversation will continue until both agents have selected STOP.
            w, w_probs: A binary message and the probability of each bit in the message being ``1``.
            y: A prediction for each class described in the descriptions.
            r: An estimate of the reward the agent will receive
        """
        debuglogger.debug(f'Input sizes...')
        debuglogger.debug(f'x: {x.size()}')
        debuglogger.debug(f'm: {m.size()}')
        debuglogger.debug(f'desc: {desc.size()}')

        # Initialize hidden state if necessary
        if self.h_z is None:
            self.h_z = self.initial_state(batch_size)

        # Process message sent from the other agent
        self.h_z = self.message_processor(m, self.h_z, use_message)
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
            debuglogger.debug(f'rand_num: {rand_num}')
            debuglogger.debug(f'prob: {prob_}')
            s_binary = _Variable(torch.from_numpy(
                (rand_num < prob_).astype('float32')))
        else:
            # Infer decisions
            # TODO check re. old implementation
            s_binary = torch.round(s_prob).detach()
        debuglogger.debug(f'stop decisions: {s_binary.size()}')
        debuglogger.debug(f'stop decisions: {s_binary}')

        # Predict classes
        # desc_proc:    bs x num_classes x hid_dim
        # h_c:  bs x hid_dim
        # hid_cat_desc: (bs x num_classes) x (hid_dim * 2)
        h_c_expand = torch.unsqueeze(
            h_c, dim=1).expand(-1, self.num_classes, -1)
        debuglogger.debug(f'h_c_expand: {h_c_expand.size()}')
        debuglogger.debug(f'h_c: {h_c}')
        debuglogger.debug(f'h_c_expand: {h_c_expand}')
        hid_cat_desc = torch.cat([h_c_expand, desc_proc], dim=2)
        debuglogger.debug(f'hid_cat_desc: {hid_cat_desc.size()}')
        hid_cat_desc = hid_cat_desc.view(-1, self.h_dim * 2)
        debuglogger.debug(f'hid_cat_desc: {hid_cat_desc.size()}')
        y = F.relu(self.y1(hid_cat_desc))
        debuglogger.debug(f'y: {y.size()}')
        y = self.y2(y).view(batch_size, -1)
        debuglogger.debug(f'y: {y.size()}')
        # y: batch_size * num_classes
        y_scores = F.softmax(y, dim=1).detach()
        debuglogger.debug(f'y_scores: {y_scores.size()}')
        debuglogger.debug(f'y_scores: {y_scores}')

        # Generate message
        w, w_probs = self.message_generator(y_scores, h_c, desc_proc, training)
        debuglogger.debug(f'w: {w.size()}')
        debuglogger.debug(f'w_probs: {w_probs.size()}')

        return (s_binary, s_prob), (w, w_probs), y, r


if __name__ == "__main__":
    print("Testing agent init and forward pass...")
    im_feature_type = ""
    im_feat_dim = 128
    h_dim = 4
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
    agent = Agent(im_feature_type,
                  im_feat_dim,
                  h_dim,
                  m_dim,
                  desc_dim,
                  num_classes,
                  s_dim,
                  use_binary,
                  use_attn,
                  attn_dim)
    print(agent)
    x = _Variable(torch.ones(batch_size, im_feat_dim))
    m = _Variable(torch.ones(batch_size, m_dim))
    desc = _Variable(torch.ones(batch_size, num_classes, desc_dim))

    for i in range(2):
        s, w, y, r = agent(x, m, i, desc, use_message, batch_size, training)
        print(f's_binary: {s[0]}')
        print(f's_probs: {s[1]}')
        print(f'w_binary: {w[0]}')
        print(f'w_probs: {w[1]}')
        print(f's_binary: {s[0]}')
        print(f's_probs: {s[1]}')
        print(f'y: {y}')
        print(f'r: {r}')
