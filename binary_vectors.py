import torch

import sys
import h5py
import numpy as np

import gflags

FLAGS = gflags.FLAGS


def extract_binary(FLAGS, load_hdf5, exchange, dev_file, batch_size, epoch, shuffle, cuda, top_k,
                   sender, receiver, desc_dict, map_labels, file_name):
    sender_out_dim = FLAGS.sender_out_dim
    output_path = FLAGS.binary_output

    desc = desc_dict["desc"]
    desc_set = desc_dict.get("desc_set", None)
    desc_set_lens = desc_dict.get("desc_set_lens", None)

    # Create hdf5 binary vectors file
    bin_vec_file = h5py.File(output_path, "w")

    bin_vec_format = np.dtype([('ExampleId', np.str_, 50),
                               ('AgentId', np.str_, 1),
                               ('Index', 'i'),
                               ('Target', 'i'),
                               ('Rank', 'i'),
                               ('BinaryProb', np.float32, (sender_out_dim, )),
                               ('BinaryVec', np.float32, (sender_out_dim, ))])
    bin_vec_communication = bin_vec_file.create_dataset("Communication",
                                                        (0, ), maxshape=(None, ), dtype=bin_vec_format)

    # Create hdf5 predictions file
    preds_format = np.dtype([('ExampleId', np.str_, 50),
                             ('AgentId', np.str_, 1),
                             ('Index', 'i'),
                             ('Target', 'i'),
                             ('Rank', 'i'),
                             ('Predictions', np.float32, (len(desc), )),
                             ('StopProb', np.float32, (1, )),
                             ('StopVec', np.float32, (1, )),
                             ('StopMask', np.float32, (1, )),
                             ])
    preds_communication = bin_vec_file.create_dataset("Predictions",
                                                      (0, ), maxshape=(None, ), dtype=preds_format)

    # Load development images
    dev_loader = load_hdf5(dev_file, batch_size, epoch, shuffle,
                           truncate_final_batch=True, map_labels=map_labels)

    for batch in dev_loader:
        # Extract images and targets

        target = batch["target"]
        data = batch[FLAGS.img_feat]
        example_ids = batch["example_ids"]
        batch_size = target.size(0)

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

        s, sen_w, rec_w, y, bs, br = exchange(
            sender, receiver, None, None, exchange_args)

        s_masks, s_feats, s_probs = s
        sen_feats, sen_probs = sen_w
        rec_feats, rec_probs = rec_w

        # TODO: Use masks. This can be tricky!
        timesteps = zip(sen_feats, sen_probs, rec_feats,
                        rec_probs, y, s_feats, s_probs, s_masks)

        for i_exchange, (_z_binary, _z_probs, _w_binary, _w_probs, _y, _s_feats, _s_probs, _s_masks) in enumerate(timesteps):

            i_exchange_batch = np.full(batch_size, i_exchange, dtype=int)

            # Extract predictions and rank of target class.
            np_preds = _y.data.cpu().numpy()
            nclasses = np_preds.shape[1]
            target_set = set(target.tolist())
            assert len(
                target_set) == 1, "Rank only works if there is one target"
            single_target = target[0]
            np_rank = np.abs(np_preds.argsort(1) - nclasses)[:, single_target]

            # Store Sender binary features and probabilities locally
            np_agent_ids = np.full(batch_size, 'S', dtype=np.dtype('S1'))
            np_index_sen = i_exchange_batch * 2
            np_target = target.cpu().numpy()
            np_probs = _z_probs.data.cpu().numpy()
            np_bin_vec = _z_binary.data.cpu().numpy()
            zipped = zip(example_ids, np_agent_ids, np_index_sen,
                         np_target, np_rank, np_probs, np_bin_vec)
            bin_vec_communication.resize(
                bin_vec_communication.shape[0] + batch_size, axis=0)
            try:
                bin_vec_communication[-batch_size:] = zipped
            except:
                import ipdb
                ipdb.set_trace()

            # Store Receiver binary features and probabilities locally
            np_agent_ids = np.full(batch_size, 'R', dtype=np.dtype('S1'))
            np_index_rec = i_exchange_batch * 2 + 1
            np_probs = _w_probs.data.cpu().numpy()
            np_bin_vec = _w_binary.data.cpu().numpy()
            np_s_feats = _s_feats.data.cpu().numpy()
            np_s_probs = _s_probs.data.cpu().numpy()
            np_s_masks = _s_masks.data.cpu().numpy()
            zipped = zip(example_ids, np_agent_ids, np_index_rec,
                         np_target, np_rank, np_probs, np_bin_vec)
            bin_vec_communication.resize(
                bin_vec_communication.shape[0] + batch_size, axis=0)
            bin_vec_communication[-batch_size:] = zipped
            # Store Receiver's prediction scores locally
            zipped = zip(example_ids, np_agent_ids, np_index_rec, np_target,
                         np_rank, np_preds, np_s_probs, np_s_feats, np_s_masks)
            preds_communication.resize(
                preds_communication.shape[0] + batch_size, axis=0)
            preds_communication[-batch_size:] = zipped
