""" Ref: https://github.com/rushhan/Generative-Adversarial-User-Model-for-Reinforcement-Learning-Based-Recommendation-System-Pytorch """
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn
import numpy as np


def mlp(x_shape, dim_hiddens, output_dim, sd, act_last=False):
    dim_hiddens = tuple(map(int, dim_hiddens.split("-")))
    cur = x_shape
    main_mod = nn.Sequential()
    for i, h in enumerate(dim_hiddens):
        main_mod.add_module('Linear-{0}'.format(i), torch.nn.Linear(cur, h))
        main_mod.add_module('act-{0}'.format(i), nn.ELU())
        cur = h

    if act_last:
        main_mod.add_module("Linear_last", torch.nn.Linear(cur, output_dim))
        main_mod.add_module("act_last", nn.ELU())
        return main_mod
    else:
        main_mod.add_module("linear_last", torch.nn.Linear(cur, output_dim))
        return main_mod


class UserModelPW(nn.Module):
    """ GAN PW in CDQN paper """

    def __init__(self, f_dim, args):
        super(UserModelPW, self).__init__()
        self.f_dim = f_dim
        self.dim_hiddens = args.dims
        self.lr = args.learning_rate

        """ Position Weight matrix(eq(4) in Sec4.2): m x n
            - m(pw_dim): a fixed number of historical steps
            - n(band_size): a set of importance weights on positions
        """
        self.pw_dim = args.pw_dim
        self.band_size = args.pw_band_size

        self.mlp_model = mlp(self.f_dim * self.pw_dim + self.f_dim, self.dim_hiddens, 1, 1e-3, act_last=False)

    def forward(self, inputs, is_train=False, index=None):
        # input is a dictionary
        if is_train == True:
            disp_current_feature = torch.tensor(inputs['disp_current_feature_x'])

            """ Xs_clicked: (num_sessions x item_feat_dim)
                In each session, a click must happen so that num of sessions is equal to num of clicks
                ** they segment the sessions by click so that this happened....
            """
            Xs_clicked = torch.tensor(inputs['feature_clicked_x'])

            item_size = torch.tensor(inputs['news_cnt_short_x'])
            section_length = torch.tensor(inputs['sec_cnt_x'])
            click_values = torch.tensor(np.ones(len(inputs['click_2d_x']), dtype=np.float32))  # click has a value of 1
            click_indices = torch.tensor(inputs['click_2d_x'])
            disp_indices = torch.tensor(np.array(inputs['disp_2d_x']))
            disp_2d_split_sec_ind = torch.tensor(inputs['disp_2d_split_sec'])
            cumsum_tril_indices = torch.tensor(inputs['tril_indice'])
            cumsum_tril_value_indices = torch.tensor(np.array(inputs['tril_value_indice'], dtype=np.int64))
            click_2d_subindex = torch.tensor(inputs['click_sub_index_2d'])
        else:
            # define the inputs for val/tst here
            disp_current_feature = torch.tensor(inputs['feature_v'][index])
            Xs_clicked = torch.tensor(inputs['feature_clicked_v'][index])
            item_size = torch.tensor(inputs['news_cnt_short_v'][index])
            section_length = torch.tensor(inputs['sec_cnt_v'][index])
            click_values = torch.tensor(np.ones(len(inputs['click_2d_v'][index]), dtype=np.float32))
            click_indices = torch.tensor(inputs['click_2d_v'][index])
            disp_indices = torch.tensor(np.array(inputs['disp_2d_v'][index]))
            disp_2d_split_sec_ind = torch.tensor(inputs['disp_2d_split_sec_v'][index])
            cumsum_tril_indices = torch.tensor(inputs['tril_ind_v'][index])
            cumsum_tril_value_indices = torch.tensor(np.array(inputs['tril_value_ind_v'][index], dtype=np.int64))
            click_2d_subindex = torch.tensor(inputs['click_sub_index_2d_v'][index])

        denseshape = [section_length, item_size]  # this wont work

        click_history = [[] for _ in range(self.pw_dim)]
        for ii in range(self.pw_dim):
            # Array with the size of band_size
            position_weight = torch.ones(size=[self.band_size]).to(dtype=torch.float64) * 0.0001

            """ Gather the values from the entries of position weight array according to cumsum_tril_value_indices.
                An array with the size of cumsum_tril_value_indices
            """
            cumsum_tril_value = position_weight[cumsum_tril_value_indices]

            """ SparseTensor is a bit tricky, I'd recommend going through a doc beforehand.
                
                ** PyTorch's doc sucks.... so that you should check TF first, then check Torch to understand
                   the spec of their API
                - Pytorch: https://pytorch.org/docs/stable/sparse.html#torch.sparse.FloatTensor.mul
                - TensorFlow: https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor
            """
            # num_session x num_session
            cumsum_tril_matrix = torch.sparse.FloatTensor(cumsum_tril_indices.t(), cumsum_tril_value,
                                                          [section_length, section_length]).to_dense()

            # (num_session x num_session) x (num_session x item_feat_dim): num_session x item_feat_dim
            click_history[ii] = torch.matmul(cumsum_tril_matrix,
                                             Xs_clicked.to(dtype=torch.float64))  # Xs_clicked: section by _f_dim

        # (m x num_session x item_feat_dim) -> (num_session x item_feat_dim)
        concat_history = torch.cat(click_history, axis=1)
        disp_history_feature = concat_history[disp_2d_split_sec_ind].float()
        # print(disp_2d_split_sec_ind)
        # print(concat_history.shape, len(click_history), click_history[0].shape, disp_2d_split_sec_ind.shape)

        # (4) combine features
        concat_disp_features = torch.reshape(torch.cat([disp_history_feature, disp_current_feature], axis=1),
                                             [-1, self.f_dim * self.pw_dim + self.f_dim])
        # print(disp_history_feature.shape, disp_current_feature.shape, concat_disp_features.shape)

        # (5) compute utility
        u_disp = self.mlp_model(concat_disp_features.float())
        # print(u_disp.shape)

        # (5)
        exp_u_disp = torch.exp(u_disp)

        sum_exp_disp_ubar_ut = segment_sum(exp_u_disp, disp_2d_split_sec_ind)  # 146 x 1
        sum_click_u_bar_ut = u_disp[click_2d_subindex]  # 146 x 1

        # (6) loss and precision
        click_tensor = torch.sparse.FloatTensor(click_indices.t(), click_values, denseshape).to_dense()
        click_cnt = click_tensor.sum(1)
        loss_sum = torch.sum(- sum_click_u_bar_ut + torch.log(sum_exp_disp_ubar_ut + 1))
        event_cnt = torch.sum(click_cnt)
        loss = loss_sum / event_cnt

        exp_disp_ubar_ut = torch.sparse.FloatTensor(disp_indices.t(), torch.reshape(exp_u_disp, (-1,)), denseshape)
        # print(exp_disp_ubar_ut.shape, denseshape)

        dense_exp_disp_util = exp_disp_ubar_ut.to_dense()
        argmax_click = torch.argmax(click_tensor, dim=1)
        argmax_disp = torch.argmax(dense_exp_disp_util, dim=1)

        top_2_disp = torch.topk(dense_exp_disp_util, k=2, sorted=False)[1]

        precision_1_sum = torch.sum((torch.eq(argmax_click, argmax_disp)))
        precision_1 = precision_1_sum / event_cnt

        precision_2_sum = (torch.eq(argmax_click[:, None].to(torch.int64), top_2_disp.to(torch.int64))).sum()
        precision_2 = precision_2_sum / event_cnt

        # self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * 0.05  # regularity
        # weight decay can be added in the optimizer for l2 decay
        return loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt


def segment_sum(data, segment_ids):
    """
    Analogous to tf.segment_sum (https://www.tensorflow.org/api_docs/python/tf/math/segment_sum).

    :param data: A pytorch tensor of the data for segmented summation.
    :param segment_ids: A 1-D tensor containing the indices for the segmentation.
    :return: a tensor of the same type as data containing the results of the segmented summation.
    """
    if not all(segment_ids[i] <= segment_ids[i + 1] for i in range(len(segment_ids) - 1)):
        raise AssertionError("elements of segment_ids must be sorted")

    if len(segment_ids.shape) != 1:
        raise AssertionError("segment_ids have be a 1-D tensor")

    if data.shape[0] != segment_ids.shape[0]:
        raise AssertionError("segment_ids should be the same size as dimension 0 of input.")

    num_segments = len(torch.unique(segment_ids))
    return unsorted_segment_sum(data, segment_ids, num_segments)


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape).scatter_add(0, segment_ids, data.float())
    tensor = tensor.type(data.dtype)
    return tensor
