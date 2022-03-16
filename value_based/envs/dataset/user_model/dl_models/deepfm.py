import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, num_feat, num_field, reg_l1=0.01, reg_l2=0.01, embedding_size=16):
        super().__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2  # L1/L2
        self.num_feat = num_feat  # denote as M
        self.num_field = num_field  # denote as F
        self.embedding_size = embedding_size  # denote as K

        # first order term parameters embedding
        self.first_weights = nn.Embedding(num_feat, 1)  # None * M * 1
        nn.init.xavier_uniform_(self.first_weights.weight)
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)

        self.feat_embeddings = nn.Embedding(num_feat, embedding_size)  # None * M * K
        nn.init.xavier_uniform_(self.feat_embeddings.weight)

    def forward(self, feat_index, feat_value):
        feat_value = torch.unsqueeze(feat_value, dim=2)  # None * F * 1
        first_weights = self.first_weights(feat_index)  # None * F * 1
        first_weight_value = torch.mul(first_weights, feat_value)  # None * F * 1
        first_weight_value = torch.squeeze(first_weight_value, dim=2)  # None * F
        y_first_order = torch.sum(first_weight_value, dim=1)  # None

        # Step2: 再计算二阶部分
        secd_feat_emb = self.feat_embeddings(feat_index)  # None * F * K
        feat_emd_value = torch.mul(secd_feat_emb, feat_value)  # None * F * K(广播)

        # sum_square part
        summed_feat_emb = torch.sum(feat_emd_value, 1)  # None * K
        interaction_part1 = torch.pow(summed_feat_emb, 2)  # None * K

        # squared_sum part
        squared_feat_emd_value = torch.pow(feat_emd_value, 2)  # None * K
        interaction_part2 = torch.sum(squared_feat_emd_value, dim=1)  # None * K

        y_secd_order = 0.5 * torch.sub(interaction_part1, interaction_part2)
        y_secd_order = torch.sum(y_secd_order, dim=1)

        output = self.bias + y_first_order + y_secd_order
        output = torch.unsqueeze(output, dim=1)
        return output