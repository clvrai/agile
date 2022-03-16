import torch
import numpy as np
from torch import nn

from gnn.GNN.model import GCN2 as GCN
from gnn.GAT import launcher
from value_based.commons.test import TestCase as test
from value_based.commons.args import SLATE_MASK_TOKEN


class ActionGraph(nn.Module):
    def __init__(self, dim_in: int = 40, dim_hidden: int = 32, dim_out: int = 1, args: dict = {}):
        super(ActionGraph, self).__init__()
        self._args = args
        self._dim_in = dim_in
        self._dim_hidden = dim_hidden
        self._dim_out = dim_out
        self._if_item_retrieval = self._args.get("if_item_retrieval", False)
        self._if_use_agg = self._args.get('use_agg_node', False)
        self._graph_type = self._args.get("graph_type", "gcn")
        self._aggregation_type = self._args.get("graph_aggregation_type", "sum")
        self._device = self._args.get("device", "cpu")
        self.pre_summarize_mlp = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden)
        ).to(device=self._device)
        self.post_summarize_mlp = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden)
        ).to(device=self._device)

        gnn = None
        if self._graph_type != "gap":  # Global Average Pooling
            if self._graph_type == "gcn":
                gnn = GCN(dim_in=dim_hidden,
                          dim_hidden=dim_hidden,
                          dim_out=dim_hidden,
                          args=args).to(device=self._device)
            elif self._graph_type.startswith("gat"):
                _gnn = launcher(model_name=self._args.get("graph_type", "gcn"))
                gnn = _gnn(dim_in=dim_hidden,
                           dim_hidden=dim_hidden,
                           dim_out=dim_hidden,
                           num_heads=self._args.get("graph_num_heads", 1),
                           args=args).to(device=self._device)
        self._gnn = gnn

    @property
    def gnn(self):
        return self._gnn

    @property
    def aggregation_type(self):
        return self._aggregation_type

    def _compute_scores(self, node_feat, user_feat):
        scores = np.asarray([np.dot(node_feat[i, :], user_feat[i, :]) for i in range(user_feat.shape[0])])
        return torch.tensor(scores)

    @staticmethod
    def get_global_attribute(_aggregation_type, node_feat):
        """ Returns the aggregated node features of Product Graph

        References
            - https://github.com/deepmind/graph_nets/blob/master/graph_nets/blocks.py#L213-L247

        Args:
            node_feat (torch.tensor): batch_size x num_nodes x dim_node

        Returns:
            _feat (torch.tensor): batch_size x dim_node
        """
        _, num_nodes, _ = node_feat.shape
        if _aggregation_type == "None" or _aggregation_type is None:
            return node_feat

        # batch_size x num_nodes x dim_node -> batch_size x dim_node
        if _aggregation_type == "sum":
            _feat = torch.sum(node_feat, dim=1)  # aggregate the features over nodes
        elif _aggregation_type == "mean":
            _feat = torch.mean(node_feat, dim=1)  # aggregate the features over nodes
        else:
            raise ValueError
        return _feat

    def reset_attention(self):
        if self._graph_type.startswith("gat"):
            self.gnn.reset_attention()

    def get_attention(self, first=True):
        """ Given the input, this returns the attention vector
        Returns:
             _weighted_a_input(torch.tensor): batch_size x num_nodes x num_nodes
        """
        if self._graph_type.startswith("gat"):
            return self.gnn.get_attention(stack=True, first=first)
        else:
            return None

    def get_attention_stats(self, first=True):
        """ Given the input, this returns the attention vector
        Returns:
             _weighted_a_input(torch.tensor): batch_size x num_nodes x num_nodes
        """
        if self._graph_type.startswith("gat"):
            return self.gnn.get_attention_stats(args=None, first=first)
        else:
            return None

    def run(self, if_e_node: bool = True):
        # Get the compact representation of node-features
        node_feat = self.pre_summarize_mlp(self.node_feat).squeeze(-1)

        if self._args["action_summarizer"] == 'gnn':
            z = self._gnn(node_feat, self.adj_mat).squeeze(-1)  # batch_size x num_nodes x dim_node
            summary_vec = self.get_global_attribute(_aggregation_type=self._aggregation_type,
                                                    node_feat=z)  # batch_size x dim_node
            summary_vec = self.post_summarize_mlp(summary_vec)  # batch_size x dim_node
            summary_vec = summary_vec[:, None, :].repeat(1, z.shape[1], 1)
        elif self._args["action_summarizer"] == 'lstm':
            raise NotImplementedError
        elif self._args["action_summarizer"] == 'deep_set':
            raise NotImplementedError
        else:
            raise NotImplementedError

        if self._graph_type.startswith("gat"):
            self._gnn.stack_attention()

        if if_e_node:  # AGILE
            node_feat = z
        return node_feat, summary_vec

    def load_graph(self, state: torch.tensor, action: torch.tensor):
        """ Prepare the input of GNN

        Args:
            state (torch.tensor): batch_step_size x num_candidates x dim_state
            action (torch.tensor): num_candidates x dim_item
        """
        batch_step_size, num_candidates, _ = state.shape
        num_candidates = num_candidates + self._if_use_agg

        # batch_step_size x (num_candidates + 1) x dim
        state = state[:, :1, :].repeat(1, num_candidates, 1)

        if self._if_use_agg:
            # batch_step_size x 1 x dim_item
            empty_tensor = torch.zeros(batch_step_size, 1, action.shape[-1], device=self._device)
            # batch_step_size x (num_candidates + 1) x dim_item
            action = torch.cat([action, empty_tensor], dim=1)  # Add the dummy node to the action nodes

        if self._args.get("summarizer_use_only_action", False):
            self.node_feat = action
        else:
            # batch_step_size x (num_candidates + 1) x (dim_state + dim_item)
            self.node_feat = torch.cat([state, action], dim=-1)

        # Assuming a complete graph; No self-loop
        self.adj_mat = torch.ones([batch_step_size, num_candidates, num_candidates], device=self._device)
        if self._args["graph_type"].lower().startswith('gcn'):
            self.adj_mat -= torch.eye(num_candidates, num_candidates, device=self._device)

        self.prep_slate_info(batch_step_size=batch_step_size, num_candidates=num_candidates)
        self.reset_attention()

    def prep_slate_info(self, batch_step_size, num_candidates):
        # self.current = torch.zeros(batch_step_size, num_candidates, device=self._device)
        # self.candidate = torch.ones(batch_step_size, num_candidates, device=self._device)
        self.candidate_neg = torch.zeros(batch_step_size, num_candidates, device=self._device)

    def rebuild_node_feat(self, slate_embed):
        if self._gnn is not None:
            pass
            # if self._args.get("node_type_dim", 5) > 0:
            #     raise ValueError
            #     # batch_step_size x num_candidates x node_type_dim
            #     _current_type = self.current * torch.tensor([-1] * self._args["node_type_dim"], device=self._device)
            #     _candidate_type = self.candidate * torch.tensor([1] * self._args["node_type_dim"], device=self._device)
            #     _node_type = _current_type + _candidate_type  # batch_step_size x num_candidates x node_type_dim
            #
            #     if self._if_use_agg:
            #         zero_type = torch.zeros([self.current.shape[0], 1, 2],
            #                                 device=self._device)  # batch_step_size x 1 x 2
            #         _node_type = torch.cat([_node_type, zero_type], dim=1)  # batch_step_size x (num_candidates + 1) x 2
            #     self.node_feat = torch.cat([self.node_feat, _node_type], dim=-1)

    def update_graph_info(self, action: torch.tensor, slate_index: int, if_call_from_update: bool = False):
        batch_step_size, num_candidates = self.candidate_neg.shape
        if if_call_from_update:
            pass
            # self.current = torch.zeros(batch_step_size, num_candidates, device=self._device)
            # self.candidate = torch.ones(batch_step_size, num_candidates, device=self._device)
            # if slate_index > 0:
            #     _temp = (
            #         torch.tensor([np.arange(batch_step_size)] * slate_index, device=self._device).t(), action)
            #     self.current = self.current.index_put(_temp, torch.tensor(1., device=self._device))
            #     self.candidate = self.candidate.index_put(_temp, torch.tensor(0., device=self._device))
        else:
            _base = torch.tensor(np.arange(batch_step_size), device=self._device)
            # self.current = self.current.index_put_((_base, action), torch.tensor(1., device=self._device))
            # self.candidate = self.candidate.index_put_((_base, action), torch.tensor(0., device=self._device))
            self.candidate_neg = self.candidate_neg.index_put_((_base, action),
                                                               torch.tensor(SLATE_MASK_TOKEN, device=self._device))
        if self._gnn is not None:
            self._update_adj_mat(action=action)

    def _update_adj_mat(self, action: np.ndarray):
        """ Remove the edges of selected nodes in the graph

        Args:
            action (np.ndaray): batch_size x intermediate_slate_size

        Returns:
            adj_mat (torch.tensor): batch_size x num_candidates x num_candidates
        """
        if self._args["agile_use_mask"]:
            # Disconnect edges from the taken action nodes
            # Ref: ./policy_based/method/models/main_method.py#L128-L130
            # adj_mat: batch_size x num_nodes x num_nodes
            self.adj_mat[np.arange(self.adj_mat.shape[0])[:, None], action] = 0
            self.adj_mat.permute([0, 2, 1])[np.arange(self.adj_mat.shape[0])[:, None], action] = 0


class Test(test):
    def __init__(self):
        self._get_args()
        # self.args.device = "cuda"
        # self.args.if_debug = True
        self.args.if_debug = False
        self.args.if_visualise_debug = True
        self.args.env_name = "recsim"
        self._prep()

    def test(self):
        self.args.action_summarizer = "gnn"

        # test main body
        for graph_type in [
            "gcn",
            "gat2",
            # "gap"
        ]:
            for graph_aggregation_type in ["sum", "mean"]:
                print(f"\n=== GNN type: {graph_type}, {graph_aggregation_type}")
                self.args.graph_type = graph_type
                self.args.graph_aggregation_type = graph_aggregation_type
                graph = ActionGraph(dim_in=self.args.dim_item * 2,
                                    dim_hidden=self.args.graph_dim_out,
                                    args=vars(self.args))
                print(graph)
                state = action = torch.randn(self.args.batch_size, self.args.num_candidates, self.args.dim_item)

                # load the features
                graph.load_graph(state=state, action=action)

                # message passing
                node_feat, summary_vec = graph.run()
                print(node_feat.shape, summary_vec.shape)


if __name__ == '__main__':
    Test().test()
