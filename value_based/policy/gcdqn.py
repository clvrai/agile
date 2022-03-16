import os
import torch
import numpy as np
from torch import nn
from typing import Dict, List

from value_based.policy.agent import Agent
from gnn.GNN.model import GCN2 as GCN
from gnn.DeepSet.models import DeepSet, BiLSTM
from gnn.GAT import launcher
from value_based.policy.architecture.Qnet import QNetwork
from value_based.commons.utils import logging
from value_based.encoder.slate_encoder import SlateEncoder
from value_based.encoder.obs_encoder import ObsEncoder
from value_based.encoder.act_encoder import ActEncoder
from value_based.commons.optimiser_factory import OptimiserFactory
from value_based.commons.pt_check_grad import check_grad
from value_based.embedding.base import BaseEmbedding
from value_based.policy.agent import Test as test
from gnn.action_graph import ActionGraph

# SLATE_MASK_TOKEN = - np.infty
SLATE_MASK_TOKEN = - 1000000000.0


class GCDQN(Agent):
    def __init__(self,
                 obs_encoder: List[ObsEncoder] = None,
                 slate_encoder: List[SlateEncoder] = None,
                 dict_embedding: Dict[str, BaseEmbedding] = None,
                 act_encoder: List[ActEncoder] = None,
                 args: dict = {}):
        super(GCDQN, self).__init__(obs_encoder=obs_encoder,
                                    slate_encoder=slate_encoder,
                                    dict_embedding=dict_embedding,
                                    act_encoder=act_encoder,
                                    args=args)
        self._use_agg = self._args.get('use_agg_node', False)
        self._use_slate = self._args.get('gcdqn_use_slate_soFar', False)

        self.main_pre_summarize_mlp = self.target_pre_summarize_mlp = self.main_post_summarize_mlp = \
            self.target_post_summarize_mlp = self.main_Q_net = self.target_Q_net = None

        self._optim_factory = OptimiserFactory(if_single_optimiser=self._if_single_optimiser,
                                               if_one_shot_instantiation=self._if_one_shot_instantiation)
        self._optim_factory.add_params(params_dict={"params": self.main_action_linear.parameters(), "lr": self._lr})

        # for SRL-based CDQN
        if self._args["agent_standardRL_type"] == "2":
            self.available_mask_summarizer = nn.Linear(
                self._num_trainItems, self._args["dim_item_hidden"]).to(self._device)
            self._optim_factory.add_params(
                params_dict={"params": self.available_mask_summarizer.parameters(), "lr": self._lr})

        if not self._args["gcdqn_no_graph"]:  # Ablation; no GNN is equivalent to CDQN-share
            if self._args["action_summarizer"] == 'gnn':
                # Define GCNs for main and target Q networks.
                if self._args.get("graph_type", "gcn") == 'gcn':
                    gnn_choice = GCN
                elif self._args.get("graph_type", "gcn").startswith("gat"):
                    gnn_choice = launcher(model_name=self._args.get("graph_type", "gcn"))
            elif self._args["action_summarizer"] == 'deep_set':
                assert self._args["graph_type"] == "None"
                gnn_choice = DeepSet
            elif self._args["action_summarizer"] in ['lstm', 'bilstm']:
                assert self._args["graph_type"] == "None"
                gnn_choice = BiLSTM
            else:
                raise ValueError('graph type not found')

            if self._args["gcdqn_singleGAT"]:
                self.main_Q_net = self.target_Q_net = gnn_choice(dim_in=self._args["main_Q_net_dim_in"],
                                                                 dim_hidden=self._args.get("graph_dim_hidden", 32),
                                                                 dim_out=self._args.get("gcdqn_dim_out", 32),
                                                                 num_heads=self._args.get("graph_gat_num_heads", 1),
                                                                 args=args).to(device=self._device)
            else:
                if self._args["gcdqn_twin_GAT"]:
                    self.main_Q_net_node = gnn_choice(dim_in=self._args["main_Q_net_dim_in"],
                                                      dim_hidden=self._args.get("graph_dim_hidden", 32),
                                                      dim_out=self._args.get("gcdqn_dim_out", 32),
                                                      num_heads=self._args.get("graph_gat_num_heads", 1),
                                                      args=args).to(device=self._device)
                    self.target_Q_net_node = gnn_choice(dim_in=self._args["main_Q_net_dim_in"],
                                                        dim_hidden=self._args.get("graph_dim_hidden", 32),
                                                        dim_out=self._args.get("gcdqn_dim_out", 32),
                                                        num_heads=self._args.get("graph_gat_num_heads", 1),
                                                        args=args).to(device=self._device)
                    self.main_Q_net_summary = gnn_choice(dim_in=self._args["main_Q_net_dim_in"],
                                                         dim_hidden=self._args.get("graph_dim_hidden", 32),
                                                         dim_out=self._args.get("gcdqn_dim_out", 32),
                                                         num_heads=self._args.get("graph_gat_num_heads", 1),
                                                         args=args).to(device=self._device)
                    self.target_Q_net_summary = gnn_choice(dim_in=self._args["main_Q_net_dim_in"],
                                                           dim_hidden=self._args.get("graph_dim_hidden", 32),
                                                           dim_out=self._args.get("gcdqn_dim_out", 32),
                                                           num_heads=self._args.get("graph_gat_num_heads", 1),
                                                           args=args).to(device=self._device)
                    self.target_Q_net_node.load_state_dict(self.main_Q_net_node.state_dict())
                    self.target_Q_net_summary.load_state_dict(self.main_Q_net_summary.state_dict())
                    self._optim_factory.add_params({"params": self.main_Q_net_node.parameters(), "lr": self._lr})
                    self._optim_factory.add_params({"params": self.main_Q_net_summary.parameters(), "lr": self._lr})
                else:
                    self.main_Q_net = gnn_choice(dim_in=self._args["main_Q_net_dim_in"],
                                                 dim_hidden=self._args.get("graph_dim_hidden", 32),
                                                 dim_out=self._args.get("gcdqn_dim_out", 32),
                                                 num_heads=self._args.get("graph_gat_num_heads", 1),
                                                 args=args).to(device=self._device)
                    self.target_Q_net = gnn_choice(dim_in=self._args["main_Q_net_dim_in"],
                                                   dim_hidden=self._args.get("graph_dim_hidden", 32),
                                                   dim_out=self._args.get("gcdqn_dim_out", 32),
                                                   num_heads=self._args.get("graph_gat_num_heads", 1),
                                                   args=args).to(device=self._device)
                    self.target_Q_net.load_state_dict(self.main_Q_net.state_dict())
                    self._optim_factory.add_params(params_dict={"params": self.main_Q_net.parameters(), "lr": self._lr})

        if self._args["action_summarizer"] not in ["", "None"]:  # AGILE
            if not args["summarizer_use_only_action"] and args["gcdqn_use_pre_summarise_mlp"]:
                if args["if_pre_summarize_linear"]:
                    if self._args["gcdqn_twin_GAT"]:
                        self.main_pre_summarize_mlp_node = nn.Linear(
                            self._dim_in, self._args.get("graph_dim_hidden", 32)).to(device=self._device)
                        self.target_pre_summarize_mlp_node = nn.Linear(
                            self._dim_in, self._args.get("graph_dim_hidden", 32)).to(device=self._device)
                        self.main_pre_summarize_mlp_summary = nn.Linear(
                            self._dim_in, self._args.get("graph_dim_hidden", 32)).to(device=self._device)
                        self.target_pre_summarize_mlp_summary = nn.Linear(
                            self._dim_in, self._args.get("graph_dim_hidden", 32)).to(device=self._device)

                        # Share the weights and register to the optimiser
                        self.target_pre_summarize_mlp_node.load_state_dict(
                            self.main_pre_summarize_mlp_node.state_dict())
                        self.target_pre_summarize_mlp_summary.load_state_dict(
                            self.main_pre_summarize_mlp_summary.state_dict())
                        self._optim_factory.add_params(
                            params_dict={"params": self.main_pre_summarize_mlp_node.parameters(), "lr": self._lr})
                        self._optim_factory.add_params(
                            params_dict={"params": self.main_pre_summarize_mlp_summary.parameters(), "lr": self._lr})
                    else:
                        self.main_pre_summarize_mlp = nn.Linear(
                            self._dim_in, self._args.get("graph_dim_hidden", 32)).to(device=self._device)
                        self.target_pre_summarize_mlp = nn.Linear(
                            self._dim_in, self._args.get("graph_dim_hidden", 32)).to(device=self._device)
                        # Share the weights and register to the optimiser
                        self.target_pre_summarize_mlp.load_state_dict(self.main_pre_summarize_mlp.state_dict())
                        self._optim_factory.add_params(
                            params_dict={"params": self.main_pre_summarize_mlp.parameters(), "lr": self._lr})
                else:
                    if self._args["gcdqn_twin_GAT"]:
                        # Summarisers before and after GAT
                        self.main_pre_summarize_mlp_node = nn.Sequential(
                            nn.Linear(self._dim_in, self._args.get("graph_dim_hidden", 32)),
                            nn.ReLU(),
                            nn.Linear(self._args.get("graph_dim_hidden", 32), self._args.get("graph_dim_hidden", 32))
                        ).to(device=self._device)
                        self.target_pre_summarize_mlp_node = nn.Sequential(
                            nn.Linear(self._dim_in, self._args.get("graph_dim_hidden", 32)),
                            nn.ReLU(),
                            nn.Linear(self._args.get("graph_dim_hidden", 32), self._args.get("graph_dim_hidden", 32))
                        ).to(device=self._device)
                        self.main_pre_summarize_mlp_summary = nn.Sequential(
                            nn.Linear(self._dim_in, self._args.get("graph_dim_hidden", 32)),
                            nn.ReLU(),
                            nn.Linear(self._args.get("graph_dim_hidden", 32), self._args.get("graph_dim_hidden", 32))
                        ).to(device=self._device)
                        self.target_pre_summarize_mlp_summary = nn.Sequential(
                            nn.Linear(self._dim_in, self._args.get("graph_dim_hidden", 32)),
                            nn.ReLU(),
                            nn.Linear(self._args.get("graph_dim_hidden", 32), self._args.get("graph_dim_hidden", 32))
                        ).to(device=self._device)

                        # Share the weights and register to the optimiser
                        self.target_pre_summarize_mlp_node.load_state_dict(
                            self.main_pre_summarize_mlp_node.state_dict())
                        self.target_pre_summarize_mlp_summary.load_state_dict(
                            self.main_pre_summarize_mlp_summary.state_dict())
                        self._optim_factory.add_params(
                            params_dict={"params": self.main_pre_summarize_mlp_node.parameters(), "lr": self._lr})
                        self._optim_factory.add_params(
                            params_dict={"params": self.main_pre_summarize_mlp_summary.parameters(), "lr": self._lr})
                    else:
                        # Summarisers before and after GAT
                        self.main_pre_summarize_mlp = nn.Sequential(
                            nn.Linear(self._dim_in, self._args.get("graph_dim_hidden", 32)),
                            nn.ReLU(),
                            nn.Linear(self._args.get("graph_dim_hidden", 32), self._args.get("graph_dim_hidden", 32))
                        ).to(device=self._device)
                        self.target_pre_summarize_mlp = nn.Sequential(
                            nn.Linear(self._dim_in, self._args.get("graph_dim_hidden", 32)),
                            nn.ReLU(),
                            nn.Linear(self._args.get("graph_dim_hidden", 32), self._args.get("graph_dim_hidden", 32))
                        ).to(device=self._device)

                        # Share the weights and register to the optimiser
                        self.target_pre_summarize_mlp.load_state_dict(self.main_pre_summarize_mlp.state_dict())
                        self._optim_factory.add_params(
                            params_dict={"params": self.main_pre_summarize_mlp.parameters(), "lr": self._lr})

            if not self._args["gcdqn_no_summary"] and self._args["gcdqn_use_post_summarise_mlp"]:
                if args["if_post_summarize_linear"]:
                    self.main_post_summarize_mlp = nn.Linear(
                        self._args.get("gcdqn_dim_out", 32), self._args.get("graph_dim_hidden", 32)
                    ).to(device=self._device)
                    self.target_post_summarize_mlp = nn.Linear(
                        self._args.get("gcdqn_dim_out", 32), self._args.get("graph_dim_hidden", 32)
                    ).to(device=self._device)
                else:
                    self.main_post_summarize_mlp = nn.Sequential(
                        nn.Linear(self._args.get("gcdqn_dim_out", 32), self._args.get("graph_dim_hidden", 32)),
                        nn.ReLU(),
                        nn.Linear(self._args.get("graph_dim_hidden", 32), self._args.get("graph_dim_hidden", 32))
                    ).to(device=self._device)
                    self.target_post_summarize_mlp = nn.Sequential(
                        nn.Linear(self._args.get("gcdqn_dim_out", 32), self._args.get("graph_dim_hidden", 32)),
                        nn.ReLU(),
                        nn.Linear(self._args.get("graph_dim_hidden", 32), self._args.get("graph_dim_hidden", 32))
                    ).to(device=self._device)

                # Share the weights and register to the optimiser
                self.target_post_summarize_mlp.load_state_dict(self.main_post_summarize_mlp.state_dict())
                self._optim_factory.add_params(
                    params_dict={"params": self.main_post_summarize_mlp.parameters(), "lr": self._lr})

        # you can't choose both
        assert not all([self._args["gcdqn_skip_connection"], self._args["gcdqn_simple_skip_connection"]])
        if self._args["gcdqn_no_graph"] or self._args["gcdqn_skip_connection"] or self._args[
            "gcdqn_simple_skip_connection"]:
            self._dim_out = 1 if self._if_sequentialQNet else self._num_trainItems
            self.final_main_Q_net = QNetwork(dim_in=self._args["skip_connection_dim_in"],
                                             dim_hiddens=self._dim_hidden,
                                             dim_out=self._dim_out).to(device=self._device)
            self.final_target_Q_net = QNetwork(dim_in=self._args["skip_connection_dim_in"],
                                               dim_hiddens=self._dim_hidden,
                                               dim_out=self._dim_out).to(device=self._device)

        if self._args["gcdqn_no_graph"] or self._args["gcdqn_skip_connection"] or self._args[
            "gcdqn_simple_skip_connection"]:
            self.final_target_Q_net.load_state_dict(self.final_main_Q_net.state_dict())
            self._optim_factory.add_params(params_dict={"params": self.final_main_Q_net.parameters(), "lr": self._lr})

        if self.main_obs_encoder is not None:
            if self.main_obs_encoder.encoder is not None:
                if self._if_debug:
                    logging("GCDQN>> Use obs_encoder")
                self._optim_factory.add_params({"params": self.main_obs_encoder.encoder.parameters(), "lr": self._lr})

        if self.main_slate_encoder is not None:
            if self.main_slate_encoder.encoder is not None:
                if self._if_debug:
                    logging("GCDQN>> Use slate_encoder")
                self._optim_factory.add_params({"params": self.main_slate_encoder.encoder.parameters(), "lr": self._lr})

        # Instantiate the optimiser(s)
        self.optimiser_list = self._optim_factory.get_optimiser()
        if self._args["if_use_lr_scheduler"]:
            lambda1 = lambda epoch: self._args["lr_scheduler_alpha"] ** epoch
            self.lr_scheduler_list = [
                torch.optim.lr_scheduler.LambdaLR(self.optimiser_list[i], lr_lambda=lambda1)
                for i in range(len(self.optimiser_list))
            ]
        del self._optim_factory
        if self._if_debug: logging("GCDQN>> {}".format(self.main_Q_net))

    def _build_nodes(self, init_nodes, current, candidate):
        """ Initialise the nodes in a graph

        Args:
            init_nodes (torch.tensor): batch_step_size x (num_candidates + 1) x (dim_state + dim_item)
            current (torch.tensor): batch_step_size x num_candidates x 1
            candidate (torch.tensor): batch_step_size x num_candidates x 1
        """
        if self._args.get("node_type_dim", 5) > 0:
            current_type = current * torch.tensor(
                [-1] * self._args.get("node_type_dim", 5),
                device=self._device)  # batch_step_size x num_candidates x node_type_dim
            candidate_type = candidate * torch.tensor(
                [1] * self._args.get("node_type_dim", 5),
                device=self._device)  # batch_step_size x num_candidates x node_type_dim
            type = current_type + candidate_type  # batch_step_size x num_candidates x node_type_dim

            if self._use_agg:
                zero_type = torch.zeros([current.shape[0], 1, 2], device=self._device)  # batch_step_size x 1 x 2
                type = torch.cat([type, zero_type], dim=1)  # batch_step_size x (num_candidates + 1) x 2
            nodes = torch.cat([init_nodes, type], dim=-1)
        else:
            nodes = init_nodes
        return nodes

    def _initialize_settings(self, inputs, batch_step_size, num_candidates):
        """ Prepare the input of GNN

        Args:
            inputs (list):
                if self._if_sequentialQNet:
                    if agent_standardRL_type == None:
                        inputs: (state_embed, item_embed) -> batch_step_size x num_candidates x dim_item_hidden
                    else:  -> this shouldn't be used
                        inputs: (state_embed, item_embed, availability_mask)
                            - state_embed: batch_step_size x num_trainItems x dim_state
                            - item_embed: batch_step_size x num_trainItems x dim_item_hidden
                            - availability_mask: batch_step_size x num_trainItems
                else:
                    if agent_standardRL_type != None:
                        inputs: (state_embed, item_embed) -> batch_step_size x dim
                    else:
                        inputs: (state_embed, item_embed, availability_mask)
                            - state_embed: batch_step_size x num_trainItems x dim_state
                            - item_embed: batch_step_size x num_trainItems x dim_item_hidden
                            - availability_mask: batch_step_size x num_trainItems
            batch_step_size (int):
            num_candidates (int):
        """
        if self._if_sequentialQNet:
            # batch_step_size x (num_candidates + 1) x dim
            state_embed = inputs[0][:, :1, :].repeat(1, num_candidates + self._use_agg, 1)
        else:
            state_embed = inputs[0]

        if self._args["agent_standardRL_type"] == "None":
            nodes = inputs[1]  # item_embed; batch_step_size x num_candidates x dim_item
            if self._use_agg:
                _, _, item_dim = inputs[1].shape  # Use the shape of item_embed
                empty_tensor = torch.zeros(batch_step_size, 1, item_dim,
                                           device=self._device)  # batch_step_size x 1 x dim_item
                # batch_step_size x (num_candidates + 1) x dim_item
                nodes = torch.cat([nodes, empty_tensor], dim=1)  # Empty state is at the end

            # batch_step_size x (num_candidates + 1) x (dim_state + dim_item)
            nodes = torch.cat([state_embed, nodes], dim=-1)  # nodes: state-action
        else:
            if self._args["agent_standardRL_type"] == "1":
                # SRL1 CDQN doesn't need item-embedding
                nodes = state_embed  # nodes: state-action
            else:
                # SRL2 CDQN uses the availability-mask
                mask = inputs[1]
                nodes = torch.cat([state_embed, mask], dim=-1)  # nodes: state-mask

        # Assuming a complete graph; No self-loop
        if self._args.get('gcdqn_empty_graph', False):
            adj_mat = torch.zeros([batch_step_size, num_candidates + self._use_agg, num_candidates + self._use_agg],
                                  device=self._device)
        else:
            adj_mat = torch.ones([batch_step_size, num_candidates + self._use_agg, num_candidates + self._use_agg],
                                 device=self._device)
            if self._args.get("graph_type", "gcn").lower().startswith('gcn'):
                adj_mat -= torch.eye(num_candidates + self._use_agg, num_candidates + self._use_agg,
                                     device=self._device)

        current = torch.zeros(batch_step_size, num_candidates, 1, device=self._device)
        candidate = torch.ones(batch_step_size, num_candidates, 1, device=self._device)
        candidate_neg = torch.zeros(batch_step_size, num_candidates, 1, device=self._device)

        return nodes, adj_mat, current, candidate, candidate_neg

    def remove_selected_nodes(self, slate_soFar: np.ndarray or None, adj_mat: torch.tensor):
        """ Remove the edges of selected nodes in the graph

        Args:
            slate_soFar (np.ndaray): batch_size x intermediate_slate_size
            adj_mat (torch.tensor): batch_size x num_candidates x num_candidates

        Returns:
            adj_mat (torch.tensor): batch_size x num_candidates x num_candidates
        """
        if self._args["gcdqn_use_mask"]:
            if slate_soFar is not None:
                # Disconnect edges from the taken action nodes
                # Ref: ./policy_based/method/models/main_method.py#L128-L130
                # adj_mat: batch_size x num_nodes x num_nodes
                adj_mat[np.arange(adj_mat.shape[0])[:, None], slate_soFar] = 0
                adj_mat.permute([0, 2, 1])[np.arange(adj_mat.shape[0])[:, None], slate_soFar] = 0
        return adj_mat

    def get_action_summary(self,
                           node_feat: torch.tensor,
                           action: torch.tensor = None,
                           adj_mat: torch.tensor = None,
                           _main_or_target: str = "main"):
        """
        Args:
            node_feat (torch.tensor): batch_size x num_nodes x dim_node
            action (torch.tensor): batch_size x num_nodes x dim_item
            adj_mat (torch.tensor):  batch_size x num_nodes x num_nodes

        Returns:
            node_feat (torch.tensor): batch_size x num_nodes x dim_node
            summary_vec (torch.tensor): batch_size x num_nodes x dim_node
        """
        if self._args["summarizer_use_only_action"]:
            if self._args["gcdqn_twin_GAT"]:
                node_summary_feat = node_node_feat = action
            else:
                node_summary_feat = action
        else:
            if self._args["gcdqn_use_pre_summarise_mlp"]:
                if self._args["gcdqn_twin_GAT"]:
                    # Get the compact representation of node-features
                    node_node_feat = getattr(self, f"{_main_or_target}_pre_summarize_mlp_node")(node_feat).squeeze(-1)
                    node_summary_feat = getattr(self, f"{_main_or_target}_pre_summarize_mlp_summary")(
                        node_feat).squeeze(-1)
                else:
                    # Get the compact representation of node-features
                    node_summary_feat = getattr(self, f"{_main_or_target}_pre_summarize_mlp")(node_feat).squeeze(-1)
            else:
                if self._args["gcdqn_twin_GAT"]:
                    node_summary_feat = node_node_feat = node_feat
                else:
                    node_summary_feat = node_feat

        if self._args["action_summarizer"] == 'gnn':
            if self._args["gcdqn_twin_GAT"]:
                # batch_size x num_nodes x dim_node; NOTE: Residual Connection is included in GAT
                z_node = getattr(self, f"{_main_or_target}_Q_net_node")(node_node_feat, adj_mat).squeeze(-1)
                z_summary = getattr(self, f"{_main_or_target}_Q_net_summary")(node_summary_feat, adj_mat).squeeze(-1)
            else:
                # batch_size x num_nodes x dim_node; NOTE: Residual Connection is included in GAT
                z_summary = getattr(self, f"{_main_or_target}_Q_net")(node_summary_feat, adj_mat).squeeze(-1)

            if not self._args["gcdqn_no_summary"]:
                summary_vec = ActionGraph.get_global_attribute(_aggregation_type=self._args["graph_aggregation_type"],
                                                               node_feat=z_summary)  # batch_size x dim_node
                if self._args["gcdqn_use_post_summarise_mlp"]:
                    # batch_size x dim_node
                    summary_vec = getattr(self, f"{_main_or_target}_post_summarize_mlp")(summary_vec)
            else:
                summary_vec = None
        elif self._args["action_summarizer"] in ["lstm", "deep_set"]:
            summary_vec = getattr(self, f"{_main_or_target}_Q_net")(node_summary_feat)
        else:
            raise NotImplementedError

        if self._graph_type.startswith("gat"):
            if self._args["gcdqn_twin_GAT"]:
                getattr(self, f"{_main_or_target}_Q_net_node").stack_attention()
            else:
                getattr(self, f"{_main_or_target}_Q_net").stack_attention()

        if self._args.get("if_e_node", False):  # AGILE
            assert self._args["action_summarizer"] == "gnn"
            if self._args["gcdqn_twin_GAT"]:
                action_feat = z_node
            else:
                action_feat = z_summary
        else:
            action_feat = action

        if not self._args["gcdqn_no_summary"]:
            summary_vec = summary_vec[:, None, :].repeat(1, node_summary_feat.shape[1], 1)
        return action_feat, summary_vec

    def _select_action(self, inputs, epsilon=0.1):
        """ Inner method of select_action

        Args:
            if self._if_sequentialQNet:
                if agent_standardRL_type == None:
                    inputs: (state_embed, item_embed) -> batch_step_size x num_candidates x dim_item_hidden
                else:  -> this shouldn't be used
                    inputs: (state_embed, item_embed, availability_mask)
                        - state_embed: batch_step_size x num_trainItems x dim_state
                        - item_embed: batch_step_size x num_trainItems x dim_item_hidden
                        - availability_mask: batch_step_size x num_trainItems
            else:
                if agent_standardRL_type != None:
                    inputs: (state_embed, item_embed) -> batch_step_size x dim
                else:
                    inputs: (state_embed, item_embed, availability_mask)
                        - state_embed: batch_step_size x num_trainItems x dim_state
                        - item_embed: batch_step_size x num_trainItems x dim_item_hidden
                        - availability_mask: batch_step_size x num_trainItems

        Returns:
            slate: batch_step_size x slate_size; Index based slate so that needs to be converted into itemIds later!
        """
        # Extract the item-embedding
        item_embed = inputs[1]

        if self._args["agent_standardRL_type"] != "None":
            state_embed, availability_mask = self._prep_input_standardRL(inputs=inputs)
            if self._args["agent_standardRL_type"] == "1":
                inputs = [state_embed]
            else:
                mask_embed = self.available_mask_summarizer((~availability_mask).float())
                inputs = [state_embed, mask_embed]

        if self._graph_type.startswith("gat"):
            if self._args["gcdqn_twin_GAT"]:
                self.main_Q_net_node.reset_attention()  # Reset the attention weight
                self.main_Q_net_summary.reset_attention()  # Reset the attention weight
            else:
                self.main_Q_net.reset_attention()  # Reset the attention weight

        if self._if_sequentialQNet:
            # batch_size depends on the epsilon at t
            batch_step_size, num_candidates, _ = inputs[0].shape
        else:
            # batch_size depends on the epsilon at t
            batch_step_size, _ = inputs[0].shape
            num_candidates = self._num_trainItems

        # init_nodes: state-action
        init_nodes, adj_mat, current, candidate, candidate_neg = self._initialize_settings(
            inputs=inputs, batch_step_size=batch_step_size, num_candidates=num_candidates
        )

        total_action = None
        slate_soFar = None

        for i in range(self._slate_size):
            if self._use_slate:
                slate_embed = self.main_slate_encoder.encode(slate=slate_soFar,
                                                             batch_size=batch_step_size,
                                                             item_embed=item_embed)
                if self._if_sequentialQNet:
                    slate_embed = slate_embed[:, None, :].repeat(1, num_candidates, 1)

            # Ablation; no GNN is equivalent to CDQN-share
            if self._args["agent_standardRL_type"] != "None" or self._args.get('gcdqn_no_graph', False):
                if self._args["agent_standardRL_type"] != "None":
                    # register the availability first
                    candidate_neg[availability_mask] = SLATE_MASK_TOKEN
                if self._use_slate:
                    q_all = self.final_main_Q_net([init_nodes, slate_embed])
                else:
                    q_all = self.final_main_Q_net(init_nodes)
            else:
                nodes = self._build_nodes(init_nodes, current, candidate)
                adj_mat = self.remove_selected_nodes(slate_soFar=slate_soFar, adj_mat=adj_mat)

                if self._args["action_summarizer"] not in ["", "None"]:  # AGILE
                    if not self._args["gcdqn_no_slate_for_node"]:  # nodes: state-action
                        nodes = torch.cat([nodes, slate_embed], dim=-1)  # nodes: state-slate-action

                    # batch_step_size x num_candidates x dim_node
                    action_feat, summary_vec = self.get_action_summary(node_feat=nodes,
                                                                       action=inputs[1],
                                                                       adj_mat=adj_mat,
                                                                       _main_or_target="main")

                    if self._use_slate:
                        if not self._args["gcdqn_no_summary"]:
                            # inputs: [state_embed, item_embed]
                            q_all = self.final_main_Q_net(
                                torch.cat([inputs[0], action_feat, summary_vec, slate_embed], dim=-1))
                        else:
                            q_all = self.final_main_Q_net(
                                torch.cat([inputs[0], action_feat, slate_embed], dim=-1))
                    else:
                        if not self._args["gcdqn_no_summary"]:
                            # inputs: [state_embed, item_embed]
                            q_all = self.final_main_Q_net(torch.cat([inputs[0], action_feat, summary_vec], dim=-1))
                        else:
                            # inputs: [state_embed, item_embed]
                            q_all = self.final_main_Q_net(torch.cat([inputs[0], action_feat], dim=-1))
                else:  # GCDQN
                    if self._use_slate:
                        q_all = self.main_Q_net(torch.cat([nodes, slate_embed], dim=-1), adj_mat)
                    else:
                        q_all = self.main_Q_net(nodes, adj_mat)

                    if self._graph_type.startswith("gat"):
                        self.main_Q_net.stack_attention()

                    if self._args["gcdqn_skip_connection"]:
                        if self._use_slate:
                            q_all = self.final_main_Q_net(torch.cat([init_nodes, q_all, slate_embed], dim=-1))
                        else:
                            q_all = self.final_main_Q_net(torch.cat([init_nodes, q_all], dim=-1))

                    if self._args["gcdqn_simple_skip_connection"]:
                        q_all = self.final_main_Q_net(q_all)

            if self._args.get('gcdqn_no_graph', False) or self._args["gcdqn_skip_connection"] or self._args[
                "gcdqn_simple_skip_connection"]:
                q_all = q_all.unsqueeze(-1)  # In the case of GAT with dim_out=1, it returns squeezed version

            if self._use_agg:
                q_candidate = q_all[:, :-1, :] + candidate_neg  # batch_step_size x num_candidates x 1
            else:
                q_candidate = q_all + candidate_neg  # batch_step_size x num_candidates x 1

            if self._train and self._args.get('boltzmann', True):
                # Boltzmann exploration, based on softmax
                action_dist = torch.distributions.categorical.Categorical(logits=q_candidate.squeeze(-1))
                action = action_dist.sample()
            else:
                action = torch.argmax(q_candidate.squeeze(-1), dim=-1)

            current = current.index_put_((torch.tensor(np.arange(batch_step_size), device=self._device), action),
                                         torch.tensor(1., device=self._device))
            candidate = candidate.index_put_((torch.tensor(np.arange(batch_step_size), device=self._device), action),
                                             torch.tensor(0., device=self._device))
            candidate_neg = candidate_neg.index_put_(
                (torch.tensor(np.arange(batch_step_size), device=self._device), action),
                torch.tensor(SLATE_MASK_TOKEN, device=self._device))
            if self._use_slate:
                slate_soFar = self.main_slate_encoder.update_slate(
                    _pos=i, slate=slate_soFar, item=action.cpu().detach().numpy().astype(np.int64)[:, None]
                )

            if total_action is None:
                total_action = action.unsqueeze(-1)
            else:
                total_action = torch.cat([total_action, action.unsqueeze(-1)], dim=-1)
        return total_action.cpu().detach().numpy().astype(np.int64)

    def _update(self,
                inputs: list,
                next_inputs: list,
                actions: torch.tensor,
                rewards: torch.tensor,
                reversed_dones: torch.tensor):
        """ Inner update method

        Args:
            inputs (list): list of input components in the form of batch_size x num_candidates x dim_data
            next_inputs (list): list of input components in the form of batch_size x num_candidates x dim_data
            actions (torch.tensor): batch_size x slate_size
            rewards (torch.tensor): batch_size x 1
            reversed_dones (torch.tensor): batch_size x 1

        Returns:
            result (dict): a dict of intermediate info during update
        """
        # from pudb import set_trace; set_trace()

        # Extract the item-embeddings
        item_embed, next_item_embed = inputs[1], next_inputs[1]

        if self._args["agent_standardRL_type"] != "None":
            state_embed, availability_mask = self._prep_input_standardRL(inputs=inputs)
            next_state_embed, next_availability_mask = self._prep_input_standardRL(inputs=next_inputs)
            if self._args["agent_standardRL_type"] == "1":
                inputs = [state_embed]
                next_inputs = [next_state_embed]
            else:
                mask_embed = self.available_mask_summarizer((~availability_mask).float())
                inputs = [state_embed, mask_embed]
                next_mask_embed = self.available_mask_summarizer((~next_availability_mask).float())
                next_inputs = [next_state_embed, next_mask_embed]

        if self._if_sequentialQNet:
            # batch_size depends on the epsilon at t
            batch_size, num_candidates, _ = inputs[0].shape
        else:
            # batch_size depends on the epsilon at t
            batch_size, _ = inputs[0].shape
            num_candidates = self._num_trainItems

        init_nodes, adj_mat, current, candidate, _ = self._initialize_settings(
            inputs=inputs, batch_step_size=batch_size, num_candidates=num_candidates
        )

        if self._graph_type.startswith("gat"):
            if self._args["gcdqn_twin_GAT"]:
                self.main_Q_net_node.reset_attention()
                self.main_Q_net_summary.reset_attention()
                self.target_Q_net_node.reset_attention()
                self.target_Q_net_summary.reset_attention()
            else:
                self.main_Q_net.reset_attention()
                self.target_Q_net.reset_attention()

        # === get Q-values associated with actions taken
        Q_vals = None
        for i in range(self._slate_size):
            current = torch.zeros(batch_size, num_candidates, 1, device=self._device)
            candidate = torch.ones(batch_size, num_candidates, 1, device=self._device)
            if i > 0:
                _temp = (torch.tensor([np.arange(batch_size)] * i, device=self._device).t(), actions[:, :i])
                current = current.index_put(_temp, torch.tensor(1., device=self._device))
                candidate = candidate.index_put(_temp, torch.tensor(0., device=self._device))

            if self._use_slate:
                slate_embed = self.main_slate_encoder.encode(slate=actions[:, :i],
                                                             batch_size=batch_size,
                                                             item_embed=item_embed)
                if self._if_sequentialQNet:
                    slate_embed = slate_embed[:, None, :].repeat(1, num_candidates, 1)

            # Ablation; no GNN is equivalent to CDQN-share
            if self._args["agent_standardRL_type"] != "None" or self._args.get('gcdqn_no_graph', False):
                if self._use_slate:
                    q_all = self.final_main_Q_net([init_nodes, slate_embed])
                else:
                    q_all = self.final_main_Q_net(init_nodes)
            else:
                nodes = self._build_nodes(init_nodes, current, candidate)
                adj_mat = self.remove_selected_nodes(slate_soFar=actions[:, :i].detach().cpu(), adj_mat=adj_mat)

                if self._args["action_summarizer"] not in ["", "None"]:  # AGILE
                    if not self._args["gcdqn_no_slate_for_node"]:
                        nodes = torch.cat([nodes, slate_embed], dim=-1)

                    # batch_step_size x num_candidates x dim_node
                    action_feat, summary_vec = self.get_action_summary(node_feat=nodes,
                                                                       action=inputs[1],
                                                                       adj_mat=adj_mat,
                                                                       _main_or_target="main")

                    if self._use_slate:
                        if not self._args["gcdqn_no_summary"]:
                            # inputs: [state_embed, item_embed]
                            q_all = self.final_main_Q_net(
                                torch.cat([inputs[0], action_feat, summary_vec, slate_embed], dim=-1))
                        else:
                            q_all = self.final_main_Q_net(
                                torch.cat([inputs[0], action_feat, slate_embed], dim=-1))
                    else:
                        if not self._args["gcdqn_no_summary"]:
                            # inputs: [state_embed, item_embed]
                            q_all = self.final_main_Q_net(torch.cat([inputs[0], action_feat, summary_vec], dim=-1))
                        else:
                            # inputs: [state_embed, item_embed]
                            q_all = self.final_main_Q_net(torch.cat([inputs[0], action_feat], dim=-1))

            if self._args.get('gcdqn_no_graph', False) or self._args["gcdqn_skip_connection"] or self._args[
                "gcdqn_simple_skip_connection"]:
                q_all = q_all.unsqueeze(-1)  # In the case of GAT with dim_out=1, it returns squeezed version

            # Get the q-value for the k-th item in the slate
            if self._use_agg:
                q_val = q_all[:, :-1, 0].gather(dim=1, index=actions[:, i][:, np.newaxis])
            else:
                q_val = q_all[:, :, 0].gather(dim=1, index=actions[:, i][:, np.newaxis])

            if i == 0:
                Q_vals = q_val
            else:
                Q_vals = torch.cat([Q_vals, q_val], dim=-1)  # batch_size x slate_size

        # === get next Q-values
        slate_soFar = None
        target_init_nodes, target_adj_mat, target_current, target_candidate, target_candidate_neg = \
            self._initialize_settings(inputs=inputs, batch_step_size=batch_size, num_candidates=num_candidates)

        if self._graph_type.startswith("gat"):
            if self._args["gcdqn_twin_GAT"]:
                self.target_Q_net_node.reset_attention()
                self.target_Q_net_summary.reset_attention()
            else:
                self.target_Q_net.reset_attention()

        """

        === Get the q-vals from the second to the last intra-time-step of the CURRENT SLATE ===

        """
        # === Prep before the main for loop
        # Add the action of the first intra-time-step
        if self._use_slate:
            if self._args["if_use_main_target_for_others"]:
                slate_soFar = self.target_slate_encoder.update_slate(
                    _pos=0, slate=slate_soFar, item=actions[:, 0].cpu().detach().numpy().astype(np.int64)[:, None])
            else:
                slate_soFar = self.main_slate_encoder.update_slate(
                    _pos=0, slate=slate_soFar, item=actions[:, 0].cpu().detach().numpy().astype(np.int64)[:, None])

        # Update the vars for the first intra-time-step
        target_current = target_current.index_put_(
            (torch.tensor(np.arange(batch_size), device=self._device), actions[:, 0]),
            torch.tensor(1., device=self._device))
        target_candidate = target_candidate.index_put_(
            (torch.tensor(np.arange(batch_size), device=self._device), actions[:, 0]),
            torch.tensor(0., device=self._device))
        target_candidate_neg = target_candidate_neg.index_put_(
            (torch.tensor(np.arange(batch_size), device=self._device), actions[:, 0]),
            torch.tensor(SLATE_MASK_TOKEN, device=self._device))
        target_adj_mat = self.remove_selected_nodes(slate_soFar=slate_soFar, adj_mat=target_adj_mat)

        # === Main for loop
        for i in range(1, self._slate_size):  # We skip the first position
            if self._use_slate:
                if self._args["if_use_main_target_for_others"]:
                    target_slate_embed = self.target_slate_encoder.encode(slate=slate_soFar,
                                                                          batch_size=batch_size,
                                                                          item_embed=item_embed)
                else:
                    target_slate_embed = self.main_slate_encoder.encode(slate=slate_soFar,
                                                                        batch_size=batch_size,
                                                                        item_embed=item_embed)
                if self._if_sequentialQNet:
                    target_slate_embed = target_slate_embed[:, None, :].repeat(1, num_candidates, 1)

            # Ablation; no GNN is equivalent to CDQN-share
            if self._args["agent_standardRL_type"] != "None" or self._args.get('gcdqn_no_graph', False):
                if self._use_slate:
                    next_q_all = self.final_target_Q_net([target_init_nodes, target_slate_embed])
                else:
                    next_q_all = self.final_target_Q_net(target_init_nodes)
            else:
                target_nodes = self._build_nodes(target_init_nodes, target_current, target_candidate)
                target_adj_mat = self.remove_selected_nodes(slate_soFar=slate_soFar, adj_mat=target_adj_mat)

                if self._args["action_summarizer"] not in ["", "None"]:  # AGILE or RCDQN
                    if not self._args["gcdqn_no_slate_for_node"]:
                        target_nodes = torch.cat([target_nodes, target_slate_embed], dim=-1)

                    # batch_step_size x num_candidates x dim_node
                    if self._args["gcdqn_singleGAT"]:
                        _main_or_target = "main"
                    else:
                        _main_or_target = "target"
                    target_action_feat, target_summary_vec = self.get_action_summary(node_feat=target_nodes,
                                                                                     action=inputs[1],
                                                                                     adj_mat=target_adj_mat,
                                                                                     _main_or_target=_main_or_target)

                    if self._use_slate:
                        if not self._args["gcdqn_no_summary"]:
                            # inputs: [state_embed, item_embed]
                            next_q_all = self.final_target_Q_net(torch.cat(
                                [inputs[0], target_action_feat, target_summary_vec, target_slate_embed], dim=-1))
                        else:
                            next_q_all = self.final_target_Q_net(
                                torch.cat([inputs[0], target_action_feat, target_slate_embed], dim=-1))
                    else:
                        if not self._args["gcdqn_no_summary"]:
                            # inputs: [state_embed, item_embed]
                            next_q_all = self.final_target_Q_net(
                                torch.cat([inputs[0], target_action_feat, target_summary_vec], dim=-1))
                        else:
                            # inputs: [state_embed, item_embed]
                            next_q_all = self.final_target_Q_net(
                                torch.cat([inputs[0], target_action_feat], dim=-1))

            if self._args.get('gcdqn_no_graph', False) or self._args["gcdqn_skip_connection"] or self._args[
                "gcdqn_simple_skip_connection"]:
                next_q_all = next_q_all.unsqueeze(-1)

            if self._use_agg:
                # batch_step_size x num_candidates x 1
                next_q_candidate = next_q_all[:, :-1, :] + target_candidate_neg
            else:
                next_q_candidate = next_q_all + target_candidate_neg  # batch_step_size x num_candidates x 1

            next_action = torch.argmax(next_q_candidate.squeeze(-1), dim=-1)

            # Get the q-value for the k-th item in the slate
            if self._use_agg:
                next_q_val = next_q_all[:, :-1, 0].gather(dim=1, index=next_action.unsqueeze(-1))
            else:
                next_q_val = next_q_all[:, :, 0].gather(dim=1, index=next_action.unsqueeze(-1))

            target_current = target_current.index_put_(
                (torch.tensor(np.arange(batch_size), device=self._device), actions[:, i]),
                torch.tensor(1., device=self._device))
            target_candidate = target_candidate.index_put_(
                (torch.tensor(np.arange(batch_size), device=self._device), actions[:, i]),
                torch.tensor(0., device=self._device))
            target_candidate_neg = target_candidate_neg.index_put_(
                (torch.tensor(np.arange(batch_size), device=self._device), actions[:, i]),
                torch.tensor(SLATE_MASK_TOKEN, device=self._device))

            if self._use_slate:
                if self._args["if_use_main_target_for_others"]:
                    slate_soFar = self.target_slate_encoder.update_slate(
                        _pos=i, slate=slate_soFar, item=actions[:, i].cpu().detach().numpy().astype(np.int64)[:, None])
                else:
                    slate_soFar = self.main_slate_encoder.update_slate(
                        _pos=i, slate=slate_soFar, item=actions[:, i].cpu().detach().numpy().astype(np.int64)[:, None])

            if self._args["gcdqn_stack_next_q"]:
                if i == 1:
                    # batch_size x 1
                    next_Q_vals = next_q_val
                else:
                    next_Q_vals = torch.cat([next_Q_vals, next_q_val], dim=-1)  # batch_size x slate_size
            else:
                next_Q_vals = next_q_val

        """
        
        === Get the q-val of the first intra-time-step of the NEXT SLATE ===
        
        """
        slate_soFar = None
        next_init_nodes, next_adj_mat, next_current, next_candidate, next_candidate_neg = self._initialize_settings(
            inputs=next_inputs, batch_step_size=batch_size, num_candidates=num_candidates
        )

        if self._args["if_use_main_target_for_others"]:
            next_slate_embed = self.target_slate_encoder.encode(slate=slate_soFar,
                                                                batch_size=batch_size,
                                                                item_embed=next_item_embed)
        else:
            next_slate_embed = self.main_slate_encoder.encode(slate=slate_soFar,
                                                              batch_size=batch_size,
                                                              item_embed=next_item_embed)
        if self._if_sequentialQNet:
            next_slate_embed = next_slate_embed[:, None, :].repeat(1, num_candidates, 1)

        # Ablation; no GNN is equivalent to CDQN-share
        if self._args["agent_standardRL_type"] != "None" or self._args.get('gcdqn_no_graph', False):
            if self._use_slate:
                next_q_all = self.final_target_Q_net([next_init_nodes, next_slate_embed])
            else:
                next_q_all = self.final_target_Q_net(next_init_nodes)
        else:
            next_nodes = self._build_nodes(next_init_nodes, next_current, next_candidate)
            next_adj_mat = self.remove_selected_nodes(slate_soFar=slate_soFar, adj_mat=next_adj_mat)

            if self._args["action_summarizer"] not in ["", "None"]:  # AGILE or RCDQN
                if not self._args["gcdqn_no_slate_for_node"]:
                    next_nodes = torch.cat([next_nodes, next_slate_embed], dim=-1)

                # batch_step_size x num_candidates x dim_node
                if self._args["gcdqn_singleGAT"]:
                    _main_or_target = "main"
                else:
                    _main_or_target = "target"
                next_action_feat, next_summary_vec = self.get_action_summary(node_feat=next_nodes,
                                                                             action=next_inputs[1],
                                                                             adj_mat=next_adj_mat,
                                                                             _main_or_target=_main_or_target)

                if self._use_slate:
                    if not self._args["gcdqn_no_summary"]:
                        # inputs: [state_embed, item_embed]
                        next_q_all = self.final_target_Q_net(
                            torch.cat([next_inputs[0], next_action_feat, next_summary_vec, next_slate_embed], dim=-1))
                    else:
                        next_q_all = self.final_target_Q_net(
                            torch.cat([next_inputs[0], next_action_feat, next_slate_embed], dim=-1))
                else:
                    if not self._args["gcdqn_no_summary"]:
                        # inputs: [state_embed, item_embed]
                        next_q_all = self.final_target_Q_net(
                            torch.cat([next_inputs[0], next_action_feat, next_summary_vec], dim=-1))
                    else:
                        # inputs: [state_embed, item_embed]
                        next_q_all = self.final_target_Q_net(torch.cat([next_inputs[0], next_action_feat], dim=-1))

        if self._args.get('gcdqn_no_graph', False) or self._args["gcdqn_skip_connection"] or self._args[
            "gcdqn_simple_skip_connection"]:
            next_q_all = next_q_all.unsqueeze(-1)

        """ Note: here we don't intentionally use next_candidate_neg since it's all zeroes for the first position
            in the first position of the next slate!
         """
        if self._use_agg:
            # batch_step_size x num_candidates x 1
            next_q_candidate = next_q_all[:, :-1, :]
        else:
            next_q_candidate = next_q_all  # batch_step_size x num_candidates x 1

        next_action = torch.argmax(next_q_candidate.squeeze(-1), dim=-1)

        # Get the q-value for the k-th item in the slate
        if self._use_agg:
            next_q_val = next_q_all[:, :-1, 0].gather(dim=1, index=next_action.unsqueeze(-1))
        else:
            next_q_val = next_q_all[:, :, 0].gather(dim=1, index=next_action.unsqueeze(-1))

        next_Q_vals = torch.cat([next_Q_vals, next_q_val], dim=-1)  # batch_size x slate_size

        if self._args["gcdqn_attention_bonus"] and self._args["graph_type"].startswith("gat"):
            dict_info_att = self.main_Q_net.get_attention_stats(args=None, mean=False)
            att_bonus = torch.tensor(dict_info_att["att_std"].mean(axis=-1)[:, None], device=self._args["device"])
            Y = (rewards + att_bonus) + next_Q_vals * reversed_dones * self._gamma
        else:
            Y = rewards + next_Q_vals * reversed_dones * self._gamma

        """ Loss for k Q-networks: (y - Q^j) where y = rewards + Q^k
            - As in Eq(11), all the k Q-networks are fitting to the same y
        """
        loss = torch.mean((Y - Q_vals).pow(2))

        # Refresh the optimisers
        [_opt.zero_grad() for _opt in self.optimiser_list]

        # Backprop the loss
        loss.backward()

        # Gradient Clipping; Backward compatibility
        # torch.nn.utils.clip_grad_norm_(self.main_action_linear.parameters(), self._grad_clip)
        # if self._if_check_grad:
        #     logging("=== action_linear ===")
        #     ave_grad_dict, max_grad_dict = check_grad(named_parameters=self.main_action_linear.named_parameters())
        #     logging("Ave grad: {}\nMax grad: {}".format(ave_grad_dict, max_grad_dict))
        if self._args["gcdqn_skip_connection"] or self._args["gcdqn_simple_skip_connection"]:
            Q_net = self.final_main_Q_net
            torch.nn.utils.clip_grad_norm_(Q_net.parameters(), self._grad_clip)
            if self._if_check_grad:
                logging("=== final_main_Q_net ===")
                ave_grad_dict, max_grad_dict = check_grad(named_parameters=Q_net.named_parameters())
                logging("Ave grad: {}\nMax grad: {}".format(ave_grad_dict, max_grad_dict))
        if not self._args.get('gcdqn_no_graph', False):
            if self._args["gcdqn_twin_GAT"]:
                for Q_net in [self.main_Q_net_node, self.main_Q_net_summary]:
                    torch.nn.utils.clip_grad_norm_(Q_net.parameters(), self._grad_clip)
                    if self._if_check_grad:
                        logging("=== main_Q_net ===")
                        ave_grad_dict, max_grad_dict = check_grad(named_parameters=Q_net.named_parameters())
                        logging("Ave grad: {}\nMax grad: {}".format(ave_grad_dict, max_grad_dict))
            else:
                Q_net = self.main_Q_net
                torch.nn.utils.clip_grad_norm_(Q_net.parameters(), self._grad_clip)
                if self._if_check_grad:
                    logging("=== main_Q_net ===")
                    ave_grad_dict, max_grad_dict = check_grad(named_parameters=Q_net.named_parameters())
                    logging("Ave grad: {}\nMax grad: {}".format(ave_grad_dict, max_grad_dict))
        if self._args["action_summarizer"] == "gnn":  # AGILE
            if self._args["gcdqn_twin_GAT"] and self._args["gcdqn_use_pre_summarise_mlp"]:
                models = list()
                models += [self.main_pre_summarize_mlp_node, self.main_pre_summarize_mlp_summary]
                if self.main_post_summarize_mlp is not None: models += [self.main_post_summarize_mlp]
                for model in models:
                    if model is None: continue
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self._grad_clip)
                    if self._if_check_grad:
                        logging(f"=== {model.__class__.__name__} ===")
                        ave_grad_dict, max_grad_dict = check_grad(named_parameters=model.named_parameters())
                        logging("Ave grad: {}\nMax grad: {}".format(ave_grad_dict, max_grad_dict))
            else:
                models = list()
                if self.main_pre_summarize_mlp is not None: models += [self.main_pre_summarize_mlp]
                if self.main_post_summarize_mlp is not None: models += [self.main_post_summarize_mlp]
                for model in models:
                    if model is None: continue
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self._grad_clip)
                    if self._if_check_grad:
                        logging(f"=== {model.__class__.__name__} ===")
                        ave_grad_dict, max_grad_dict = check_grad(named_parameters=model.named_parameters())
                        logging("Ave grad: {}\nMax grad: {}".format(ave_grad_dict, max_grad_dict))

        for model in [self.main_obs_encoder, self.main_slate_encoder]:
            if model is None:
                continue
            if hasattr(model, "encoder"):
                if model.encoder is None:
                    continue
            model = model.encoder
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self._grad_clip)
            if self._if_check_grad:
                logging("=== Encoder: {} ===".format(model.__class__.__name__))
                ave_grad_dict, max_grad_dict = check_grad(named_parameters=model.named_parameters())
                logging("Ave grad: {}\nMax grad: {}".format(ave_grad_dict, max_grad_dict))

        # apply processed gradients to the network
        [_opt.step() for _opt in self.optimiser_list]
        if self._args["if_use_lr_scheduler"]:
            print([_opt.param_groups[0]["lr"] for _opt in self.optimiser_list])
            [_scheduler.step(epoch=self.epoch) for _scheduler in self.lr_scheduler_list]
            print([_opt.param_groups[0]["lr"] for _opt in self.optimiser_list])

        # Create the summary of stats of agent during update
        res = {"loss": loss.item()}

        res.update({
            "max_q_val": Q_vals.cpu().detach().numpy().max(-1).mean(),
            "mean_q_val": Q_vals.cpu().detach().numpy().mean(-1).mean(),
        })

        if self._args["action_summarizer"] not in ["", "None"]:
            if summary_vec is not None:
                res.update({
                    "std_summary_vec": summary_vec.std(-1).mean().cpu().detach().numpy(),
                    "max_summary_vec": summary_vec.max(-1)[0].mean().cpu().detach().numpy(),
                    "std_action_feat": action_feat.std(-1).mean().cpu().detach().numpy(),
                    "max_action_feat": action_feat.max(-1)[0].mean().cpu().detach().numpy(),
                })

        if self._args.get("if_visualise_agent", False):
            if not self._args.get('gcdqn_no_graph', False) and self._args["graph_type"].startswith("gat"):
                if self._args["gcdqn_twin_GAT"]:
                    dict_info_att = self.main_Q_net_node.get_attention_stats(args=None)
                else:
                    dict_info_att = self.main_Q_net.get_attention_stats(args=None)
                res.update(dict_info_att)
        return res

    def _save(self, save_dir: str, epoch: int = 0):
        ep = epoch
        if not self._args["gcdqn_no_graph"]:
            if self._args["gcdqn_twin_GAT"]:
                torch.save(self.main_Q_net_node.state_dict(), os.path.join(save_dir, f"main_node_ep{ep}.pkl"))
                torch.save(self.target_Q_net_node.state_dict(), os.path.join(save_dir, f"target_node_ep{ep}.pkl"))
                torch.save(self.main_Q_net_summary.state_dict(), os.path.join(save_dir, f"main_summary_ep{ep}.pkl"))
                torch.save(self.target_Q_net_summary.state_dict(), os.path.join(save_dir, f"target_summary_ep{ep}.pkl"))
            else:
                torch.save(self.main_Q_net.state_dict(), os.path.join(save_dir, f"main_ep{ep}.pkl"))
                torch.save(self.target_Q_net.state_dict(), os.path.join(save_dir, f"target_ep{ep}.pkl"))
        if self._args["gcdqn_skip_connection"] or self._args["gcdqn_simple_skip_connection"]:
            torch.save(self.final_main_Q_net.state_dict(), os.path.join(save_dir, f"final_main_ep{ep}.pkl"))
            torch.save(self.final_target_Q_net.state_dict(), os.path.join(save_dir, f"final_target_ep{ep}.pkl"))

        if self._args["gcdqn_twin_GAT"] and self._args["gcdqn_use_pre_summarise_mlp"]:
            torch.save(self.main_pre_summarize_mlp_node.state_dict(),
                       os.path.join(save_dir, f"main_pre_summarize_mlp_node_ep{ep}.pkl"))
            torch.save(self.target_pre_summarize_mlp_node.state_dict(),
                       os.path.join(save_dir, f"target_pre_summarize_mlp_node_ep{ep}.pkl"))
            torch.save(self.main_pre_summarize_mlp_summary.state_dict(),
                       os.path.join(save_dir, f"main_pre_summarize_mlp_summary_ep{ep}.pkl"))
            torch.save(self.target_pre_summarize_mlp_summary.state_dict(),
                       os.path.join(save_dir, f"target_pre_summarize_mlp_ep{ep}.pkl"))
        else:
            if self.main_pre_summarize_mlp is not None:
                torch.save(self.main_pre_summarize_mlp.state_dict(),
                           os.path.join(save_dir, f"main_pre_summarize_mlp_ep{ep}.pkl"))
                torch.save(self.target_pre_summarize_mlp.state_dict(),
                           os.path.join(save_dir, f"target_pre_summarize_mlp_ep{ep}.pkl"))
        if self.main_post_summarize_mlp is not None:
            torch.save(self.main_post_summarize_mlp.state_dict(),
                       os.path.join(save_dir, f"main_post_summarize_mlp_ep{ep}.pkl"))
            torch.save(self.target_post_summarize_mlp.state_dict(),
                       os.path.join(save_dir, f"target_post_summarize_mlp_ep{ep}.pkl"))

    def _load(self, save_dir: str, epoch: int = 0):
        ep = epoch
        if not self._args["gcdqn_no_graph"]:
            if self._args["gcdqn_twin_GAT"]:
                self.main_Q_net_node.load_state_dict(torch.load(os.path.join(save_dir, f"main_node_ep{ep}.pkl")))
                self.target_Q_net_node.load_state_dict(torch.load(os.path.join(save_dir, f"target_node_ep{ep}.pkl")))
                self.main_Q_net_summary.load_state_dict(torch.load(os.path.join(save_dir, f"main_summary_ep{ep}.pkl")))
                self.target_Q_net_summary.load_state_dict(
                    torch.load(os.path.join(save_dir, f"target_summary_ep{ep}.pkl")))
            else:
                self.main_Q_net.load_state_dict(torch.load(os.path.join(save_dir, f"main_ep{ep}.pkl")))
                self.target_Q_net.load_state_dict(torch.load(os.path.join(save_dir, f"target_ep{ep}.pkl")))
        if self._args["gcdqn_skip_connection"] or self._args["gcdqn_simple_skip_connection"]:
            self.final_main_Q_net.load_state_dict(torch.load(os.path.join(save_dir, f"final_main_ep{ep}.pkl")))
            self.final_target_Q_net.load_state_dict(torch.load(os.path.join(save_dir, f"final_target_ep{ep}.pkl")))

        if self._args["gcdqn_twin_GAT"] and self._args["gcdqn_use_pre_summarise_mlp"]:
            self.main_pre_summarize_mlp_node.load_state_dict(
                torch.load(os.path.join(save_dir, f"main_pre_summarize_mlp_node_ep{ep}.pkl")))
            self.target_pre_summarize_mlp_node.load_state_dict(
                torch.load(os.path.join(save_dir, f"target_pre_summarize_mlp_node_ep{ep}.pkl")))
            self.main_pre_summarize_mlp_summary.load_state_dict(
                torch.load(os.path.join(save_dir, f"main_pre_summarize_mlp_summary_ep{ep}.pkl")))
            self.target_pre_summarize_mlp_summary.load_state_dict(
                torch.load(os.path.join(save_dir, f"target_pre_summarize_mlp_summary_ep{ep}.pkl")))
        else:
            if self.main_pre_summarize_mlp is not None:
                self.main_pre_summarize_mlp.load_state_dict(
                    torch.load(os.path.join(save_dir, f"main_pre_summarize_mlp_ep{ep}.pkl")))
                self.target_pre_summarize_mlp.load_state_dict(
                    torch.load(os.path.join(save_dir, f"target_pre_summarize_mlp_ep{ep}.pkl")))
        if self.main_post_summarize_mlp is not None:
            self.main_post_summarize_mlp.load_state_dict(
                torch.load(os.path.join(save_dir, f"main_post_summarize_mlp_ep{ep}.pkl")))
            self.target_post_summarize_mlp.load_state_dict(
                torch.load(os.path.join(save_dir, f"target_post_summarize_mlp_ep{ep}.pkl")))

    def _sync(self, tau: float = 0.0):
        # soft-update only for GAT
        if self._args["gcdqn_twin_GAT"]:
            if self._args["gcdqn_twin_GAT_soft_update"] > 0.0:
                for param, target_param in zip(self.main_Q_net_node.parameters(), self.target_Q_net_node.parameters()):
                    # tau * local_param.data + (1.0 - tau) * target_param.data
                    target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
                for param, target_param in zip(self.main_Q_net_summary.parameters(),
                                               self.target_Q_net_summary.parameters()):
                    # tau * local_param.data + (1.0 - tau) * target_param.data
                    target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

        if tau > 0.0:  # Soft update of params
            if not self._args["gcdqn_singleGAT"] and not self._args["gcdqn_no_graph"]:
                if self._args["gcdqn_twin_GAT"]:
                    for param, target_param in zip(self.main_Q_net_node.parameters(),
                                                   self.target_Q_net_node.parameters()):
                        # tau * local_param.data + (1.0 - tau) * target_param.data
                        target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
                    for param, target_param in zip(self.main_Q_net_summary.parameters(),
                                                   self.target_Q_net_summary.parameters()):
                        # tau * local_param.data + (1.0 - tau) * target_param.data
                        target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
                else:
                    for param, target_param in zip(self.main_Q_net.parameters(), self.target_Q_net.parameters()):
                        # tau * local_param.data + (1.0 - tau) * target_param.data
                        target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

            if self._args["gcdqn_skip_connection"] or self._args["gcdqn_simple_skip_connection"]:
                for param, target_param in zip(self.final_main_Q_net.parameters(),
                                               self.final_target_Q_net.parameters()):
                    # tau * local_param.data + (1.0 - tau) * target_param.data
                    target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
            if self._args["action_summarizer"] not in ["", "None"]:  # AGILE
                if self._args["gcdqn_twin_GAT"] and self._args["gcdqn_use_pre_summarise_mlp"]:
                    for param, target_param in zip(self.main_pre_summarize_mlp_node.parameters(),
                                                   self.target_pre_summarize_mlp_node.parameters()):
                        # tau * local_param.data + (1.0 - tau) * target_param.data
                        target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
                    for param, target_param in zip(self.main_pre_summarize_mlp_summary.parameters(),
                                                   self.target_pre_summarize_mlp_summary.parameters()):
                        # tau * local_param.data + (1.0 - tau) * target_param.data
                        target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
                else:
                    if self.main_pre_summarize_mlp is not None:
                        for param, target_param in zip(self.main_pre_summarize_mlp.parameters(),
                                                       self.target_pre_summarize_mlp.parameters()):
                            # tau * local_param.data + (1.0 - tau) * target_param.data
                            target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

                if self.main_post_summarize_mlp is not None:
                    for param, target_param in zip(self.main_post_summarize_mlp.parameters(),
                                                   self.target_post_summarize_mlp.parameters()):
                        # tau * local_param.data + (1.0 - tau) * target_param.data
                        target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
        else:  # Hard update of params
            if not self._args["gcdqn_singleGAT"]:
                if self._args["gcdqn_twin_GAT"] and not self._args["gcdqn_bug_fixed_target_gat"]:
                    self.target_Q_net_node.load_state_dict(self.main_Q_net_node.state_dict())
                    self.target_Q_net_summary.load_state_dict(self.main_Q_net_summary.state_dict())
                else:
                    if self.target_Q_net is not None:
                        self.target_Q_net.load_state_dict(self.main_Q_net.state_dict())
            if self._args["gcdqn_skip_connection"] or self._args["gcdqn_simple_skip_connection"]:
                self.final_target_Q_net.load_state_dict(self.final_main_Q_net.state_dict())
            if self._args["gcdqn_twin_GAT"] and self._args["gcdqn_use_pre_summarise_mlp"]:
                self.target_pre_summarize_mlp_node.load_state_dict(self.main_pre_summarize_mlp_node.state_dict())
                self.target_pre_summarize_mlp_summary.load_state_dict(self.main_pre_summarize_mlp_summary.state_dict())
            else:
                if self.main_pre_summarize_mlp is not None:
                    self.target_pre_summarize_mlp.load_state_dict(self.main_pre_summarize_mlp.state_dict())
            if self.main_post_summarize_mlp is not None:
                self.target_post_summarize_mlp.load_state_dict(self.main_post_summarize_mlp.state_dict())

    def visualise(self):
        res = dict()
        for i in range(self._slate_size):
            _res = self._get_summary_of_params(model=self.target_Q_net, model_name="target{}".format(i))
            for k, v in _res.items(): res[k] = v

            _res = self._get_summary_of_params(model=self.main_Q_net, model_name="main{}".format(i))
            for k, v in _res.items(): res[k] = v
        return res

    def _get_summary_of_params(self, model, model_name):
        _res = dict()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                _mean = torch.mean(param.data).cpu().detach().numpy()
                _std = torch.std(param.data).cpu().detach().numpy()
                _res["{}_{}_mean".format(model_name, param_name)] = float(_mean)
                _res["{}_{}_std".format(model_name, param_name)] = float(_std)
        return _res


class Test(test):
    def __init__(self):
        self._get_args()
        # self.args.device = "cuda"
        self.args.if_debug = True
        self.args.if_debug = False
        self.args.if_visualise_debug = True
        self.args.env_name = "recsim"
        self.args.agent_type = "gcdqn"
        self.args.batch_size = 3
        self._prep()

    def test(self):
        from value_based.commons.launcher import launch_agent, launch_encoders
        from value_based.commons.args import add_args
        from value_based.commons.args_agents import params_dict, check_param

        # main test part
        for agent_name in [
            # "DRAG",
            # "CDQN-GCDQN",
            # "SRL-GCDQN",
            # "RCDQN",
            "AGILE",
            # "AGILE-TWIN-GAT",
            # "AGILE-action",
            # "AGILE-state-action",
            # "Summary-state-action",
            # "Summary-deepset-state-action",
            # "AGILE-no-pre-summariser",
            # "AGILE-no-post-summariser",
            # "AGILE-no-slate-for-node",
            # "AGILE-no-summary",
            # "AGILE-state-action-no-summary",
            # "AGILE-action-no-summary",
        ]:
            for _params_dict in params_dict[agent_name]:
                logging(f"\n=== params {_params_dict} ===")
                if not check_param(_params_dict=_params_dict): continue

                # Update the hyper-params with the test specific ones
                args = self.update_args_from_dict(args=self.args, _dict=_params_dict)
                args = add_args(args=args)
                encoders_dict = launch_encoders(item_embedding=self.dict_embedding["item"], args=args)
                self.agent = launch_agent(dict_embedding=self.dict_embedding,
                                          encoders_dict=encoders_dict,
                                          args=args)
                self.agent.set_baseItems_dict(baseItems_dict=self.baseItems_dict)
                self.test_select_action()
                self.test_update()
                # self.agent.sync()
                # self.test_save()
                # self.test_load()
                # self.test_visualise()


if __name__ == '__main__':
    Test().test()
