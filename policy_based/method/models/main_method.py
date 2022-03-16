'''
    Our Policy Class Logic is implemented here
'''

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from policy_based.rlf.rl.utils import init
import os.path as osp
from gnn.GAT.models_geometric import GAT4 as GAT4_GEO
from gnn.GAT.models_geometric import GAT2 as GAT2_GEO
from gnn.GAT.models import GAT2, GATSimple
from value_based.commons.plot_agent import __visualise as visualise
from policy_based.envs.create_game import ToolGenerator
from policy_based.envs.gym_minigrid.wrappers import skill_types, skill_category

import policy_based.method.utils as meth_utils

from policy_based.rlf import BasePolicy
from policy_based.rlf.rl.distributions import FixedCategorical, Categorical, MaskCategorical


EPS = 1e-6

class ActionFeatureCategorical(nn.Module):
    def __init__(self, num_inputs, dist_mem, args):
        super().__init__()

        hidden_dim = args.dist_hidden_dim
        self.args = args

        if args.dist_non_linear_final:
            # Default is TRUE
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim + num_inputs, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1))
        else:
            self.mlp = nn.Linear(hidden_dim + num_inputs, 1)

    def forward(self, x, add_input, action_features):
        """
        Parameters
        ----------
        x : torch.tensor: batch_size x num_inputs
            Contains state and action_set summary information
            For NoGNN baseline: action_set summary is absent.
        add_input : torch.tensor : batch_size x action_set_size
            Available actions for indexing to dist_mem
        action_features : torch.tensor : batch_size x action_set_size x
                action_features
            Processed Action Features for available actions.
            G-PPO (Ours): action-features are processed from GNN output layer
            Summary-GNN/LSTM/Deep_set (Baselines) : action-features are independently processed by non-relational NNs
        Returns
        ----------
        FixedCategorical (torch.distributions.distribution): batch_size x
            action_set_size x 1
            Distribution over available actions
        """
        act = action_features
        x = torch.cat([x.view([x.shape[0], 1, x.shape[1]]).repeat(1, act.shape[1], 1), act], dim=-1)
        x = self.mlp(x).squeeze(-1)
        if self.args.use_dist_double:
            x = x.double()
        return FixedCategorical(logits=x)

class GraphCategorical(nn.Module):
    def __init__(self, num_inputs, dist_mem, args):
        super().__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        init2_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=nn.init.calculate_gain('relu'))

        hidden_dim = args.dist_hidden_dim
        self.args = args
        action_dim = args.o_dim if args.use_option_embs else args.z_dim
        self.dist_mem = dist_mem

        if args.gnn_gat_model.upper() == 'GAT2':
            self.gnn_choice = GAT2
        elif args.gnn_gat_model.upper() == 'GAT_SIMPLE':
            self.gnn_choice = GATSimple
        elif args.gnn_gat_model.upper() == 'GAT2_GEO':
            self.gnn_choice = GAT2_GEO
        elif args.gnn_gat_model.upper() == 'GAT4_GEO':
            self.gnn_choice = GAT4_GEO
        else:
            raise NotImplementedError

        self.utility_network = self.gnn_choice(
                 dim_in=hidden_dim,
                 dim_hidden=hidden_dim,
                 dim_out=hidden_dim,
                 num_heads=args.gat_num_attention_heads,
                 args=self.args)

        if self.args.gnn_use_action_summary:
            self.gnn_action_summary = self.gnn_choice(
                     dim_in=hidden_dim,
                     dim_hidden=hidden_dim,
                     dim_out=hidden_dim,
                     num_heads=args.gat_num_attention_heads,
                     args=self.args)
            self.concat_summary_linear = nn.Linear(2 * hidden_dim, hidden_dim)

        if args.use_state_mlp:
            self.state_linear = nn.Sequential(
                    (nn.Linear(num_inputs, hidden_dim)), nn.ReLU(),
                    (nn.Linear(hidden_dim, hidden_dim)))

        if args.dist_linear_action:
            self.action_linear = nn.Linear(action_dim, hidden_dim)
        else:
            self.action_linear = nn.Sequential(
                    (nn.Linear(action_dim, hidden_dim)), nn.ReLU(),
                    (nn.Linear(hidden_dim, hidden_dim)))

        if args.gnn_add_state_act:
            assert args.use_state_mlp
            mlp_dim = hidden_dim
        else:
            mlp_dim = 2 * hidden_dim if args.use_state_mlp else hidden_dim + num_inputs
            if args.state_act_layer_norm:
                self.layer_norm = nn.LayerNorm(mlp_dim)

        if self.args.gnn_use_main_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(mlp_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim))
        else:
            self.mlp = nn.Linear(mlp_dim, hidden_dim)

        self.residual_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim if args.gnn_use_orig_state_act else hidden_dim,
                hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, x, add_input):
        # Concatenate state and action
        copy_add_input = add_input.clone()
        slate_index = (add_input[0] < 0).sum()
        if slate_index > 0:
            copy_add_input[add_input < 0] = 0
        aval_actions = copy_add_input.long()
        action_embs = self.dist_mem.get_action_embeddings(
            aval_actions, options=self.args.use_option_embs)
        act = self.action_linear(action_embs)
        if self.args.use_state_mlp:
            state = self.state_linear(x)
            if self.args.gnn_add_state_act:
                nodes = state.unsqueeze(1) + act
        else:
            state = x
        if not self.args.gnn_add_state_act:
            nodes = torch.cat([state.view([state.shape[0], 1, state.shape[1]]).repeat(1, act.shape[1], 1), act], dim=-1)
            if self.args.state_act_layer_norm:
                nodes = self.layer_norm(nodes)

        # Node-Non-Linearity
        nodes = self.mlp(nodes).squeeze(-1)

        # Graph Net Manipulation
        adj_mat = torch.ones([x.shape[0], act.shape[1], act.shape[1]], device=('cuda' if self.args.cuda else 'cpu'))
        if slate_index > 0:
            # Disconnect edges from the taken action nodes
            adj_mat[add_input < 0, :] = 0
            torch.transpose(adj_mat, 1, 2)[add_input < 0, :] = 0
        if self.args.gnn_use_action_summary:
            if self.args.summarizer_use_only_action:
                action_summary = self.gnn_action_summary(act, adj_mat)
            else:
                action_summary = self.gnn_action_summary(nodes, adj_mat)
            action_summary = torch.mean(action_summary, dim=1)
            nodes = torch.cat([nodes, action_summary.view(nodes.shape[0], 1, -1).repeat(1, nodes.shape[1], 1)], dim=-1)
            nodes = self.concat_summary_linear(nodes)
        if self.args.ablation_without_gnn:
            final_nodes = nodes
        else:
            for i in range(self.args.gnn_num_message_passing):
                final_nodes = self.utility_network(nodes, adj_mat).squeeze(-1)
                # Residual Connection
                if self.args.gnn_residual_connection and self.gnn_choice != GATSimple:
                    final_nodes += nodes
                nodes = final_nodes
            if self.args.gnn_gat_model.upper().startswith('GAT'):
                self.utility_network.get_attention_stats(self.args)

        if self.args.gnn_use_orig_state_act:
            final_nodes = torch.cat([
                final_nodes,
                state.view(state.shape[0], 1, state.shape[1]).repeat(1, act.shape[1], 1),
                act], dim=-1)
        final_nodes = self.residual_mlp(final_nodes).squeeze(-1)

        # Getting distribution
        if self.args.use_dist_double:
            final_nodes = final_nodes.double()
        if slate_index > 0:
            final_nodes[add_input < 0] = -1e15
        return FixedCategorical(logits=final_nodes)

class OrderInvariantCategorical(nn.Module):
    def __init__(self, num_inputs, dist_mem, args):
        super().__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        hidden_dim = args.dist_hidden_dim
        self.args = args
        action_dim = args.o_dim if args.use_option_embs else args.z_dim
        self.dist_mem = dist_mem

        if args.dist_linear_action:
            self.action_linear = init_(nn.Linear(action_dim, hidden_dim))
        else:
            self.action_linear = nn.Sequential(
                init_(nn.Linear(action_dim, hidden_dim)), nn.ReLU(),
                init_(nn.Linear(hidden_dim, hidden_dim)))

        if args.dist_non_linear_final:
            self.linear = nn.Sequential(
                init_(nn.Linear(hidden_dim + num_inputs, hidden_dim)), nn.ReLU(),
                init_(nn.Linear(hidden_dim, 1)))
        else:
            self.linear = init_(nn.Linear(hidden_dim + num_inputs, 1))

        # self.linear = init_(nn.Linear(hidden_dim + num_inputs, 1))

    def forward(self, x, add_input):
        aval_actions = add_input.long()
        action_embs = self.dist_mem.get_action_embeddings(
            aval_actions, options=self.args.use_option_embs)

        act = self.action_linear(action_embs)
        x = torch.cat([x.view([x.shape[0], 1, x.shape[1]]).repeat(1, act.shape[1], 1), act], dim=-1)
        x = self.linear(x).squeeze(-1)
        if self.args.use_dist_double:
            x = x.double()
        return FixedCategorical(logits=x)

class RndCategorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super().__init__()

        self.args = args
        self.num_outputs = num_outputs


    def forward(self, x, add_input):
        return FixedCategorical(logits=torch.rand((x.shape[0], self.num_outputs)).cuda())


def extract_aval(extra, infos):
    aval = []
    for i, info in enumerate(infos):
        aval.append(info['aval'])

    aval = torch.FloatTensor(aval)
    return aval

def init_extract_aval(args, evaluate=False):
    init_aval = torch.LongTensor(args.aval_actions)
    init_aval = init_aval.unsqueeze(0).repeat(
        args.eval_num_processes if (evaluate and args.eval_num_processes is not None) else args.num_processes,
        1)
    return init_aval

class MainMethod(BasePolicy):
    def __init__(self, args, obs_space, action_space):
        self.args = args
        self.emb_mem = args.emb_mem
        self.dist_mem = args.dist_mem
        self._is_slate = args.env_name.startswith('RecSim')

        '''
            1. Initialize actor critic 
            2. Modify action space for NEAREST NEIGHBOR Baseline
        '''
        super().__init__(args, obs_space, action_space, is_slate=self._is_slate)

    def _get_disc_policy(self, num_outputs):
        if num_outputs == 2 and self.args.separate_skip:
            return Categorical(self.actor_critic.base.output_size,
                    num_outputs, self.args)
        if self.args.random_policy:
            return RndCategorical(self.actor_critic.base.output_size,
                    num_outputs, self.args)
        elif self.args.mask_categorical:
            return MaskCategorical(self.actor_critic.base.output_size,
                    num_outputs, self.args)
        elif self.args.action_feature_categorical:
            return ActionFeatureCategorical(self.actor_critic.base.output_size,
                    self.dist_mem, self.args)
        elif self.args.nearest_neighbor:
            return Categorical(self.actor_critic.base.output_size,
                    num_outputs, self.args)
        elif self.args.gnn_ppo:
            return GraphCategorical(self.actor_critic.base.output_size,
                    self.dist_mem, self.args)
        else:
            return OrderInvariantCategorical(self.actor_critic.base.output_size,
                    self.dist_mem, self.args)


    def _create_actor_critic(self):
        super()._create_actor_critic()

        if self.args.fine_tune:
            for name, module in self.actor_critic.named_modules():
                if isinstance(module, Categorical):
                    self.ignore_layers.append(name)

    def get_dim_add_input(self):
        if self.args.env_name.startswith('RecSim'): return self.args.num_candidates
        return self.args.aval_actions.shape[-1]

    def get_add_input(self, extra, infos):
        return extract_aval(extra, infos)

    def get_init_add_input(self, args, evaluate=False):
        return init_extract_aval(args, evaluate=evaluate)

    def compute_fixed_action_set(self, take_action, aval_actions, args):
        if take_action.shape[-1] == 3:
            # This is a parameterized action space as with logic game or block
            # stacking.
            nn_result = self.emb_mem.nearest_neighbor_action(
                take_action[:, 0],
                args.training_fixed_action_set,
                aval_actions.long())
            if nn_result.shape[-1] == 1:
                nn_result = nn_result.squeeze(-1)
            take_action[:, 0] = nn_result
        else:
            take_action = self.emb_mem.nearest_neighbor_action(
                take_action.squeeze(-1),
                args.training_fixed_action_set,
                aval_actions.long())
            take_action = torch.LongTensor(
                np.round(take_action.cpu().numpy()))
        return take_action


    def get_action(self, state, add_input, recurrent_hidden_state,
                   mask, args, network=None, num_steps=None):
        # Sample actions
        with torch.no_grad():
            extra = {}
            parts = self.actor_critic.act(state, recurrent_hidden_state, mask,
                                     add_input=add_input,
                                     deterministic=args.deterministic_policy)
            value, action, action_log_prob, rnn_hxs, act_extra = parts
            if isinstance(act_extra, dict):
                act_extra = [act_extra]
            action_cpu = action.cpu().numpy()

            take_action = action_cpu
            if take_action.dtype == np.int64:
                take_action = torch.LongTensor(take_action)
            else:
                take_action = torch.Tensor(take_action)

            if args.load_fixed_action_set:
                take_action = self.compute_fixed_action_set(take_action,
                                                            add_input, args)

            if 'inferred_z' in act_extra:
                extra = {
                    **extra, **meth_utils.add_mag_stats(extra, act_extra[0]['inferred_z'])}

            entropy_reward = act_extra[0]['dist_entropy'].cpu() * args.reward_entropy_coef
            extra['alg_add_entropy_reward'] = entropy_reward.mean().item()
            extra['add_input'] = None
            if ((args.gnn_ppo and not args.ablation_without_gnn) or (args.action_set_summary and args.action_summarizer == 'gnn')) and args.gnn_gat_model.upper().startswith('GAT'):
                extra['alg_add_gat_attention_std'] = self.actor_critic.args.gat_attention_std
                extra['alg_add_gat_attention_max'] = self.actor_critic.args.gat_attention_max
                extra['alg_add_gat_attention_min'] = self.actor_critic.args.gat_attention_min

            add_reward = entropy_reward
            ac_outs = (value, action, action_log_prob, rnn_hxs)
            q_outs = (take_action, add_reward, extra)
            return ac_outs, q_outs

    def _get_action_emb(self, actions_idx):
        action_emb = self.dist_mem.get_action_embeddings(
            actions_idx[:, 0].long())
        if self.args.cuda:
            action_emb = action_emb.cuda()
        return action_emb

    def visualise_attention(self, add_input, file_name, action, title=None):
        ATTENTION_WEIGHT_MAGNIFY = 50
        assert (self.args.gnn_ppo or (self.args.action_set_summary and self.args.action_summarizer == 'gnn')) and self.args.gnn_gat_model.upper().startswith('GAT')
        if self.args.action_set_summary:
            attention_weight = self.actor_critic.base.action_set_summarizer.get_attention(first=True)
        elif self.args.conditioned_aux or self.args.env_name.startswith('MiniGrid'):
            attention_weight = self.actor_critic.dist.utility_network.get_attention(first=True)
        else:
            attention_weight = self.actor_critic.dist.disc_parts[0].utility_network.get_attention(first=True)
        # Currently only implemented for non-slate version
        assert len(attention_weight.shape) == 3

        selected_batch = np.random.choice(np.arange(attention_weight.shape[0]))
        embedding = attention_weight[selected_batch]

        candidate_tools = add_input[selected_batch].long().cpu().numpy()
        if self.args.env_name.startswith('Create'):
            tool_list = ToolGenerator(self.args.gran_factor).tools
            labels = [self.get_label(tool_list, j, i) for (i, j) in enumerate(candidate_tools)]
            plot_labels = [self.get_label(tool_list, j, i, False) for (i, j) in enumerate(candidate_tools)]
            category = [tool_list[i].required_activator() for i in candidate_tools]
        elif self.args.env_name.startswith('MiniGrid'):
            labels = [skill_types[x] for x in candidate_tools]
            plot_labels = [skill_types[x] for x in candidate_tools]
            category = [skill_category(x) for x in candidate_tools]
        else:
            raise NotImplementedError


        np.fill_diagonal(a=embedding, val=0.)
        embedding *= ATTENTION_WEIGHT_MAGNIFY
        _embedding = embedding.copy()

        action_node = action[selected_batch].int().cpu().numpy()[0]
        # random_node = np.random.choice(np.arange(1, _embedding.shape[0]))
        labels[action_node] = 'Action_' + labels[action_node]
        # selected_node = action_node if np.random.rand() < 0.75 else random_node
        selected_node = action_node

        mask = np.asarray([i == selected_node for i in range(_embedding.shape[0])])
        _embedding[~mask, :] = 0.0
        df = pd.DataFrame(_embedding, index=labels, columns=labels)
        if self.args.env_name.startswith('Create'):
            save_image = visualise(df=df, category=category, file_name=file_name,
                    save_dir=osp.join(self.args.attention_dir, self.args.env_name, self.args.prefix),
                    # rgb_values=['darkorange', 'aqua', 'yellow', 'violet', 'lightgray', 'springgreen'],
                    # mapping=['Fire', 'Water', 'Electric', 'Magnet', 'Spring', 'None'],
                    # rgb_values=['darkorange', 'aqua', 'yellow', 'violet', 'lightgray'],
                    rgb_values=['#ff9770', '#70d6ff', '#e9ff70', '#ff70a6', '#e8e9f3'],
                    mapping=['Fire', 'Water', 'Electric', 'Magnet', 'Spring'],
                    selected_column=df.columns[selected_node],
                    plot_labels=dict(zip(labels, plot_labels)),
                    env='create'
                    )
        elif self.args.env_name.startswith('MiniGrid'):
            save_image = visualise(df=df, category=category, file_name=file_name,
                    save_dir=osp.join(self.args.attention_dir, self.args.env_name, self.args.prefix),
                    rgb_values=['whitesmoke', 'palegreen', 'lightblue', 'orange', 'hotpink'],
                    mapping=['Step', 'Turn', 'Forward', 'Dig Orange', 'Dig Pink'],
                    selected_column=df.columns[selected_node],
                    plot_labels=dict(zip(labels, plot_labels)),
                    env='gw',
                    title=title
                    )
        return save_image

    def get_label(self, tool_list, tool_id, i, index=True):
        if tool_list[tool_id].tool_type not in ['no_op', 'Activator']:
            if index:
                return (str(i) + '_' + tool_list[tool_id].abbreviate()).lower()
            else:
                return (tool_list[tool_id].abbreviate()).lower()
        else:
            return (tool_list[tool_id].abbreviate()).lower()
