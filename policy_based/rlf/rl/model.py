import numpy as np
import torch
import torch.nn as nn
from policy_based.method.utils import Conv2d3x3
from policy_based.rlf.rl.utils import init
from policy_based.rlf.rl.distributions import get_action_mask
from gnn.GAT.models_geometric import GAT4 as GAT4_GEO
from gnn.GAT.models_geometric import GAT2 as GAT2_GEO
from gnn.GAT.models import GAT2, GATSimple
from gnn.GNN.model import GCN2
from gnn.DeepSet.models import DeepSet, BiLSTM


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_space, args,
                 base=None,
                 dist_mem=None,
                 z_dim=None,
                 add_input_dim=0,
                 is_slate=False):
        super().__init__()
        self.obs_shape = obs_shape
        self.add_input_dim = add_input_dim
        self.is_slate = is_slate
        if self.is_slate:
            add_obs_dim = (args.slate_size - 1) * args.dim_item
            assert len(obs_shape) == 1
            obs_shape = tuple([obs_shape[0] + add_obs_dim])

        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase_NEW
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError(
                    'Observation space is %s' % str(obs_shape))

        self.action_space = action_space
        self.args = args
        self.env_name = args.env_name

        use_action_output_size = 0

        self.base = base(obs_shape[0], add_input_dim,
                         action_output_size=use_action_output_size,
                         recurrent=args.recurrent_policy, hidden_size=args.state_encoder_hidden_size,
                         use_batch_norm=args.use_batch_norm, args=args)

    def clone_fresh(self):
        p = Policy(self.obs_shape, self.action_space, self.args,
                   type(self.base) if self.base is not None else None,
                   self.add_input_dim)

        if list(self.parameters())[0].is_cuda:
            p = p.cuda()

        return p

    def get_policies(self):
        return [self]

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def get_pi(self, inputs, rnn_hxs, masks, add_input=None):
        value, actor_features, rnn_hxs, action_features = self.base(
            inputs, rnn_hxs, masks, add_input)

        dist = self.dist(actor_features, add_input, action_features)
        return dist, value

    def act(self, inputs, rnn_hxs, masks, deterministic=False, add_input=None):
        dist, value = self.get_pi(
            inputs, rnn_hxs, masks, add_input)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        self.prev_action = action

        action_log_probs = dist.log_probs(action.double() if self.args.use_dist_double else action)
        dist_entropy = dist.entropy()

        if self.args.use_dist_double:
            if self.action_space.__class__.__name__ != "Discrete":
                action = action.float()
            action_log_probs = action_log_probs.float()
            dist_entropy = dist_entropy.float()
        if len(dist_entropy.shape) == 1:
            dist_entropy = dist_entropy.unsqueeze(-1)
        extra = {
            'dist_entropy': dist_entropy
        }

        return value, action, action_log_probs, rnn_hxs, extra

    def get_value(self, inputs, rnn_hxs, masks, action, add_input):
        value, _, _, _ = self.base(inputs, rnn_hxs, masks, add_input)

        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, add_input):
        value, actor_features, rnn_hxs, action_features = self.base(inputs, rnn_hxs, masks, add_input)

        dist = self.dist(actor_features, add_input, action_features)


        action_log_probs = dist.log_probs(action.double() if self.args.use_dist_double else action)
        dist_entropy = dist.entropy()

        if self.args.use_dist_double:
            action_log_probs = action_log_probs.float()
            dist_entropy = dist_entropy.float()

        return value, action_log_probs, dist_entropy, rnn_hxs, dist


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size, args):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self.args = args
        self._recurrent = recurrent

        if args.gnn_gat_model.upper() == 'GAT2':
            self.gnn_choice = GAT2
        elif args.gnn_gat_model.upper() == 'GAT_SIMPLE':
            self.gnn_choice = GATSimple
        elif args.gnn_gat_model.upper() == 'GAT2_GEO':
            self.gnn_choice = GAT2_GEO
        elif args.gnn_gat_model.upper() == 'GAT4_GEO':
            self.gnn_choice = GAT4_GEO
        elif args.gnn_gat_model.upper() == 'GCN2':
            self.gnn_choice = GCN2
        else:
            raise NotImplementedError

        if args.action_summarizer == 'deep_set':
            assert args.method_name == 'summary'
            self.gnn_choice = DeepSet
        elif args.action_summarizer in ['lstm', 'bilstm']:
            assert args.method_name == 'summary'
            self.gnn_choice = BiLSTM

        self.dist_mem = args.dist_mem
        action_dim = self.dist_mem.option_embs.shape[-1]
        """ Get Action Features """
        if args.dist_linear_action:
            # Default is True
            self.action_linear = nn.Linear(action_dim, hidden_size)
        else:
            self.action_linear = nn.Sequential(
                    (nn.Linear(action_dim, hidden_size)), nn.ReLU(),
                    (nn.Linear(hidden_size, hidden_size)))

        """ Action Summary and Features """
        if args.action_set_summary:
            if args.input_mask_categorical:
                self.total_actions = args.env_total_train_actions
                if args.input_mask_mlp:
                    self.mask_action_set_summarizer = nn.Sequential(
                            nn.Linear(self.total_actions, hidden_size), nn.ReLU(),
                            nn.Linear(hidden_size, hidden_size))
            else:
                pre_summarize_mlp_dim = hidden_size if args.summarizer_use_only_action else 2 * hidden_size
                self.pre_summarize_mlp = nn.Sequential(
                    nn.Linear(pre_summarize_mlp_dim, hidden_size), nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size))
                self.action_set_summarizer = self.gnn_choice(
                         dim_in=hidden_size,
                         dim_hidden=hidden_size,
                         dim_out=hidden_size,
                         num_heads=args.gat_num_attention_heads,
                         args=args)
                if args.separate_critic_action_set_summary:
                    if self.args.separate_critic_summary_nodes:
                        self.critic_pre_summarize_mlp = nn.Sequential(
                            nn.Linear(pre_summarize_mlp_dim, hidden_size), nn.ReLU(),
                            nn.Linear(hidden_size, hidden_size))
                    self.critic_action_set_summarizer = self.gnn_choice(
                             dim_in=hidden_size,
                             dim_hidden=hidden_size,
                             dim_out=hidden_size,
                             num_heads=args.gat_num_attention_heads,
                             args=args)
                if args.concat_relational_action_features:
                    self.concat_act_linear = nn.Linear(2 * hidden_size, hidden_size)

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

    def mask_action_summary(self, state, add_input):
        action_mask = get_action_mask(state.shape[0], self.total_actions, add_input.long(), self.args)
        if self.args.input_mask_mlp:
            action_mask = self.mask_action_set_summarizer(action_mask)
        state = torch.cat([state, action_mask], dim=-1)
        return state, None

    def action_summary(self, state, act, critic_state):
        """
        critic_state: Only possibly different from state for relational if args.separate_critic_state == True
        """
        if self.args.summarizer_use_only_action:
            nodes = act
        else:
            nodes = torch.cat([state.view([state.shape[0], 1, state.shape[1]]).repeat(1, act.shape[1], 1), act], dim=-1)
            nodes = self.pre_summarize_mlp(nodes).squeeze(-1)
        if self.args.action_summarizer == 'gnn':
            adj_mat = torch.ones([state.shape[0], act.shape[1], act.shape[1]], device=('cuda' if self.args.cuda else 'cpu'))
            if self.args.gnn_gat_model.upper() == 'GCN2':
                adj_mat -= torch.eye(act.shape[1], act.shape[1], device=('cuda' if self.args.cuda else 'cpu'))
            relational_action_features = self.action_set_summarizer(nodes, adj_mat).squeeze(-1)
            if self.args.gnn_gat_model.upper().startswith('GAT'):
                self.action_set_summarizer.get_attention_stats(self.args)
            if self.args.separate_critic_action_set_summary:
                if self.args.summarizer_use_only_action:
                    summary_nodes = act
                elif self.args.separate_critic_summary_nodes:
                    summary_nodes = torch.cat([
                        critic_state.view(
                            [critic_state.shape[0], 1, critic_state.shape[1]]
                            ).repeat(1, act.shape[1], 1),
                        act], dim=-1)
                    summary_nodes = self.critic_pre_summarize_mlp(summary_nodes).squeeze(-1)
                else:
                    summary_nodes = nodes
                summary_action_features = self.critic_action_set_summarizer(summary_nodes, adj_mat).squeeze(-1)
                action_summary = torch.mean(summary_action_features, dim=1)
            else:
                action_summary = torch.mean(relational_action_features, dim=1)
            if self.args.gnn_residual_connection and self.gnn_choice != GATSimple:
                relational_action_features += nodes
        elif self.args.action_summarizer in ['lstm', 'bilstm']:
            action_summary = self.action_set_summarizer(nodes)
        elif self.args.action_summarizer == 'deep_set':
            action_summary = self.action_set_summarizer(nodes)
        else:
            raise NotImplementedError
        state = torch.cat([state, action_summary], dim=-1)

        if self.args.method_name.lower() == 'relational':
            assert self.args.action_summarizer == 'gnn'
            action_features = relational_action_features
            if self.args.concat_relational_action_features:
                action_features = torch.cat([relational_action_features, act], -1)
                action_features = self.concat_act_linear(action_features)
        else:
            action_features = act
        return state, action_features

    def compute_action_features(self, add_input=None):
        if add_input is None:
            return None
        aval_actions = add_input.long()
        action_embs = self.dist_mem.get_action_embeddings(
            aval_actions, options=self.args.use_option_embs)
        act = self.action_linear(action_embs)
        return act




class CNNBase_NEW(NNBase):
    def __init__(self, num_inputs, add_input_dim,
                 action_output_size=32,
                 recurrent=False, hidden_size=64,
                 use_batch_norm=False, args=None):
        super().__init__(recurrent, hidden_size, hidden_size, args)

        self.conv_layers = nn.ModuleList([
            Conv2d3x3(in_channels=num_inputs,
                      out_channels=16, downsample=True),
            # shape is now (-1, 16, 42, 42)
            Conv2d3x3(in_channels=16, out_channels=16, downsample=True),
            # shape is now (-1, 16, 21, 21)
            Conv2d3x3(in_channels=16, out_channels=32, downsample=True),
            # shape is now (-1, 16, 11, 11)
            Conv2d3x3(in_channels=32, out_channels=32, downsample=True),
            # shape is now (-1, 32, 6, 6)
            Conv2d3x3(in_channels=32, out_channels=32, downsample=True),
            # shape is now (-1, 32, 3, 3)
        ])


        self.flat_size = 32 * 3 * 3

        def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.
                                  constant_(x, 0), nn.init.calculate_gain('relu'))

        if args.use_state_mlp:
            # Default is True
            self.state_mlp = nn.Sequential(
                (nn.Linear(self.flat_size, hidden_size)), nn.ReLU(),
                (nn.Linear(hidden_size, hidden_size)))
        else:
            self.state_mlp = nn.Linear(self.flat_size, hidden_size)

        if self.args.separate_critic_state:
            self.separate_conv_layers = nn.ModuleList([
                Conv2d3x3(in_channels=num_inputs, out_channels=16, downsample=True),
                Conv2d3x3(in_channels=16, out_channels=16, downsample=True),
                Conv2d3x3(in_channels=16, out_channels=32, downsample=True),
                Conv2d3x3(in_channels=32, out_channels=32, downsample=True),
                Conv2d3x3(in_channels=32, out_channels=32, downsample=True),
            ])
            if args.use_state_mlp:
                # Default is True
                self.separate_state_mlp = nn.Sequential(
                    (nn.Linear(self.flat_size, hidden_size)), nn.ReLU(),
                    (nn.Linear(hidden_size, hidden_size)))
            else:
                self.separate_state_mlp = nn.Linear(self.flat_size, hidden_size)

        self.nonlinearity = nn.ReLU()
        self.raw_state_emb = None
        self.hidden_size = hidden_size

        if args.action_set_summary and args.input_mask_categorical:
            if args.input_mask_mlp:
                state_final_dim = 2 * hidden_size
            else:
                state_final_dim = hidden_size + self.total_actions
        elif args.action_set_summary:
            state_final_dim = 2 * hidden_size
        else:
            state_final_dim = hidden_size

        self.actor = nn.Sequential(
            init_(nn.Linear(state_final_dim, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        self.critic = nn.Sequential(
            init_(nn.Linear(state_final_dim, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.train()

    def forward(self, inputs, rnn_hxs, masks, add_input):
        if inputs.dtype == torch.uint8:
            inputs = inputs.float()
        x = inputs
        for conv in self.conv_layers:
            x = conv(x)
            x = self.nonlinearity(x)
        x = x.view(-1, self.flat_size)
        state = self.state_mlp(x)
        if self.args.separate_critic_state:
            separate_x = inputs
            for conv in self.separate_conv_layers:
                separate_x = conv(separate_x)
                separate_x = self.nonlinearity(separate_x)
            separate_x = separate_x.view(-1, self.flat_size)
            separate_state = self.separate_state_mlp(separate_x)
        else:
            separate_state = state

        self.raw_state_emb = state.clone()

        if self.is_recurrent:
            state, rnn_hxs = self._forward_gru(state, rnn_hxs, masks)

        action_features = self.compute_action_features(add_input)

        if self.args.action_set_summary and self.args.input_mask_categorical:
            state, action_features = self.mask_action_summary(state, add_input)
        elif self.args.action_set_summary:
            state, action_features = self.action_summary(state, action_features, separate_state)

        hidden_critic = self.critic(state)
        hidden_actor = self.actor(state)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs, action_features


class MLPBase(NNBase):
    def __init__(self, num_inputs, add_input_dim,
                 action_output_size=32,
                 recurrent=False, hidden_size=64,
                 use_batch_norm=False, args=None):
        super().__init__(recurrent, num_inputs, hidden_size, args)

        if recurrent:
            num_inputs = hidden_size

        def init_(m): return init(m, nn.init.orthogonal_, lambda x: nn.init.
                                  constant_(x, 0), np.sqrt(2))

        if args.use_state_mlp:
            # Default is True
            self.state_mlp = nn.Sequential(
                (nn.Linear(num_inputs, hidden_size)), nn.ReLU(),
                (nn.Linear(hidden_size, hidden_size)))
        else:
            self.state_mlp = nn.Linear(num_inputs, hidden_size)

        if self.args.separate_critic_state:
            if args.use_state_mlp:
                # Default is True
                self.separate_state_mlp = nn.Sequential(
                    (nn.Linear(num_inputs, hidden_size)), nn.ReLU(),
                    (nn.Linear(hidden_size, hidden_size)))
            else:
                self.separate_state_mlp = nn.Linear(num_inputs, hidden_size)


        if args.action_set_summary and args.input_mask_categorical:
            if args.input_mask_mlp:
                state_final_dim = 2 * hidden_size
            else:
                state_final_dim = hidden_size + self.total_actions
        elif args.action_set_summary:
            state_final_dim = 2 * hidden_size
        else:
            state_final_dim = hidden_size
        self.actor = nn.Sequential(
            init_(nn.Linear(state_final_dim, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(state_final_dim, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(
            nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks, add_input):
        x = inputs
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        action_features = self.compute_action_features(add_input)
        state = self.state_mlp(x)
        if self.args.separate_critic_state:
            separate_state = self.separate_state_mlp(x)
        else:
            separate_state = state

        if self.args.action_set_summary and self.args.input_mask_categorical:
            state, action_features = self.mask_action_summary(state, add_input)
        elif self.args.action_set_summary:
            state, action_features = self.action_summary(state, action_features, separate_state)

        hidden_critic = self.critic(state)
        hidden_actor = self.actor(state)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs, action_features
