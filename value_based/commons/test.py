import argparse
import numpy as np
from random import sample

from value_based.commons.seeds import set_randomSeed


class TestCase(object):
    def prep(self):
        self._get_args()
        self._prep()

    def _get_args(self):
        from value_based.commons.args import cdqn_get_args
        ps = cdqn_get_args()
        self.args = ps.parse_args()

    def _prep(self):
        from value_based.commons.args import HISTORY_SIZE
        from value_based.encoder.obs_encoder import BasicObsEncoder
        from value_based.commons.launcher import launch_embedding
        from value_based.commons.observation import ObservationFactory

        # Assume the args have been updated by subclass of test, we get the additional args
        from value_based.commons.args import add_args
        self.args = add_args(args=self.args)

        # for reproducibility purpose
        set_randomSeed(seed=self.args.random_seed)

        # Test: Product Graph
        self.args.graph_dim_in = self.args.dim_item + self.args.dim_user
        # self.action_graph = Graph(dim_in=self.args.graph_dim_in,
        #                            dim_hidden=self.args.graph_dim_hidden,
        #                            dim_out=self.args.graph_dim_out,
        #                            args=vars(self.args))

        # necessary component
        self.dict_embedding = launch_embedding(args=self.args)
        self.item_embedding = self.dict_embedding["item"]

        self.baseItems_dict = {"train": list(range(self.args.num_trainItems))}

        # if self.args.env_name == "recsim":
        #     self.env = launch_env(dict_embedding=self.dict_embedding, args=self.args)
        # else:
        #     self.env = None

        # For test select action
        self.action_shape = (self.args.batch_size, self.args.slate_size)
        self.user_id = np.random.choice(a=self.args.num_allUsers, size=self.args.batch_size)
        self.history_seq = np.asarray([
            np.random.choice(a=self.args.num_trainItems, size=self.args.batch_size) for _ in range(HISTORY_SIZE)
        ]).T
        self.candidate_list = np.asarray(sample(list(range(self.args.num_trainItems)), self.args.num_candidates))
        self.check_duplication(_arr=self.candidate_list)

        # for test select_action of agent
        if self.args.env_name == "recsim":
            self.obses = self.next_obses = ObservationFactory(batch_size=self.args.batch_size)
            self.obses.load_history_seq(history_seq=self.history_seq)
            self.next_obses.load_history_seq(history_seq=self.history_seq)
        else:
            self.obses = self.next_obses = ObservationFactory(batch_size=self.args.batch_size, user_id=self.user_id)

        # for test update(Actions are sampled from available item list)
        self.user_id = np.random.choice(a=self.args.num_allUsers, size=self.args.batch_size)
        self.rewards = np.random.randn(self.args.batch_size, 1)
        self.actions = np.asarray(
            [np.random.choice(a=self.candidate_list, size=self.args.slate_size, replace=False)
             for _ in range(self.args.batch_size)]
        )
        self.dones = np.random.choice(a=[False, True], size=(self.args.batch_size, 1), p=[0.9, 0.1])
        self.candidate_lists = np.asarray([self.candidate_list for _ in range(self.args.batch_size)])
        self.check_duplication(_arr=self.candidate_lists)

        # batch_size x num_candidates x dim
        self.obs_encoder = BasicObsEncoder(args=vars(self.args))
        # self.state_embed = self.obs_encoder.encode(obs=self.obses)
        # self.state_embed = self.state_embed[:, None, :].repeat(1, self.args.num_candidates, 1)
        # self.slate_embed = torch.randn(self.args.batch_size, ML100K_NUM_ITEMS,
        #                                self.args.slate_encoder_dim_out)
        # self.gnn_embed = torch.randn(self.args.batch_size, ML100K_NUM_ITEMS,
        #                              self.args.act_encoder_dim_out)
        #
        # # User Choice Model
        # self.user_choice_model = CascadingClickModel(args=vars(self.args))
        #
        # # for Reward Model
        # state = self.obs_encoder.encode(obs=self.obses)
        # state = state[:, None, :].repeat(1, self.args.slate_size, 1)
        # item_feat = self.item_embedding.get(index=self.actions)
        # x = torch.cat([state, item_feat.type_as(state)], dim=-1)
        # self.x = x.reshape(shape=(x.shape[0] * x.shape[1], x.shape[2]))
        # y = torch.tensor(self.rewards, device=self.args.device)[:, None, :].repeat(1, self.args.slate_size, 1)
        # self.y = y.reshape(shape=(y.shape[0] * self.args.slate_size, 1))

    def sample_items(self):
        self.candidate_list = np.asarray(sample(list(range(self.args.num_trainItems)), self.args.num_candidates))
        self.check_duplication(_arr=self.candidate_list)

    @staticmethod
    def check_duplication(_arr):
        assert np.unique(ar=_arr, axis=-1).shape == _arr.shape, _arr

    @staticmethod
    def update_args_from_dict(args: argparse.Namespace, _dict: dict):
        _args = vars(args)
        _args.update(_dict)
        return argparse.Namespace(**_args)


if __name__ == '__main__':
    TestCase().prep()
