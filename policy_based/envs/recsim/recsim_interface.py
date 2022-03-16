from policy_based.envs.action_env import get_aval_actions
from policy_based.envs.action_env_interface import ActionEnvInterface
from value_based.envs.recsim.environments import interest_evolution
from value_based.embedding.base import BaseEmbedding
import torch

class RecSimInterface(ActionEnvInterface):
    def __init__(self, args):
        self.args = args

    def setup(self, args, task_id):
        super().setup(args, task_id)
        # TODO: Check if overall_aval_actions need to be set here. It might be needed for standard RL 1 and 2 baselines
        # args.overall_aval_actions = get_aval_actions(args, 'reco')


    def env_trans_fn(self, env, set_eval):
        env = super()._generic_setup(env, set_eval)
        return env

    def get_gt_embs(self):
        env = interest_evolution.create_multiuser_environment(args=vars(self.args))
        item_class = BaseEmbedding()
        item_class.load(embedding=env.item_embedding)
        return item_class.get_all()

    def get_id(self):
        return 'RE'

    def get_special_stat_names(self):
        return ['ep_avg_ctr']

    def get_env_option_names(self):
        indv_labels = [('Train' if ind in self.train_action_set else 'Test') for ind in self.train_test_action_set]
        label_list = sorted(list(set(indv_labels)))
        return indv_labels, label_list
