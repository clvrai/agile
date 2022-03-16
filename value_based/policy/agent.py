import os
import torch
import numpy as np
from typing import Dict, List
from torch import nn
from copy import deepcopy

from value_based.policy import RandomPolicy
from value_based.commons.observation import ObservationFactory
from value_based.encoder.slate_encoder import SlateEncoder
from value_based.encoder.obs_encoder import ObsEncoder
from value_based.encoder.act_encoder import ActEncoder
from value_based.embedding.base import BaseEmbedding
from value_based.commons.test import TestCase as test
from value_based.commons.utils import softmax, logging

AGENTS_WHICH_NEEDS_GNN_LIST = ["rcdqn", "r3cdqn"]


class Agent(object):
    """ Boilerplate of agent """

    def __init__(self,
                 obs_encoder: List[ObsEncoder] = None,
                 slate_encoder: List[SlateEncoder] = None,
                 dict_embedding: Dict[str, BaseEmbedding] = None,
                 act_encoder: List[ActEncoder] = None,
                 args: dict = {}):
        # basic args
        self._args = args
        self._if_debug = self._args.get("if_debug", False)
        self._agent_type = self._args.get("agent_type", "cdqn")
        self._graph_type = self._args.get("graph_type", "gat")
        self._if_item_retrieval = self._args.get("if_item_retrieval", False) if self._graph_type != "None" else False
        self._num_candidates = self._args.get("num_candidates", 5000)
        self._num_intermediate_candidates = self._args.get("num_intermediate_candidates", 50)
        self._device = self._args.get("device", "cpu")
        self._slate_size = self._args.get("slate_size", 5)
        self._batch_size = self._args.get("batch_size", 100)
        self._if_sequentialQNet = self._args.get("if_sequentialQNet", True)
        self._dim_in = self._args.get("dim_in", 32)
        self._dim_hidden = self._args.get("q_net_dim_hidden", "256_32")
        self._lr = self._args.get("q_net_lr", 1e-4)
        if self._args.get("use_intra_slate_gamma"):
            self._gamma = torch.tensor([1, self._args.get("gamma", 0.99)], device=self._device)
        else:
            self._gamma = self._args.get("gamma", 0.99)
        self._grad_clip = self._args.get("grad_clip", 1.0)
        self._if_single_optimiser = self._args.get("if_single_optimiser", False)
        self._if_one_shot_instantiation = self._args.get("if_one_shot_instantiation", False)
        self._if_check_grad, self._if_visualise, self._train = False, False, True
        self._rng = np.random.RandomState(self._args.get("random_seed", 2021))

        # Convert all the items into item embedding space
        self.main_action_linear = nn.Linear(self._args["dim_item"], self._args["dim_item_hidden"]).to(self._device)
        if self._args["if_use_main_target_for_others"]:
            self.target_action_linear = nn.Linear(
                self._args["dim_item"], self._args["dim_item_hidden"]).to(self._device)
            self.target_action_linear.load_state_dict(self.main_action_linear.state_dict())

        # Update the arg for StandardRL agents
        if self._args["recsim_if_special_items"]:
            self._num_trainItems = self._args["num_trainItems"] + self._args["recsim_num_categories"]
        else:
            self._num_trainItems = self._args["num_trainItems"]

        # instantiate the necessary components
        self.epoch = 0  # for summary writer
        self.dict_embedding = dict_embedding
        self.main_obs_encoder, self.main_slate_encoder = obs_encoder[0], slate_encoder[0]
        if self._args["if_use_main_target_for_others"]:
            self.target_obs_encoder, self.target_slate_encoder = obs_encoder[1], slate_encoder[1]
        self.random_policy = RandomPolicy(slate_size=self._slate_size)
        self.out_info = None

    def select_action(self, obs: ObservationFactory, candidate_list: np.ndarray, epsilon: float):
        """ Produce a slate or a batch of slates

        Args:
            obs (ObservationFactory): batch_step_size x history_seq
            candidate_list (np.ndarray): (num_candidates)-size array
            epsilon (float): float

        Returns:
            slate: batch_step_size x slate_size
        """
        # batch_step_size x history_size x dim_obs
        obs = obs.make_obs(dict_embedding=self.dict_embedding, if_separate=False, device=self._device)

        # Instantiate the main action base and we will fill this using random policy and agent
        action = np.zeros((obs.shape[0], self._slate_size))  # batch_step_size x slate_size

        # === Epsilon decay policy
        _rand = self._rng.uniform(low=0.0, high=1.0, size=obs.shape[0])  # With prob epsilon select a random action
        mask = _rand < epsilon  # epsilon decay
        random_policy_obs, policy_obs = obs[mask, :], obs[~mask, :]  # total would be: batch_step_size x dim_state

        # === Select actions using the random policy
        action_random_ind = self.random_policy.select_action(batch_size=np.sum(mask), candidate_list=candidate_list)

        if not np.alltrue(mask):
            # === Select actions using the policy
            if self.main_obs_encoder is not None:
                # We use the sequential model to deal with the user's historical seq
                state_embed = self.main_obs_encoder.encode(obs=policy_obs)  # batch_policy_size x dim_state
            else:
                # We assume that Q-net can handle obs directly; batch_policy_size x dim_obs
                state_embed = torch.tensor(policy_obs, dtype=torch.float32, device=self._device)

            # get the corresponding item_embedding; batch_step_size x num_candidates x dim_item
            if self._if_sequentialQNet:
                if self._if_item_retrieval:
                    state_embed = state_embed[:, None, :].repeat(1, policy_obs.shape[1], 1)
                else:
                    if self._args["agent_standardRL_type"] != "None":  # todo: we don't need this condition
                        # batch_policy_size x num_allItems x dim_gnn_embed
                        state_embed = state_embed[:, None, :].repeat(1, self._num_trainItems, 1)
                    else:
                        # batch_policy_size x num_candidates x dim_gnn_embed
                        state_embed = state_embed[:, None, :].repeat(1, self._num_candidates, 1)

            if self._args["agent_standardRL_type"] != "None":
                # batch_policy_size x num_trainItems x dim_item
                _items = np.tile(A=np.arange(self._num_trainItems), reps=(policy_obs.shape[0], 1))
                item_embed = self.dict_embedding["item"].get(index=_items)
            else:
                # batch_policy_size x num_candidates x dim_item
                item_embed = self.dict_embedding["item"].get(index=candidate_list)
                item_embed = item_embed[None, ...].repeat(policy_obs.shape[0], 1, 1)

            # Prep the input
            inputs = self._prep_inputs(state_embed=state_embed,
                                       item_embed=item_embed,
                                       candidate_list=candidate_list,
                                       if_update=False)

            if self._args["agent_standardRL_type"] == "None":
                action_policy_ind = self._select_action(inputs=inputs, epsilon=epsilon)
                # Test.check_duplication(action_policy_ind)
                # Convert the indices of Q-values into itemIds
                action_policy_id = self.ind2id(action=action_policy_ind, candidate_list=candidate_list)
            else:
                action_policy_ind = self._select_action(inputs=inputs, epsilon=epsilon)
                action_policy_id = self.ind2id(action=action_policy_ind, candidate_list=self._baseItems_dict["train"])
                # Test.check_action(action=action_policy_id, candidate_list=self._baseItems_dict["train"])

            # Concatenate with the random action
            if action_random_ind.shape[0] == 0:
                action = action_policy_id
            else:
                action[mask, :] = action_random_ind
                action[~mask, :] = action_policy_id
        else:
            action = action_random_ind  # batch_step_size x slate_size

        action = np.asarray(action).astype(np.int)  # batch_step_size x slate_size
        # Test.check_action(action=action, candidate_list=candidate_list)
        if self._if_visualise:  # this is supposed to be used non Random agent so that inputs should exist!!
            # state_embed: batch_step_size x dim_state
            # gnn_embed: batch_step_size x (dim_node or dim_global)
            # slate_embed: batch_step_size x slate_size x dim_slate
            self.out_info = {
                # "state_embed": state_embed[:, 0, :].cpu().detach().numpy() if state_embed is not None else None,
                # "gnn_embed": gnn_embed[:, 0, :].cpu().detach().numpy() if gnn_embed is not None else None,
                # See cdqn.py for more details!!
                # "slate_embed": np.asarray(self._slate_embed) if getattr(self, "_slate_embed", False) else None,
                "action_ind": action_policy_ind
            }
            attention_weight = None
            if self._args["agent_type"] == "agile":
                attention_weight = self.action_graph.get_attention(first=True)
                if torch.is_tensor(attention_weight):
                    attention_weight = attention_weight.cpu().detach().numpy()
            elif self._args["agent_type"] in ["cdqn", "gcdqn"]:
                if self._args.get("graph_type", "gcn").startswith("gat"):
                    if self._args["gcdqn_twin_GAT"]:
                        attention_weight = self.main_Q_net_node.get_attention(stack=True, first=True)
                    else:
                        attention_weight = self.main_Q_net.get_attention(stack=True, first=True)
                if attention_weight is not None:
                    if torch.is_tensor(attention_weight):
                        attention_weight = attention_weight.cpu().detach().numpy()
            # batch_size x num_nodes x num_nodes or batch_size x intra_slate_step x num_nodes x num_nodes
            self.out_info.update({"attention_weight": attention_weight})
        return action

    def get_out_info(self):
        return self.out_info

    def _select_action(self, inputs: list, epsilon: float):
        """ Inner method of select_action

        Args:
            inputs:
                state_embed (torch.tensor): batch_step_size x num_candidates x dim
                item_embed (torch.tensor): batch_step_size x num_candidates x dim
                gnn_embed (torch.tensor): batch_step_size x num_candidates x dim
                if agent_standardRL_type is either '1' or '2' in string, then
                    availability_mask (np.ndarray): batch_step_size x num_candidates

        Returns:
            slate: batch_step_size x slate_size; Index based slate so that needs to be converted into itemIds later!
        """
        raise NotImplementedError

    def _prep_inputs(self, state_embed, item_embed, candidate_list, if_update: bool = False, if_next: bool = False):
        """ Collect the input components

        Args:
            state_embed (torch.tensor): batch_step_size x num_candidates x dim
            item_embed (torch.tensor): batch_step_size x num_candidates x dim
            candidate_list (np.ndarray): batch_step_size x num_candidates
            if_update (bool): if this is called in select_action or update

        Returns:
            inputs:
                state_embed (torch.tensor): batch_step_size x num_candidates x dim
                item_embed (torch.tensor): batch_step_size x num_candidates x dim
                if agent_standardRL_type is either '1' or '2' in string, then
                    availability_mask (np.ndarray): batch_step_size x num_candidates
        """
        # Transform item-embedding
        if self._args["if_use_main_target_for_others"] and if_next:
            item_embed = self.target_action_linear(item_embed)
        else:
            item_embed = self.main_action_linear(item_embed)

        if self._args["agent_standardRL_type"] != "None":
            if if_update:
                availability_ind = self.id2ind(action=candidate_list,
                                               candidate_lists=np.tile(A=self._baseItems_dict["train"],
                                                                       reps=(candidate_list.shape[0], 1)))
                availability_mask = np.ones([availability_ind.shape[0], self._num_trainItems]).astype(np.bool)
                availability_mask[np.arange(availability_ind.shape[0])[:, None], availability_ind] = False
            else:
                availability_ind = self.id2ind(action=np.array(candidate_list)[None, :],
                                               candidate_lists=np.array(self._baseItems_dict["train"])[None, :])[0]
                _base = np.ones(self._num_trainItems)
                _base[availability_ind] = 0.0
                _base = _base.astype(np.bool)
                availability_mask = np.tile(A=_base[None, :], reps=(state_embed.shape[0], 1))
            inputs = [state_embed, item_embed, availability_mask]
        else:
            inputs = [state_embed, item_embed]
        inputs = [i for i in inputs if i is not None]
        return inputs

    @staticmethod
    def ind2id(action, candidate_list):
        """ Convert the indices into itemIds

        Args:
            action: batch_policy_size x slate_size
            candidate_list: (num_candidate)-size array or batch_policy_size x slate_size

        Returns:
            action: batch_policy_size x slate_size
        """
        if type(candidate_list) == list:
            candidate_list = np.asarray(candidate_list)
        if len(candidate_list.shape) == 1:
            return np.asarray(candidate_list)[action]
        elif len(candidate_list.shape) > 1:
            _result = list()
            for user_id in range(candidate_list.shape[0]):
                _result.append(candidate_list[user_id, action[user_id, :]].tolist())
            return np.asarray(_result)
        else:
            raise ValueError

    @staticmethod
    def id2ind(action, candidate_lists):
        """ Recover the index of an itemId from the candidate list at each time-step in the mini-batch from ReplayBuffer

        Args:
            action: batch_size x slate_size
            candidate_lists: batch_size x num_candidates

        Returns:
            action: batch_size x slate_size
        """
        _actions = list()
        for slate, candidate_list in zip(action, candidate_lists):
            _actions.append([candidate_list.tolist().index(i) for i in slate])
        return np.asarray(_actions)

    def update(self,
               obses: ObservationFactory,
               actions: np.ndarray,
               rewards: np.ndarray,
               next_obses: ObservationFactory,
               dones: np.ndarray,
               candidate_lists: np.ndarray):
        """ Inner update method

        Args:
            obses (ObservationFactory):
            actions (np.ndarray): batch_size x slate_size
            rewards (np.ndarray): batch_size x 1
            next_obses (ObservationFactory):
            dones (np.ndarray): batch_size x 1
            candidate_lists (np.ndarray): batch_size x num_candidates

        Returns:
            result: dict of intermediate info during update
        """
        obses = obses.make_obs(dict_embedding=self.dict_embedding, device=self._device)
        next_obses = next_obses.make_obs(dict_embedding=self.dict_embedding, device=self._device)

        if self.main_obs_encoder is not None:
            if self._args["if_use_main_target_for_others"]:
                # We use the sequential model to deal with the user's historical seq
                state_embed = self.main_obs_encoder.encode(obs=obses)  # batch_size x dim_item
                next_state_embed = self.target_obs_encoder.encode(obs=next_obses)  # batch_size x dim_item
            else:
                state_embed = self.main_obs_encoder.encode(obs=obses)  # batch_size x dim_item
                next_state_embed = self.main_obs_encoder.encode(obs=next_obses)  # batch_size x dim_item
        else:
            # We assume that Q-net can handle obs directly; batch_policy_size x dim_obs
            state_embed = torch.tensor(obses, dtype=torch.float32, device=self._device)
            next_state_embed = torch.tensor(next_obses, dtype=torch.float32, device=self._device)

        if self._if_sequentialQNet:
            if self._args["agent_standardRL_type"] != "None":
                # batch_size x num_candidates x dim_item
                state_embed = state_embed[:, None, :].repeat(1, self._num_trainItems, 1)
                next_state_embed = next_state_embed[:, None, :].repeat(1, self._num_trainItems, 1)
            else:
                # batch_size x num_candidates x dim_item
                state_embed = state_embed[:, None, :].repeat(1, candidate_lists.shape[-1], 1)
                next_state_embed = next_state_embed[:, None, :].repeat(1, state_embed.shape[1], 1)

        if self._args["agent_standardRL_type"] != "None":
            # batch_size x num_allItems x dim_gnn_embed
            _items = np.tile(A=self._baseItems_dict["train"], reps=(next_state_embed.shape[0], 1))
            next_item_embed = item_embed = self.dict_embedding["item"].get(index=_items)
        else:
            # get the corresponding item_embedding; batch_size x num_candidates x dim_item
            item_embed = self.dict_embedding["item"].get(index=candidate_lists)
            next_item_embed = self.dict_embedding["item"].get(index=candidate_lists)

        # Prep the inputs to the Policy
        inputs = self._prep_inputs(state_embed=state_embed,
                                   item_embed=item_embed,
                                   candidate_list=candidate_lists,  # this is okay for SRL, too
                                   if_update=True)
        next_inputs = self._prep_inputs(state_embed=next_state_embed,
                                        item_embed=next_item_embed,
                                        # we assume the candidate-set doesn't change across time-steps
                                        candidate_list=candidate_lists,  # this is okay for SRL, too
                                        if_update=True,
                                        if_next=True)

        # Convert the itemIds into indices in Q-net
        if self._args["agent_standardRL_type"] == "None":
            actions = self.id2ind(action=actions, candidate_lists=candidate_lists)
        else:
            actions = self.id2ind(action=actions, candidate_lists=np.tile(A=self._baseItems_dict["train"],
                                                                          reps=(actions.shape[0], 1)))

        # convert them into tensors
        actions = torch.tensor(actions, device=self._device)
        rewards = torch.tensor(rewards.astype(np.float32), device=self._device)
        reversed_dones = torch.tensor((1 - dones).astype(np.float32), device=self._device)
        if len(reversed_dones.shape) == 1:
            reversed_dones = reversed_dones[:, None]

        return self._update(inputs=inputs,
                            next_inputs=next_inputs,
                            actions=actions,
                            rewards=rewards,
                            reversed_dones=reversed_dones)

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
            next_state_embed (torch.tensor): batch_size x dim_state
            reversed_dones (torch.tensor): batch_size x 1

        Returns:
            result (dict): a dict of intermediate info during update
        """

        raise NotImplementedError

    def minor_update(self, **kwargs):
        """ Second update method for minor things in agents """
        self._minor_update(**kwargs)

    def _minor_update(self, **kwargs):
        pass

    def _prep_input_standardRL(self, inputs: list):
        """ Extract the necessary input from the list of input components and returns in list

        Args:
            if self._if_sequentialQNet:
                if agent_standardRL_type == None:
                    inputs: (state_embed, item_embed) -> batch_step_size x num_candidates x dim
                else:
                    inputs: (state_embed, availability_mask)
                        - state_embed: batch_step_size x num_trainItems x dim
                        - availability_mask: batch_step_size x num_trainItems
            else:
                if agent_standardRL_type != None:
                    inputs: (state_embed, item_embed) -> batch_step_size x dim
                else:
                    inputs: (state_embed, item_embed, availability_mask)
                        - state_embed: batch_step_size x dim
                        - availability_mask: batch_step_size x num_trainItems
        """
        [state_embed, _, availability_mask] = inputs
        availability_mask = torch.tensor(availability_mask, device=self._device)
        return state_embed, availability_mask

    def increment_epoch(self, _v=1):
        self.epoch += _v

    def set_if_check_grad(self, flg: bool):
        self._if_check_grad = flg

    def set_if_visualise(self, flg: bool):
        self._if_visualise = flg

    def load(self, save_dir: str, epoch: int = 0):
        ep = epoch
        # if the directory found
        if os.path.exists(save_dir):
            logging("Load the agent: {}".format(save_dir))
        else:
            raise ValueError

        if self.main_obs_encoder is not None:
            if self.main_obs_encoder.encoder is not None:
                self.main_obs_encoder.encoder.load_state_dict(
                    torch.load(os.path.join(save_dir, f"main_obs_encoder_ep{ep}.pkl")))
                if self._args["if_use_main_target_for_others"]:
                    self.target_obs_encoder.encoder.load_state_dict(
                        torch.load(os.path.join(save_dir, f"target_obs_encoder_ep{ep}.pkl")))

        if self.main_slate_encoder is not None:
            if self.main_slate_encoder.encoder is not None:
                self.main_slate_encoder.encoder.load_state_dict(
                    torch.load(os.path.join(save_dir, f"main_slate_encoder_ep{ep}.pkl")))
                if self._args["if_use_main_target_for_others"]:
                    self.target_slate_encoder.encoder.load_state_dict(
                        torch.load(os.path.join(save_dir, f"target_slate_encoder_ep{ep}.pkl")))

        self.main_action_linear.load_state_dict(torch.load(os.path.join(save_dir, f"main_action_linear_ep{ep}.pkl")))
        if self._args["if_use_main_target_for_others"]:
            self.main_action_linear.load_state_dict(
                torch.load(os.path.join(save_dir, f"target_action_linear_ep{ep}.pkl")))

        self._load(save_dir=save_dir, epoch=epoch)

    def _load(self, save_dir: str, epoch: int = 0):
        """ Child class can just load the weight of the agent """
        raise NotImplementedError

    def save(self, save_dir: str, epoch: int = 0):
        ep = epoch
        # make the directory if it doesn't exist yet
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        logging("Save the agent: {}".format(save_dir))

        if self.main_obs_encoder is not None:
            if self.main_obs_encoder.encoder is not None:
                torch.save(self.main_obs_encoder.encoder.state_dict(),
                           os.path.join(save_dir, f"main_obs_encoder_ep{ep}.pkl"))
                if self._args["if_use_main_target_for_others"]:
                    torch.save(self.target_obs_encoder.encoder.state_dict(),
                               os.path.join(save_dir, f"target_obs_encoder_ep{ep}.pkl"))

        if self.main_slate_encoder is not None:
            if self.main_slate_encoder.encoder is not None:
                torch.save(self.main_slate_encoder.encoder.state_dict(),
                           os.path.join(save_dir, f"main_slate_encoder_ep{ep}.pkl"))
                if self._args["if_use_main_target_for_others"]:
                    torch.save(self.main_slate_encoder.encoder.state_dict(),
                               os.path.join(save_dir, f"target_slate_encoder_ep{ep}.pkl"))

        torch.save(self.main_action_linear.state_dict(), os.path.join(save_dir, f"main_action_linear_ep{ep}.pkl"))
        if self._args["if_use_main_target_for_others"]:
            torch.save(self.main_action_linear.state_dict(), os.path.join(save_dir, f"target_action_linear_ep{ep}.pkl"))

        self._save(save_dir=save_dir, epoch=epoch)

    def _save(self, save_dir: str, epoch: int = 0):
        """ Child class can just save the weight of the agent """
        raise NotImplementedError

    def sync(self, tau: float = 0.0):
        if self._args["if_use_main_target_for_others"]:
            if tau > 0.0:
                for param, target_param in zip(self.main_action_linear.parameters(),
                                               self.target_action_linear.parameters()):
                    # tau * local_param.data + (1.0 - tau) * target_param.data
                    target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

                if self.main_obs_encoder is not None:
                    if self.main_obs_encoder.encoder is not None:
                        for param, target_param in zip(self.main_obs_encoder.encoder.parameters(),
                                                       self.target_obs_encoder.encoder.parameters()):
                            # tau * local_param.data + (1.0 - tau) * target_param.data
                            target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
                if self.main_slate_encoder is not None:
                    if self.main_slate_encoder.encoder is not None:
                        for param, target_param in zip(self.main_slate_encoder.encoder.parameters(),
                                                       self.target_slate_encoder.encoder.parameters()):
                            # tau * local_param.data + (1.0 - tau) * target_param.data
                            target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
            else:
                self.target_action_linear.load_state_dict(self.main_action_linear.state_dict())

                if self.main_obs_encoder is not None:
                    if self.main_obs_encoder.encoder is not None:
                        self.target_obs_encoder.encoder.load_state_dict(self.main_obs_encoder.encoder.state_dict())
                if self.main_slate_encoder is not None:
                    if self.main_slate_encoder.encoder is not None:
                        self.target_slate_encoder.encoder.load_state_dict(self.main_slate_encoder.encoder.state_dict())
        self._sync(tau=tau)

    def _sync(self, tau: float = 0.0):
        raise NotImplementedError

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    def set_baseItems_dict(self, baseItems_dict: dict):
        self._baseItems_dict = baseItems_dict


class RandomAgent(Agent):
    def select_action(self, obs, candidate_list, epsilon):
        """ Produce a slate or a batch of slates

        Args:
            obs: batch_step_size x history_seq
            candidate_list: (num_candidates)-size array
            epsilon: float

        Returns:
            slate: batch_step_size x slate_size
        """
        return self.random_policy.select_action(batch_size=obs.shape[0], candidate_list=candidate_list)


class Test(test):
    def __init__(self):
        self._get_args()
        # self.args.if_debug = True
        self.args.if_debug = False
        # self.args.env_name = "ml-100k"
        self.args.env_name = "recsim"
        self.args.agent_type = "random"
        self.args.if_item_retrieval = True
        self._prep()

    def test(self):
        logging("=== test: RandomAgent ===")
        self.agent = RandomAgent(obs_encoder=None, slate_encoder=None, args=vars(self.args))
        self.test_select_action()

    @staticmethod
    def check_action(action, candidate_list):
        for i, slate in enumerate(action):
            for item in slate:
                assert item in candidate_list, f"{i} {item} {slate} {candidate_list}"

    def test_select_action(self):
        logging("=== test_select_action ===")
        for _ in range(3):
            self.sample_items()
            action = self.agent.select_action(obs=self.obses, candidate_list=self.candidate_list, epsilon=0.5)
            logging(action[0])
            if self.args.if_debug: logging(action)
            assert action.shape == self.action_shape, f"{action.shape} is not matching {self.action_shape}"
            self.check_action(action=action, candidate_list=self.candidate_list)
            self.check_duplication(action)

        if self.args.agent_type not in ["random", "dqn"]:
            self.agent.set_if_visualise(flg=True)
            for _ in range(3):
                self.sample_items()
                action = self.agent.select_action(obs=self.obses, candidate_list=self.candidate_list, epsilon=0.5)
                if self.args.if_debug: logging(action)
                assert action.shape == self.action_shape, f"{action.shape} is not matching {self.action_shape}"
                self.check_action(action=action, candidate_list=self.candidate_list)
                self.check_duplication(action)

    def test_update(self):
        logging("=== test_update ===")
        self.agent.set_if_check_grad(flg=True)
        result = self.agent.update(obses=self.obses,
                                   actions=self.actions,
                                   rewards=self.rewards,
                                   next_obses=self.next_obses,
                                   dones=self.dones,
                                   candidate_lists=self.candidate_lists)
        logging(result)

    def test_save(self):
        logging("=== test_save ===")
        self.agent.save(save_dir="../../result/test/weights")

    def test_load(self):
        logging("=== test_load ===")
        self.agent.load(save_dir="../../result/test/weights")

    def test_visualise(self):
        logging("=== test_visualise ===")
        logging(self.agent.visualise())


if __name__ == '__main__':
    Test().test()
