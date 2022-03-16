import os
import torch
import numpy as np
from typing import Dict, List

from value_based.policy.agent import Agent
from value_based.policy.gcdqn import SLATE_MASK_TOKEN
from value_based.policy.architecture.Qnet import QNetwork
from value_based.commons.utils import logging
from value_based.encoder.slate_encoder import SlateEncoder
from value_based.encoder.obs_encoder import ObsEncoder
from value_based.encoder.act_encoder import ActEncoder
from value_based.commons.optimiser_factory import OptimiserFactory
from value_based.commons.pt_check_grad import check_grad
from value_based.embedding.base import BaseEmbedding
from value_based.policy.gcdqn import Test as test


class DQN(Agent):
    def __init__(self,
                 obs_encoder: List[ObsEncoder] = None,
                 slate_encoder: List[SlateEncoder] = None,
                 dict_embedding: Dict[str, BaseEmbedding] = None,
                 act_encoder: List[ActEncoder] = None,
                 args: dict = {}):
        super(DQN, self).__init__(obs_encoder=obs_encoder,
                                  slate_encoder=slate_encoder,
                                  dict_embedding=dict_embedding,
                                  act_encoder=act_encoder,
                                  args=args)
        self._optim_factory = OptimiserFactory(if_single_optimiser=True,
                                               if_one_shot_instantiation=self._if_one_shot_instantiation)
        self._optim_factory.add_params(params_dict={"params": self.main_action_linear.parameters(), "lr": self._lr})

        # Init the main Q-net
        self._dim_out = 1 if self._if_sequentialQNet else self._num_trainItems
        self.main_Q_net = QNetwork(dim_in=self._dim_in,
                                   dim_hiddens=self._args.get("q_net_dim_hidden", "256_32"),
                                   dim_out=self._dim_out).to(device=self._device)

        # Init the target Q-net with the params of the main one
        self.target_Q_net = QNetwork(dim_in=self._dim_in,
                                     dim_hiddens=self._args.get("q_net_dim_hidden", "256_32"),
                                     dim_out=self._dim_out).to(device=self._device)
        self.target_Q_net.load_state_dict(self.main_Q_net.state_dict())

        self._optim_factory.add_params(params_dict={"params": self.main_Q_net.parameters(), "lr": self._lr})

        if self.main_obs_encoder is not None:
            if self.main_obs_encoder.encoder is not None:
                self._optim_factory.add_params({"params": self.main_obs_encoder.encoder.parameters(), "lr": self._lr})

        self.optimiser = self._optim_factory.get_optimiser()[0]
        if self._args["if_use_lr_scheduler"]:
            lambda1 = lambda epoch: self._args["lr_scheduler_alpha"] ** epoch
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda=lambda1)
        del self._optim_factory
        if self._if_debug: print("DQN>> {}".format(self.main_Q_net))

    def _select_action(self, inputs, epsilon):
        """ Inner method of select_action

        Args:
            if self._if_sequentialQNet:
                if agent_standardRL_type == None:
                    inputs: (state_embed, item_embed, gnn_embed) -> batch_step_size x num_candidates x dim
                else:
                    inputs: (state_embed, availability_mask)
                        - state_embed: batch_step_size x num_trainItems x dim
                        - availability_mask: batch_step_size x num_trainItems
            else:
                if agent_standardRL_type != None:
                    inputs: (state_embed, item_embed, gnn_embed) -> batch_step_size x dim
                else:
                    inputs: (state_embed, availability_mask)
                        - state_embed: batch_step_size x dim
                        - availability_mask: batch_step_size x num_trainItems

        Returns:
            slate: batch_step_size x slate_size
                - slate of indices in Q-net so that this needs to be converted into itemIds later!
        """
        if self._args["agent_standardRL_type"] != "None":
            inputs, availability_mask = self._prep_input_standardRL(inputs=inputs)
        q_i = self.main_Q_net(inputs)  # batch_step_size x num_candidates
        if self._args["agent_standardRL_type"] != "None":
            q_i[availability_mask] = SLATE_MASK_TOKEN
        slate = torch.topk(q_i, k=self._slate_size).indices  # batch_step_size x slate_size
        slate = slate.cpu().detach().numpy().astype(np.int64)
        return np.asarray(slate)

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
        if self._args["agent_standardRL_type"] != "None":
            inputs, availability_mask = self._prep_input_standardRL(inputs=inputs)
            next_inputs, next_availability_mask = self._prep_input_standardRL(inputs=next_inputs)

        # Compute the q values
        Q_vals = self.main_Q_net(inputs)  # batch_size x num_candidates
        Q_vals = Q_vals.gather(dim=-1, index=actions)
        next_Q_vals = self.target_Q_net(next_inputs)
        if self._args["agent_standardRL_type"] != "None":
            next_Q_vals[next_availability_mask] = SLATE_MASK_TOKEN
        next_Q_vals = torch.topk(next_Q_vals, k=self._slate_size).values  # batch_size x slate_size

        """ Define the loss for k Q-networks: (y - Q^j) 
            ** As in Eq(11), all the k Q-networks are fitting to the same y
        """
        Y = rewards + next_Q_vals * reversed_dones * self._gamma
        loss = torch.mean((Y - Q_vals).pow(2))

        # Optimise the model
        self.optimiser.zero_grad()

        loss.backward()

        # gradient clipping and checking
        torch.nn.utils.clip_grad_norm_(self.main_Q_net.parameters(), self._grad_clip)
        if self._if_check_grad:
            print("=== Q-net ===")
            ave_grad_dict, max_grad_dict = check_grad(named_parameters=self.main_Q_net.named_parameters())
            print("Ave grad: {}\nMax grad: {}".format(ave_grad_dict, max_grad_dict))

        if self.main_obs_encoder is not None:
            if self.main_obs_encoder.encoder is not None:
                torch.nn.utils.clip_grad_norm_(self.main_obs_encoder.encoder.parameters(), self._grad_clip)
                if self._if_check_grad:
                    print("=== Obs Encoders ===")
                    ave_grad_dict, max_grad_dict = check_grad(
                        named_parameters=self.main_obs_encoder.encoder.named_parameters())
                    print("Ave grad: {}\nMax grad: {}".format(ave_grad_dict, max_grad_dict))

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.main_Q_net.parameters(), self._grad_clip)

        # apply processed gradients to the network
        self.optimiser.step()
        if self._args["if_use_lr_scheduler"]:
            self.lr_scheduler.step(epoch=self.epoch)
        return {"loss": loss.item()}

    def _save(self, save_dir: str, epoch: int = 0):
        ep = epoch
        logging("Save the agent: {}".format(save_dir))
        torch.save(self.main_Q_net.state_dict(), os.path.join(save_dir, f"main_ep{ep}.pkl"))
        torch.save(self.target_Q_net.state_dict(), os.path.join(save_dir, f"target_ep{ep}.pkl"))

    def _load(self, save_dir: str, epoch: int = 0):
        ep = epoch
        logging("Load the agent: {}".format(save_dir))
        self.main_Q_net.load_state_dict(torch.load(os.path.join(save_dir, f"main_ep{ep}.pkl")))
        self.target_Q_net.load_state_dict(torch.load(os.path.join(save_dir, f"target_ep{ep}.pkl")))

    def _sync(self, tau: float = 0.0):
        if tau > 0.0:  # Soft update of params
            for param, target_param in zip(self.main_Q_net.parameters(), self.target_Q_net.parameters()):
                # tau * local_param.data + (1.0 - tau) * target_param.data
                target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)
        else:
            self.target_Q_net.load_state_dict(self.main_Q_net.state_dict())

    def visualise(self):
        res = dict()
        _res = self._get_summary_of_params(model=self.target_Q_net, model_name="target")
        for k, v in _res.items(): res[k] = v

        _res = self._get_summary_of_params(model=self.main_Q_net, model_name="main")
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
        # self.args.if_debug = True
        self.args.if_debug = False
        self.args.env_name = "recsim"
        self.args.agent_type = "dqn"
        self._prep()

    def test(self):
        from value_based.commons.launcher import launch_agent, launch_encoders
        from value_based.commons.args import add_args
        from value_based.commons.args_agents import params_dict, check_param

        # main test part
        for agent_name in [
            "DQN",
            # "RDQN",
            # "STANDARD-DQN",
        ]:
            for _params_dict in params_dict[agent_name]:
                print(f"\n=== params {_params_dict} ===")
                if not check_param(_params_dict=_params_dict): continue
                # Update the hyper-params with the test specific ones
                self.args = self.update_args_from_dict(args=self.args, _dict=_params_dict)
                self.args = add_args(args=self.args)
                encoders_dict = launch_encoders(item_embedding=self.dict_embedding["item"], args=self.args)
                self.agent = launch_agent(dict_embedding=self.dict_embedding,
                                          encoders_dict=encoders_dict,
                                          args=self.args)
                self.test_select_action()
                self.test_update()
                # self.test_save()
                # self.test_visualise()


if __name__ == '__main__':
    Test().test()
