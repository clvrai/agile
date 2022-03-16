import torch
import numpy as np
from typing import Dict

from value_based.embedding.base import BaseEmbedding
from value_based.commons.args import HISTORY_SIZE, SKIP_TOKEN, EMPTY_USER_ID, EMPTY_ITEM_ID
from value_based.commons.test import TestCase


class ObservationFactory(object):
    """ Manipulates the obs """

    def __init__(self, batch_size: int, user_id: np.ndarray = None, if_mdp: bool = False):
        self._batch_size = batch_size
        self._user_id = user_id if user_id is not None else np.ones(self._batch_size) * EMPTY_USER_ID
        # FIFO queue; batch_step_size x history_size
        self._history_seq = np.ones((self._batch_size, HISTORY_SIZE)) * EMPTY_ITEM_ID

        # for MDP setting in RecSim
        self._if_mdp = if_mdp
        self._state = None

    @property
    def shape(self):
        if self._if_mdp:
            return self._state.shape
        else:
            return self._history_seq.shape

    def update_history_seq(self, a_user: np.ndarray, active_user_mask: np.ndarray = None):
        """ Updates the user history sequence

        Args:
            a_user (np.ndarray): (batch_size)-size array
        """
        # Update happens only on the clicked items
        _mask = (a_user != SKIP_TOKEN)

        # FIFO update; batch_step_size x history_size
        self._history_seq[_mask] = np.concatenate([self._history_seq[_mask, 1:], a_user[_mask, None]], axis=1)
        if active_user_mask is not None:
            self._history_seq = self._history_seq[active_user_mask]  # Remove the obs of inactive users

    def load_history_seq(self, history_seq: np.ndarray):
        """ Load the user history sequence

        Args:
            history_seq (np.ndarray): batch_size x history_size
        """
        self._history_seq = history_seq

    def make_obs(self, dict_embedding: Dict[str, BaseEmbedding], if_separate: bool = False, device: str = "cpu"):
        """ Make an observation by concatenating the user attributes and its sequence of clicked items' embedding

        Args:
            dict_embedding (dict):
            if_separate (bool): binary flag changing the format of output either np.ndarray or dict
            device (str): cpu or cuda

        Returns:
            self._if_mdp:
                state (np.ndarray): batch_size x dim_user
            else:
                if_separate:
                    _dict (dict):
                        user_feat (np.ndarray): batch_size x history_size x dim_user
                        history_seq (np.ndarray): batch_size x history_size x dim_item
                else:
                    obs (np.ndarray): batch_step_size x history_size x (dim_user + dim_item)
        """
        if not self._if_mdp:
            history_seq = torch.tensor(self._history_seq, dtype=torch.int64, device=device)
            user_history_feat = dict_embedding["item"].get(index=history_seq, if_np=True)

            if all(self._user_id != EMPTY_USER_ID):
                user_id = torch.tensor(self._user_id, dtype=torch.int64, device=device)
                user_feat = dict_embedding["user"].get(index=user_id, if_np=True)[:, None, :]
                user_feat = np.tile(A=user_feat, reps=(1, self._history_seq.shape[1], 1))
                if if_separate:
                    return {"user_feat": user_feat, "history_seq": user_history_feat}
                else:
                    return np.concatenate([user_feat, user_history_feat], axis=-1)
            return user_history_feat
        else:
            return self._state

    def __getitem__(self, item):
        """ This is for ReplayBuffer """
        if self._if_mdp:
            return self._state[item, :]
        else:
            return {"user_id": self._user_id[item], "history_seq": self._history_seq[item, :]}

    @property
    def data(self):
        """ This is for ReplayBuffer """
        if self._if_mdp:
            return self._state
        else:
            return {"user_id": self._user_id, "history_seq": self._history_seq}

    def load_state(self, state: np.ndarray):
        self._if_mdp = True
        self._state = state

    def create_empty_obs(self):
        return {"user_id": 0, "history_seq": np.zeros_like(self._history_seq[0])}


class Test(TestCase):
    def __init__(self):
        self._get_args()
        # self.args.if_debug = True
        self.args.if_debug = False
        self.args.device = "cuda"
        self._prep()

    def test(self):
        print("=== test ===")

        # Case: we know the userIds; MovieLens etc
        obs = ObservationFactory(batch_size=self.args.batch_size, user_id=self.user_id)
        print(obs.make_obs(dict_embedding=self.dict_embedding, if_separate=False, device=self.args.device))
        asdf
        for i in range(HISTORY_SIZE):
            a_user = self.history_seq[:, i]
            obs.update_history_seq(a_user=a_user)
        obs.load_history_seq(history_seq=self.history_seq)
        print(obs.make_obs(dict_embedding=self.dict_embedding, if_separate=False, device=self.args.device).shape)

        # Case: we don't know the userIds: RecSim etc
        obs = ObservationFactory(batch_size=self.args.batch_size)
        for i in range(HISTORY_SIZE):
            a_user = self.history_seq[:, i]
            obs.update_history_seq(a_user=a_user)
        obs.load_history_seq(history_seq=self.history_seq)
        print(obs.make_obs(dict_embedding=self.dict_embedding, if_separate=False, device=self.args.device).shape)


if __name__ == '__main__':
    Test().test()
