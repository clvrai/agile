import numpy as np
import random

from value_based.commons.observation import ObservationFactory
from value_based.commons.test import TestCase


class ReplayBuffer(object):
    """ Experience Replay Memory Buffer which can accommodate the candidate sets """

    def __init__(self, size: int, if_mdp: bool = False):
        self._maxsize = size
        self._if_mdp = if_mdp
        self._storage = []
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done, candidate_list):
        """

        Args:
            obs_t (dict or np.ndarray): np.ndarray if self._if_mdp else dict
            action (np.ndarray):
            reward (np.ndarray):
            obs_tp1 (dict or np.ndarray): np.ndarray if self._if_mdp else dict
            done (bool):
            candidate_list (np.ndarray):
        """
        data = (obs_t, action, reward, obs_tp1, done, candidate_list)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes: list):
        """ One step sampling method

        Args:
            idxes (list):

        Returns:
            obses_t (ObservationFactory): batch of observations
            actions (np.ndarray): batch of actions executed given obs_batch
            rewards (np.ndarray): rewards received as results of executing act_batch
            obses_tp1 (ObservationFactory): next set of observations seen after executing act_batch
            dones (np.ndarray): done_mask[i] = 1 if executing act_batch[i] resulted at the end of episode otherwise 0
        """
        actions, rewards, dones, candidate_lists = list(), list(), list(), list()

        if self._if_mdp:  # for mdp based recsim
            obses_t, obses_tp1 = list(), list()
        else:
            obses_t, obses_tp1 = {"user_id": [], "history_seq": []}, {"user_id": [], "history_seq": []}

        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, candidate_list = data
            if self._if_mdp:  # for mdp based recsim
                obses_t.append(obs_t)
                obses_tp1.append(obs_tp1)
            else:
                obses_t["user_id"].append(obs_t["user_id"])
                obses_t["history_seq"].append(obs_t["history_seq"])
                obses_tp1["user_id"].append(obs_tp1["user_id"])
                obses_tp1["history_seq"].append(obs_tp1["history_seq"])
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            dones.append(done)
            candidate_lists.append(candidate_list)

        if self._if_mdp:  # for mdp based recsim
            _obses_t = ObservationFactory(batch_size=len(idxes), if_mdp=True)
            _obses_t.load_state(state=np.asarray(obses_t))
            _obses_tp1 = ObservationFactory(batch_size=len(idxes), if_mdp=True)
            _obses_tp1.load_state(state=np.asarray(obses_tp1))
            obses_t, obses_tp1 = _obses_t, _obses_tp1  # maintain the compatibility
        else:
            obses_t = ObservationFactory(batch_size=len(idxes), user_id=np.asarray(obses_t["user_id"]))
            obses_tp1 = ObservationFactory(batch_size=len(idxes), user_id=np.asarray(obses_tp1["user_id"]))
        return obses_t, np.array(actions), np.array(rewards), obses_tp1, np.array(dones), \
               np.array(candidate_lists)

    def sample(self, batch_size):
        """ Sample a batch of experiences

        Args:
            batch_size (int): How many transitions to sample

        Returns:
            obses_t (list): batch of observations
            actions (np.ndarray): batch of actions executed given obs_batch
            rewards (np.ndarray): rewards received as results of executing act_batch
            obses_tp1 (list): next set of observations seen after executing act_batch
            dones (np.ndarray): done_mask[i] = 1 if executing act_batch[i] resulted at the end of episode otherwise 0
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def refresh(self):
        self._storage = []
        self._next_idx = 0


class Test(TestCase):
    def __init__(self):
        self._get_args()
        # self.args.if_debug = True
        self.args.if_debug = False
        self._prep()

    def test(self):
        print("=== test ===")

        # instantiate the replay memory
        replay_buffer = ReplayBuffer(size=100)

        for i in range(self.args.batch_size):
            replay_buffer.add(obs_t=self.obses[i],
                              action=self.actions[i, :],
                              reward=self.rewards[i, :],
                              obs_tp1=self.next_obses[i],
                              done=False,
                              candidate_list=self.candidate_lists[i, :])

        obses, actions, rewards, next_obses, dones, candidate_lists = \
            replay_buffer.sample(batch_size=self.args.batch_size)
        assert obses.shape[0] == self.args.batch_size
        assert actions.shape == (self.args.batch_size, self.args.slate_size)
        assert rewards.shape == (self.args.batch_size, 1)
        assert next_obses.shape[0] == self.args.batch_size
        assert dones.shape == (self.args.batch_size,)
        assert candidate_lists.shape == (self.args.batch_size, self.args.num_candidates)


if __name__ == '__main__':
    Test().test()
