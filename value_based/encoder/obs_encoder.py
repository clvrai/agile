import torch
import numpy as np

from gnn.DeepSet.models import DeepSet, BiLSTM
from value_based.encoder.sequential_models import RNNFamily
from value_based.commons.test import TestCase as test


class ObsEncoder(object):
    def __init__(self, args: dict):
        self._args = args
        self._device = self._args.get("device", "cpu")
        self._dim_in = self._args.get("obs_encoder_dim_in", 32)
        self._dim_hidden = self._args.get("obs_encoder_dim_hidden", 32)
        self._mlp_dim_hidden = self._args.get("obs_encoder_mlp_dim_hidden", "32_32")
        self._dim_out = self._args.get("obs_encoder_dim_out", 32)

    def encode(self, obs):
        """ Encode an obs into a latent state

        Args:
            obs(np.ndarray): batch_size x history_size x dim_obs

        Returns:
            s_t(torch.tensor): batch_size x dim_state
        """
        state_embed = self._encode(obs=obs)
        if torch.is_tensor(state_embed):
            return state_embed
        else:
            return torch.tensor(state_embed.astype(np.float32), device=self._device)

    def _encode(self, obs):
        raise NotImplementedError

    @property
    def encoder(self):
        return None


class BasicObsEncoder(ObsEncoder):
    def __init__(self, args: dict):
        super(BasicObsEncoder, self).__init__(args=args)

    def _encode(self, obs):
        """ Average over the history_sequence

        Args:
            obs(np.ndarray): batch_size x history_size x dim_obs

        Returns:
            s_t(torch.tensor): batch_size x dim_state
        """
        state_embed = obs.mean(axis=1).mean(axis=-1)[:, None]  # batch_size x 1
        return np.tile(A=state_embed, reps=(1, self._dim_out))


class SequentialObsEncoder(ObsEncoder):
    def __init__(self, args: dict):
        super(SequentialObsEncoder, self).__init__(args=args)
        if args["obs_encoder_type"] == "sequential-rnn":
            self._encoder = RNNFamily(dim_in=self._dim_in,
                                      dim_hidden=self._dim_hidden,
                                      mlp_dim_hidden=self._mlp_dim_hidden,
                                      dim_out=self._dim_out,
                                      batch_first=True,
                                      dropout_rate=0.0,
                                      args=args,
                                      device=self._device).to(self._device)
        elif args["obs_encoder_type"] == "sequential-lstm":
            self._encoder = BiLSTM(dim_in=self._dim_in,
                                   dim_hidden=self._dim_hidden,
                                   mlp_dim_hidden=self._mlp_dim_hidden,
                                   dim_out=self._dim_out,
                                   batch_first=True,
                                   dropout_rate=0.0,
                                   args=args,
                                   device=self._device).to(self._device)
        elif args["obs_encoder_type"] == "sequential-deep_set":
            self._encoder = DeepSet(dim_in=self._dim_in,
                                    dim_hidden=self._dim_hidden,
                                    mlp_dim_hidden=self._mlp_dim_hidden,
                                    dim_out=self._dim_out,
                                    batch_first=True,
                                    dropout_rate=0.0,
                                    args=args,
                                    device=self._device).to(self._device)
        else:
            raise ValueError
        if self._args["if_debug"]: print(">> {}".format(self._encoder))

    def _encode(self, obs):
        """ Inner method of encoding

        Args:
            obs(np.ndarray): batch_size x history_size x dim_obs

        Returns:
            s_t(torch.tensor): batch_size x dim_state
        """
        return self._encoder(torch.tensor(obs.astype(np.float32), device=self._device))

    @property
    def encoder(self):
        return self._encoder


class Test(test):
    def __init__(self):
        self._get_args()
        # self.args.if_debug = True
        self.args.if_debug = False
        # self.args.env_name = "ml-100k"
        self.args.env_name = "recsim"
        from value_based.commons.args import add_args
        self.args = add_args(args=self.args)
        self._prep()

    def test(self):
        print("=== _test ===")
        for obs_encoder in [
            BasicObsEncoder(args=vars(self.args)),
            SequentialObsEncoder(args=vars(self.args))
        ]:
            print(obs_encoder)
            obses = self.obses.make_obs(dict_embedding=self.dict_embedding, device=self.args.device)
            state_embed = obs_encoder.encode(obs=obses)
            assert state_embed.shape == (self.args.batch_size, self.args.obs_encoder_dim_out)
            assert torch.all(~torch.isnan(state_embed))


if __name__ == '__main__':
    Test().test()
