import torch
import numpy as np

from value_based.embedding.base import BaseEmbedding
from value_based.encoder.sequential_models import RNNFamily
from gnn.DeepSet.models import DeepSet, BiLSTM
from value_based.commons.test import TestCase as test


class SlateEncoder(object):
    def __init__(self, item_embedding: BaseEmbedding, args: dict):
        self._args = args
        self._device = self._args.get("device", "cpu")
        self._dim_in = self._args["slate_encoder_dim_in"]
        self._dim_hidden = self._args["slate_encoder_dim_hidden"]
        self._mlp_dim_hidden = self._args["slate_encoder_mlp_dim_hidden"]
        self._dim_out = self._args["slate_encoder_dim_out"]
        self.item_embedding = item_embedding

    def encode(self, slate, item_embed, batch_size: int):
        """ Encode a slate and get an action representation

        Args:
            slate: batch_size x slate_size
            item_embed: num_candidate x dim_hidden_item

        Returns:
            a_t(torch.tensor): batch_size x dim_item
        """
        if torch.is_tensor(slate):
            slate = slate.cpu().detach().numpy().astype(np.int64)

        if (slate is None) or (slate.size == 0):
            # At the beginning, we will fill with Zeroes
            a_t = np.zeros((batch_size, self._dim_out))
        else:
            slate = np.tile(A=slate[..., None], reps=(1, 1, item_embed.shape[-1]))
            item_embed = item_embed.gather(dim=1, index=torch.tensor(slate, device=self._device))
            a_t = self._encode(item_embed=item_embed, batch_size=batch_size)
        if not torch.is_tensor(a_t):
            a_t = torch.tensor(a_t, dtype=torch.float32, device=self._device)
        return a_t

    def _encode(self, item_embed: np.ndarray, batch_size: int):
        """ Encode a slate and get an action representation

        Args:
            slate(np.ndarray): batch_size x slate_size

        Returns:
            a_t(torch.tensor): batch_size x dim_item
        """
        raise NotImplementedError

    def update_slate(self, _pos, slate, item):
        """ Inserting a new item at the last position in a given slate

        Args:
            _pos: an int representing a position of a slate
            slate: batch_size x num_items_so_far_in_a_slate
            item: batch_size x 1

        Returns:
            slate: updated slate
        """
        if _pos == 0:
            slate = item  # batch_size x 1
        else:
            slate = np.hstack([slate, item])  # batch_size x num_items_so_far_in_a_slate
        return slate

    @property
    def encoder(self):
        return None


class BasicSlateEncoder(SlateEncoder):
    def _encode(self, slate: np.ndarray, batch_size: int):
        """ Encode a slate and get an action representation

        Args:
            slate(np.ndarray): batch_size x slate_size

        Returns:
            a_t(torch.tensor): batch_size x dim_item
        """
        _embed = self.item_embedding.get(index=slate)
        _a = torch.cat([_embed.mean(axis=1), _embed.std(axis=1)], dim=-1)  # Original CDQN does this!
        _a[torch.isnan(_a)] = 0.0  # replace nan with 0
        return _a


class SequentialSlateEncoder(SlateEncoder):
    def __init__(self, item_embedding, args):
        super(SequentialSlateEncoder, self).__init__(item_embedding=item_embedding, args=args)
        if args["slate_encoder_type"] == "sequential-rnn":
            self._encoder = RNNFamily(dim_in=self._args["slate_encoder_dim_in"],
                                      dim_hidden=self._args["slate_encoder_dim_hidden"],
                                      mlp_dim_hidden=self._args["slate_encoder_mlp_dim_hidden"],
                                      dim_out=self._args["slate_encoder_dim_out"],
                                      batch_first=True,
                                      args=args,
                                      device=self._device).to(self._device)
        elif args["slate_encoder_type"] == "sequential-lstm":
            self._encoder = BiLSTM(dim_in=self._args["slate_encoder_dim_in"],
                                   dim_hidden=self._args["slate_encoder_dim_hidden"],
                                   mlp_dim_hidden=self._args["slate_encoder_mlp_dim_hidden"],
                                   dim_out=self._args["slate_encoder_dim_out"],
                                   batch_first=True,
                                   args=args,
                                   device=self._device).to(self._device)
        elif args["slate_encoder_type"] == "sequential-deep_set":
            self._encoder = DeepSet(dim_in=self._args["slate_encoder_dim_in"],
                                    dim_hidden=self._args["slate_encoder_dim_hidden"],
                                    mlp_dim_hidden=self._args["slate_encoder_mlp_dim_hidden"],
                                    dim_out=self._args["slate_encoder_dim_out"],
                                    batch_first=True,
                                    args=args,
                                    device=self._device).to(self._device)
        else:
            raise ValueError

        if self._args["if_debug"]: print(">> {}".format(self._encoder))

    def _encode(self, item_embed: torch.tensor, batch_size: int):
        """ Encode a slate and get an action representation

        Args:
            item_embed(torch.tensor): batch_size x num_items x dim_item

        Returns:
            a_t(torch.tensor): batch_size x dim_item
        """
        return self._encoder(item_embed)

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
        self.args.agent_type = "cdqn"
        self._prep()

    def test(self):
        print("=== _test ===")
        for slate_encoder in [
            BasicSlateEncoder(item_embedding=self.item_embedding, args=vars(self.args)),
            SequentialSlateEncoder(item_embedding=self.item_embedding, args=vars(self.args))
        ]:
            print(slate_encoder.__class__.__name__)
            slate_soFar = None
            for i in range(self.args.slate_size):
                a_t = slate_encoder.encode(slate=slate_soFar, batch_size=self.args.batch_size)
                slate_soFar = slate_encoder.update_slate(_pos=i, slate=slate_soFar, item=self.actions[:, i][:, None])
                assert torch.is_tensor(a_t)
                assert a_t.shape == (self.args.batch_size, self.args.slate_encoder_dim_out), \
                    "{} is expected but {} is received".format(
                        (self.args.batch_size, self.args.slate_encoder_dim_out), a_t.shape
                    )
                assert torch.all(~torch.isnan(a_t))


if __name__ == '__main__':
    Test().test()
