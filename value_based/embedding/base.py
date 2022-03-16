import torch
import pickle
import numpy as np
from scipy import stats
from random import sample

from value_based.commons.utils import logging
from value_based.commons.test import TestCase as test
from value_based.commons.args import EMPTY_ITEM_ID


class BaseEmbedding(object):
    def __init__(self,
                 num_embeddings: int = 100,
                 dim_embed: int = 32,
                 embed_type: str = "random",
                 embed_path: str = None,
                 device: str = "cpu",
                 if_debug: bool = False):
        self._path = embed_path
        self._embed_type = embed_type
        self._num_embeddings = num_embeddings
        self._dim_embed = dim_embed
        self._device = device
        self._if_debug = if_debug

        if self._embed_type == "random":
            _embedding = np.random.randn(self._num_embeddings, self._dim_embed)
        elif self._embed_type == "one_hot":
            _embedding = np.eye(self._num_embeddings)
        elif self._embed_type == "pretrained":
            if self._path.endswith(".npy"):
                _embedding = np.load(self._path)
                logging("BaseEmbedding>> Load -> Shape: {}".format(_embedding.shape))
            else:
                with open(self._path, "rb") as file:
                    _embedding = pickle.load(file)
        else:
            raise ValueError

        if self._if_debug:
            logging("BaseEmbedding>> Shape: {}".format(_embedding.shape))
        self._embedding = torch.nn.Embedding.from_pretrained(
            embeddings=torch.tensor(_embedding, dtype=torch.float32, device=self._device))

    @property
    def embedding(self):
        return self._embedding

    def get(self, index, if_np: bool = False):
        """ Return an embedding based on the given index

        Args:
            index (): index of an embedding(embeddings)
            if_np (bool): whether we return in np.ndarray or torch.tensor

        Returns:
            _embedding (np.ndarray or torch.tensor): index_size x dim_embed
        """
        if not torch.is_tensor(index):
            index = torch.tensor(index, dtype=torch.int64, device=self._device)

        # Check if there is any empty id in the given index and replace it with temp id
        mask = index == EMPTY_ITEM_ID
        index[mask] = 0

        # Get the corresponding embedding
        _embedding = self._embedding(index)

        # # Replace the temp embedding with zero embedding
        _embedding[mask] = 0

        # Cast the data type
        if if_np:
            return _embedding.cpu().detach().numpy()
        else:
            return _embedding

    def __getitem__(self, index):
        """ Supports the list-style of accessing items
            x.__getitem__(index) <==> x[index]
        """
        return self.get(index=index)

    def get_all(self, if_np: bool = False):
        embed = self._embedding.weight
        if if_np:
            return embed.detach().cpu().numpy()
        else:
            return embed

    def get_batch(self, index, batch_size: int = 32, if_np: bool = False):
        """ Returns a batch of item-embeddings

        Args:
            index: (num_item)-size list
            batch_size: int
            if_np: if return in np array

        Returns:
            _embedding: batch_size x num_items x dim_item
        """
        _embedding = self.get(index=index)
        _embedding = _embedding[None, :].repeat(batch_size, 1, 1)  # batch_size x num_items x dim_item
        if if_np:
            return _embedding.cpu().detach().numpy()
        else:
            return _embedding

    def get_stats_embedding(self):
        # check the variance of each dim(col)
        return stats.describe(self._embedding, axis=0)._asdict()

    def update(self):
        pass

    def load(self, embedding):
        if isinstance(embedding, BaseEmbedding):
            embedding = embedding.get_all(if_np=True)
        if not torch.is_tensor(embedding):
            embedding = torch.tensor(embedding, dtype=torch.float32, device=self._device)
        self._num_embeddings, self._dim_embed = embedding.shape
        self._embedding = torch.nn.Embedding.from_pretrained(embeddings=embedding)
        if self._if_debug:
            logging("BaseEmbedding>> Load: {}".format(embedding.shape))

    def save(self):
        pass

    @property
    def shape(self):
        return self._embedding.num_embeddings, self._embedding.embedding_dim

    def sample(self, num_samples: int, if_np: bool = False):
        _ids = sample(range(self._embedding.num_embeddings), num_samples)
        return self.get(index=_ids, if_np=if_np)


class Test(test):
    def __init__(self):
        self._get_args()
        # self.args.if_debug = True
        self.args.if_debug = False
        self._prep()

    def test(self):
        logging("=== test ===")
        # 1D array case: (3,)
        index1 = [1, 2, 3]

        # 2D matrix case: 3 x 3
        index2 = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]

        # 3D tensor case; 3 x 3 x 3
        index3 = [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ],
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ],
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ],
        ]

        for embed_type in [
            "random",
            "one_hot",
            # "pt"
        ]:
            logging("=== {} ===".format(embed_type))
            self.args.embed_type = embed_type
            item_embedding = BaseEmbedding(num_embeddings=self.args.num_allItems,
                                           dim_embed=self.args.dim_item,
                                           embed_type=self.args.item_embedding_type,
                                           embed_path=self.args.item_embedding_path,
                                           device=self.args.device)
            user_embedding = BaseEmbedding(num_embeddings=self.args.num_allUsers,
                                           dim_embed=self.args.dim_user,
                                           embed_type=self.args.user_embedding_type,
                                           embed_path=self.args.user_embedding_path,
                                           device=self.args.device)
            logging("item_embedding: {}".format(item_embedding.shape))
            logging("item_embedding: {}".format(item_embedding.get_all().shape))

            for index in [index1, index2, index3]:
                logging("index: {}".format(np.asarray(index).shape))
                embedding = item_embedding.get(index=index, if_np=True)
                logging("extracted embedding: {}".format(embedding.shape))


if __name__ == '__main__':
    Test().test()
