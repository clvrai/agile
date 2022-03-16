# coding=utf-8
# coding=utf-8
# Copyright 2019 The RecSim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Classes to represent and interface with documents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import six

from value_based.commons.args import ENV_RANDOM_SEED


# Some notes:
#
#   These represent properties of the documents. Presumably we can add more or
#   remove documents from the candidate set. But these do not include features
#   that depend on both the user and the document.
#


class CandidateSet(object):
    """Class to represent a collection of AbstractDocuments.

       The candidate set is represented as a hashmap (dictionary), with documents
       indexed by their document ID.
    """

    def __init__(self, args: dict):
        """Initializes a document candidate set with 0 documents."""
        self._args = args
        self._documents = {}
        self._special_documents = {}
        self._category_mat = np.eye(self._args.get("recsim_num_categories", 10))

    def size(self):
        """Returns an integer, the number of documents in this candidate set."""
        return len(self._documents)

    def get_all_documents(self):
        """Returns all documents."""
        return self.get_documents(self._documents.keys())

    def get_documents(self, document_ids):
        """Gets the documents associated with the specified document IDs.

        Args:
          document_ids: an array representing indices into the candidate set.
            Indices can be integers or string-encoded integers.

        Returns:
          (documents) an ordered list of AbstractDocuments associated with the
            document ids.
        """
        return [self._documents[int(k)] for k in document_ids]

    def add_document(self, document, if_special: bool = False):
        """Adds a document to the candidate set."""
        if if_special:
            self._special_documents[document.doc_id()] = document
        else:
            self._documents[document.doc_id()] = document

    def remove_document(self, document, if_special: bool = False):
        """Removes a document from the set (to simulate a changing corpus)."""
        if if_special:
            del self._special_documents[document.doc_id()]
        else:
            del self._documents[document.doc_id()]

    def create_observation(self):
        """Returns a dictionary of observable features of documents."""
        # TODO: we don't need this API for this project but need to think about how to use special-docs here...
        return {str(k): self._documents[k].create_observation() for k in self._documents.keys()}

    def get_categories(self):
        """Returns a dictionary of category ids of documents."""
        return {str(k): self._documents[k].category_id for k in self._documents.keys()}

    def get_fullCategory(self, itemId: int):
        """Returns a category ids of an item."""
        return self._documents[itemId].category_id

    def get_fullCategory_vec(self, itemId: int):
        """ Returns a category ids of an item.
            Returns: int or list
                int: categoryId
                list: [main_categoryId, sub_categoryId]
        """
        if self._args.get("recsim_if_use_subcategory", False):
            # list: [main_categoryId, sub_categoryId]
            return [int(i) for i in self._documents[itemId].category_id.split("_")]
        else:
            # int: categoryId
            return self.get_fullCategory(itemId=itemId)

    def get_mainCategory(self, itemId: int):
        """ Returns a category ids of an item.
            Returns: int or list
                int: categoryId
                list: [main_categoryId, sub_categoryId]
        """
        if self._args.get("recsim_if_use_subcategory", False):
            # list: [main_categoryId, sub_categoryId]
            return int(self._documents[itemId].category_id.split("_")[0])
        else:
            # int: categoryId
            return self.get_fullCategory(itemId=itemId)

    def get_slate_category(self, slate: list):
        """ Returns the dict of categories of items

        Returns:
            category_mat (np.ndarray): num_categories x num_subcategories or num_categories x 1
        """
        # Get the categories of items in slate
        if self._args.get("recsim_if_use_subcategory", False):
            _subcategories = [self.get_fullCategory(i) for i in slate]
            # num_categories x num_subcategories
            category_mat = np.zeros((self._args.get("recsim_num_categories", 10),
                                     self._args.get("recsim_num_subcategories", 3)))
            for _category_key in _subcategories:
                _main, _sub = _category_key.split("_")
                category_mat[int(_main), int(_sub)] += 1.0
        else:
            category_mat = np.asarray([self._category_mat[self.get_mainCategory(i)] for i in slate])
            category_mat = np.sum(category_mat, axis=0)[:, None]  # num_categories x 1
        return category_mat

    # def observation_space(self):
    #     return spaces.Dict({str(k): self._documents[k].observation_space() for k in self._documents.keys()})


@six.add_metaclass(abc.ABCMeta)
class AbstractDocumentSampler(object):
    """Abstract class to sample documents."""

    def __init__(self, doc_ctor, **kwargs):
        self._doc_ctor = doc_ctor
        self.reset_sampler()

    @property
    def random_seed(self):
        return ENV_RANDOM_SEED

    def reset_sampler(self):
        self._rng = np.random.RandomState(ENV_RANDOM_SEED)

    @abc.abstractmethod
    def sample_document(self, doc_id: int, flg_trainTest: str):
        """Samples and return an instantiation of AbstractDocument."""

    def get_doc_ctor(self):
        """Returns the constructor/class of the documents that will be sampled."""
        return self._doc_ctor

    @property
    def num_categories(self):
        """Returns the number of document clusters. Returns 0 if not applicable."""
        return 0

    def update_state(self, documents, responses):
        """Update document state (if needed) given user's (or users') responses."""
        pass


@six.add_metaclass(abc.ABCMeta)
class AbstractDocument(object):
    """Abstract class to represent a document and its properties."""

    def __init__(self, doc_id):
        self._doc_id = doc_id  # Unique identifier for the document

    def doc_id(self):
        """Returns the document ID."""
        return self._doc_id

    def set_doc_id(self, _id: int):
        """Updates the document ID."""
        self._doc_id = _id

    def category_id(self):
        return self._category_id

    @abc.abstractmethod
    def create_observation(self):
        """Returns observable properties of this document as a float array."""

    # @classmethod
    # @abc.abstractmethod
    # def observation_space(cls):
    #     """Gym space that defines how documents are represented."""

    @abc.abstractmethod
    def create_state(self):
        """Returns all the properties of this document as a float array."""
