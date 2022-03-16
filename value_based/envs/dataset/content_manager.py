import numpy as np
import pandas as pd


class ContentManager(object):
    """ this manages the item embedding and item category of items in the dataset """

    def __init__(self, items_dict: dict, df_master: pd.DataFrame, args: dict):
        self._args = args
        self._items_dict = items_dict
        self._category_cols = self._args.get("data_category_cols", "None/None").split("/")  # Name: mainCat_subCat
        self._df_master = df_master
        self._prep()
        self._category_mat = np.eye(self.num_categories)

    def _prep(self):
        self.categories, self.category_master_dict, self.main_category_master_dict = dict(), dict(), dict()
        for flg, list_itemIds in self._items_dict.items():
            _item_info = self._df_master[self._category_cols].values[list_itemIds]
            self.categories[flg] = {k: f"{v[0]}_{v[1]}" for k, v in zip(list_itemIds, _item_info)}

            _category_master_dict = {f"{v[0]}_{v[1]}": list() for v in _item_info}
            _main_category_master_dict = {v[0]: list() for v in _item_info}
            for k, v in zip(list_itemIds, _item_info):
                _category_master_dict[f"{v[0]}_{v[1]}"].append(k)
                _main_category_master_dict[v[0]].append(k)
            self.category_master_dict[flg] = _category_master_dict
            self.main_category_master_dict[flg] = _main_category_master_dict

    def get_mainCategory(self, itemId: int) -> int:
        return self._df_master[self._category_cols[0]][itemId]

    def get_subCategory(self, itemId: int) -> int:
        return self._df_master[self._category_cols[1]][itemId]

    def get_fullCategory(self, itemId: int) -> int or list:
        out = self._df_master[self._category_cols].values[itemId]
        return f"{out[0]}_{out[1]}"

    def get_slate_category(self, slate):
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

    def get_fullCategory_vec(self, itemId: int):
        """ Returns a category ids of an item.
            Returns: int or list
                int: categoryId
                list: [main_categoryId, sub_categoryId]
        """
        if self._args.get("recsim_if_use_subcategory", False):
            # list: [main_categoryId, sub_categoryId]
            return self.get_fullCategory(itemId=itemId)
        else:
            # int: categoryId
            return self.get_mainCategory(itemId=itemId)

    @property
    def num_categories(self):
        return len(self._df_master[self._category_cols[0]].unique())

    def set_items_dict(self, items_dict: dict):
        self._items_dict = items_dict
        self._prep()


def test():
    print("=== test ===")
    from random import shuffle

    args = {"data_category_cols": "genre_size"}

    num_allItems = 50

    all_items = list(range(num_allItems))
    shuffle(all_items)
    items_dict = {"train": all_items[:25], "test": all_items[25:]}

    df_master = pd.read_csv("../../../data/sample/item_category.csv")
    cc = ContentManager(items_dict=items_dict, df_master=df_master, args=args)
    res = cc.get_mainCategory(itemId=1)
    print(res)
    res = cc.get_subCategory(itemId=1)
    print(res)
    res = cc.get_fullCategory(itemId=1)
    print(res)


if __name__ == '__main__':
    test()
