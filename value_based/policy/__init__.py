import numpy as np
from random import sample


class RandomPolicy(object):
    def __init__(self, slate_size):
        self._slate_size = slate_size

    def select_action(self, batch_size, candidate_list):
        """ Randomly provides the batch of slates

        Args:
            batch_size: int
            candidate_list: (num_candidates)size array

        Returns:
            batch_size x slate_size
        """
        if type(candidate_list) != list:
            candidate_list = candidate_list.tolist()

        """ Note
            - For random library, the random seed is set in set_randomSeed!
            - This was the fastest way to work out the slate! Faster than using numpy APIs
        """
        return np.asarray([sample(candidate_list, k=self._slate_size) for _ in range(batch_size)]).astype(np.int64)


class Test(object):
    def test(self):
        slate_size = 3
        batch_size = 4
        num_candidates = 10

        self.agent = RandomPolicy(slate_size=slate_size)
        print("=== test_select_action ===")
        a = self.agent.select_action(batch_size=batch_size, candidate_list=list(range(num_candidates)))
        print(a)


if __name__ == '__main__':
    from value_based.commons.seeds import set_randomSeed

    set_randomSeed(seed=1)
    Test().test()
    set_randomSeed(seed=1)
    Test().test()
