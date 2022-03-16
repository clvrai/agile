import torch
import numpy as np
import random


def set_randomSeed(seed: int = 2021):
    # [WARNING] Performance Degradation: https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def test():
    print("=== test ===")
    set_randomSeed()
    print(np.random.randn(1))
    print(torch.randn(1, device="cpu"))
    print(torch.randn(1, device="cuda"))


if __name__ == '__main__':
    for _ in range(3):
        test()
