import numpy as np


class AnnealingSchedule:
    def __init__(self, start=1.0, end=0.1, decay_steps=500):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.annealed_value = np.linspace(start, end, decay_steps)

    def get_value(self, ts):
        return self.annealed_value[min(max(1, ts), self.decay_steps) - 1]  # deal with the edge case


def _test_scheduler():
    from value_based.commons.args import get_all_args
    args = get_all_args()

    scheduler = AnnealingSchedule(start=args.epsilon_start, end=args.epsilon_end, decay_steps=args.decay_steps)
    for i in range(1, args.decay_steps, 100000):
        print(scheduler.get_value(ts=i))


if __name__ == '__main__':
    _test_scheduler()
