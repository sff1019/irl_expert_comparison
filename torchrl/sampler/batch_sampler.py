from torchrl.sampler.base import BaseSampler


class BatchSampler(BaseSampler):
    def __init__(self, algo):
        self.algo = algo
