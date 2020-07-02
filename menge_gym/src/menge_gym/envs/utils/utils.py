from copy import deepcopy


class DeviationWindow(object):
    def __init__(self, size):
        self.size = size
        self.index = 0
        self.content = size * [0]
        self.deviations = size * [0]

    def __str__(self):
        return str(self.content)

    def __call__(self, item):
        idx = self.index
        self.content[idx] = item
        self.deviations[idx] = item - self.content[(idx - 1) % self.size]
        self.index = (idx + 1) % self.size

    def __setitem__(self, key, value):
        self.content[key] = value

    def __getitem__(self, item):
        return self.content[item]

    def __abs__(self):
        return list(map(abs, self.deviations))

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.content)

    def copy(self):
        return deepcopy(self)

    def mean(self):
        return 1 / self.size * sum(self.deviations)

    def mean_of_abs(self):
        return 1 / self.size * sum((map(abs, self.deviations)))
