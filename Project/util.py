class IncrementalAverage:
    """
    This class enables us to store the running average
    of some datapoints on an incremental way, without
    the need of saving all the values.
    """
    def __init__(self):
        self.counter = 1
        self.sum  = 0

    def __str__(self):
        return str(self.sum)

    def __repr__(self):
        return "[counter: {0}, sum: {1}]".format(self.counter - 1, self.sum)

    def add(self, x):
        """
        Add a new data point.
        :param x: New data point to be added to the average.
        """
        self.sum += (1 / self.counter) * (x - self.sum)
        self.counter += 1

    def get(self):
        """
        :return: The average.
        """
        return self.sum

    def getNumberOfItems(self):
        """
        :return: The number of items averaged over.
        """
        return self.counter - 1

    def reset(self):
        """
        Clears the cache.
        """
        self.counter = 1
        self.sum = 0
