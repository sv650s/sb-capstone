
# base class
class Sampler:

    def __init__(self, has_header:bool = True):
        self.has_header = has_header

    def collect(self, index:int) -> True:
        """
        not yet implemented
        :param index:
        :return:
        """
        assert False, "do not use base class"
        pass


class SimpleSampler(Sampler):

    def __init__(self, sample_rate:int, has_header:bool = True):
        super().__init__(has_header)
        self.sample_rate = sample_rate

    def collect(self, index:int) -> bool:
        """
        returns True if we want to keep this file
        :param index:
        :return:
        """
        # if file has header and we are at the first line, then keep
        if self.sample_rate == 0:
            return True
        if self.has_header and index == 0:
            return True
        if index % self.sample_rate == 0:
            return True
        return False


class RandomSampler(Sampler):
    """
    Randomly sample file to create
    """

    def __init__(self, target_file_size:int, infile:str, has_header:bool = True):
        """

        :param target_file_size:
        :param infile:
        """
        super().__init__(has_header)
        self.target_file_size = target_file_size
        self.infile = infile

    def collect(self, index:int):
        assert False, "not yet implemented"
        pass
