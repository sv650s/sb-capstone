import csv
import os
import logging
from .file_samplers import Sampler


logger = logging.getLogger(__name__)


def convert_tsv_to_csv(infile: str, outfile: str, sampler:Sampler = None):
    """
    Read a tsv file line by line then covert it into a readable csv file
    :param infile: input TSV file
    :param outfile: output CSV file
    :param sampling_rate: sepcify smapling rate to reduce file size. 1 is default = no sampling
    :return:
    """

    logger.debug(f"Converting {infile} to {outfile} with sampling {sampler}")

    if os.path.isfile(infile):

        counter = 0
        with open(infile, "r+") as old_f, open(outfile, "w") as new_f:
            writer = csv.writer(new_f)
            for line in old_f:
                # never sample 0 because that's the header column
                if sampler is None or sampler.collect(counter):
                    data = line.strip('\n').split('\t')
                    writer.writerow(data)
                counter += 1





