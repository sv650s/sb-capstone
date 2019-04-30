import csv
import argparse
import os
import logging


logger = logging.getLogger(__name__)


def convert_tsv_to_csv(infile: str, outfile: str, sampling_rate:int = 1):
    """
    Read a tsv file line by line then covert it into a readable csv file
    :param infile: input TSV file
    :param outfile: output CSV file
    :param sampling_rate: sepcify smapling rate to reduce file size. 1 is default = no sampling
    :return:
    """

    logger.debug(f"Converting {infile} to {outfile} with sampling rate of {sampling_rate}")

    if os.path.isfile(infile):

        counter = 0
        with open(infile, "r+") as old_f, open(outfile, "w") as new_f:
            writer = csv.writer(new_f)
            for line in old_f:
                # never sample 0 because that's the header column
                if counter == 0 or counter % sampling_rate == 0:
                    data = line.strip('\n').split('\t')
                    writer.writerow(data)
                counter += 1




if __name__ == '__main__':
    # allow this to run as a standalone program
    parser = argparse.ArgumentParser(description="Conver TSV to CSV")
    parser.add_argument("infile", help="Input TSV file")
    parser.add_argument("outfile", help="Output CSV file")
    parser.add_argument("-l", "--loglevel", help="log level")
    parser.add_argument("-s", "--sampling_rate", help="sampling rate. Max 100. default = 0", type=int, default=0)
    args = parser.parse_args()

    # set up logging
    LOG_FORMAT = "%(asctime)-15s %(levelname)-7s %(name)s.%(funcName)s" \
                 " [%(lineno)d] - %(message)s"
    # process argument
    loglevel = logging.INFO
    if args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)

    logging.basicConfig(format=LOG_FORMAT, level=loglevel)


    if args.sampling_rate is not None:
        convert_tsv_to_csv(args.infile, args.outfile, args.sampling_rate)
    else:
        convert_tsv_to_csv(args.infile, args.outfile)

