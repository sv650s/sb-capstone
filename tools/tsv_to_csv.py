"""
Converts a TSV file to CSV file format

This program has the capability of providing sampling so you can reduce the size of the file at the same time
"""
import argparse
import logging
from util.file_samplers import SimpleSampler
from util.file_util import convert_tsv_to_csv


logger = logging.getLogger(__name__)






if __name__ == '__main__':
    # allow this to run as a standalone program
    parser = argparse.ArgumentParser(description="Conver TSV to CSV")
    parser.add_argument("infile", help="Input TSV file")
    parser.add_argument("outfile", help="Output CSV file")
    parser.add_argument("-l", "--loglevel", help="log level")
    parser.add_argument("-f", "--function", help="functions available: tsv_to_csv (default: tsv_to_csv)", default="tsv_to_csv")
    parser.add_argument("-s", "--sampling_param", help="sampling param. "
            "For simple sampler - this means the frequency to keep. Max 100. default = 0", type=int, default=0)
    args = parser.parse_args()

    # set up logging
    LOG_FORMAT = "%(asctime)-15s %(levelname)-7s %(name)s.%(funcName)s" \
                 " [%(lineno)d] - %(message)s"
    # process argument
    loglevel = logging.INFO
    if args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)

    logging.basicConfig(format=LOG_FORMAT, level=loglevel)


    sampler = SimpleSampler(sample_rate=args.sampling_param, has_header=True)

    if(args.function == "tsv_to_csv"):
        if args.sampling_param is not None:
            convert_tsv_to_csv(args.infile, args.outfile, sampler)
        else:
            convert_tsv_to_csv(args.infile, args.outfile)
