#!/bin/python
#
# this is a one time program used to add star_rating column back into feature files
# forgot to append the star_rating column to feature files so wrote this program to zip them together
#
import argparse
import logging
import pandas as pd



LOG_FORMAT = '%(asctime)s %(name)s.%(funcName)s:%(lineno)d %(levelname)s - %(message)s'
CLASS_COLUMN="star_rating"




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add a class column ')
    parser.add_argument("config_file", help="file with parameters to drive the permutations")
    parser.add_argument("-l", "--loglevel", help="log level ie, DEBUG", default="INFO")
    # get command line arguments
    args = parser.parse_args()

    # process argument
    if args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(format=LOG_FORMAT, level=loglevel)
    log = logging.getLogger(__name__)



    config = pd.read_csv(args.config_file)

    for i, row in config.iterrows():
        try:
            infile = row["infile"]
            outfile = row["outfile"]
            classfile = row["classfile"]

            log.info(f"infile: {infile}")
            log.info(f"outfile: {outfile}")
            log.info(f"classfile: {classfile}")

            in_pd = pd.read_csv(infile)
            classes = pd.read_csv(classfile)[CLASS_COLUMN]
            in_pd[CLASS_COLUMN] = classes

            in_pd.to_csv(outfile, index=False)
        except Exception as e:
            log.error(str(e))
