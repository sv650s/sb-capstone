"""
Taks in  list of files from configuration. will load each configuration file and time how long it takes
to load the file and record it into file_load_time_min column

Required columns:
    data_dir
    infile
"""

import pandas as pd
import logging
from datetime import datetime
import argparse
import traceback2





LOG_FORMAT = '%(asctime)s %(name)s.%(funcName)s:%(lineno)d %(levelname)s - %(message)s'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reads in data files and records load time for the file in the original config file')
    parser.add_argument("config_file", help="configuration file. Requires data_dir and infile columns")
    parser.add_argument("-l", "--loglevel", help="log level ie, DEBUG", default="INFO")

    # get command line arguments
    args = parser.parse_args()

    # process argument
    if args.loglevel is not None:
        loglevel = getattr(logging, args.loglevel.upper(), None)
    logging.basicConfig(format=LOG_FORMAT, level=loglevel)
    log = logging.getLogger(__name__)

    # read in configuration
    config_df = pd.read_csv(args.config_file)
    log.debug(config_df.head())

    for index, row in config_df.iterrows():
        infile = f'{row["data_dir"]}/{row["infile"]}'


        start_load_time = None
        end_load_time = None


        try:

            start_load_time = datetime.now()

            log.info(f"Loading {infile}")
            df = pd.read_csv(infile)
            log.debug(df.shape)

            end_load_time = datetime.now()
            config_df.loc[index, "status"] = "success"

        except Exception as e:
            log.error(traceback2.format_exc())
            config_df.loc[index, "status"] = "failed"
        finally:
            if start_load_time and end_load_time:
                load_time_min = round((end_load_time - start_load_time).total_seconds() / 60, 2)
            elif start_load_time:
                load_time_min = round((datetime.now() - start_load_time).total_seconds() / 60, 2)
            config_df.loc[index, "file_load_time_min"] = load_time_min
            config_df.to_csv(args.config_file, index=False)
