"""
Testing picking feature files and reading them back in to see if this might be a faster way of
processing
"""

import pandas as pd
import pickle
import logging
import util.file_util as fu
from util.program_util import Keys
from util.ConfigBasedProgram import ProgramIteration, ConfigBasedProgram

log = logging.getLogger(__name__)
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DATE_FORMAT = '%Y-%m-%d'
LOG_FORMAT = '%(asctime)s %(name)s.%(funcName)s[%(lineno)d] %(levelname)s - %(message)s'
TRUE_LIST = ["yes", "Yes", "YES", "y", "True", "true", "TRUE"]
CV_TIME_MIN = "cv_time_min"
MODEL_SAVE_TIME_MIN = "model_save_time_min"
PICKLE_WRITE_TIME_MIN = "pickle_write_time_min"
PICKLE_READ_TIME_MIN = "pickle_read_time_min"


class Pickler(ProgramIteration):

    def execute(self):
        log.debug(f'Executing {self.index}')

        infile = f'{self.get_config("data_dir")}/{self.get_config("data_file")}'
        outdir, out_basename = fu.get_dir_basename(infile)
        outfile = f'{outdir}/{out_basename}.pkl'
        log.debug(f'infile {infile}')
        log.debug(f'outfile {outfile}')
        self.report.record("file", infile)
        self.report.record("outfile", outfile)

        self.report.start_timer(Keys.FILE_LOAD_TIME_MIN)
        in_df = pd.read_csv(infile)
        self.report.end_timer(Keys.FILE_LOAD_TIME_MIN)
        log.debug(in_df.head())

        self.report.start_timer(PICKLE_WRITE_TIME_MIN)
        pickle.dump(in_df, open(outfile, 'wb'))
        self.report.end_timer(PICKLE_WRITE_TIME_MIN)

        if self.args.load_pickle:
            self.report.start_timer(PICKLE_READ_TIME_MIN)
            new_df = pickle.load(open(outfile, 'rb'))
            self.report.end_timer(PICKLE_READ_TIME_MIN)
            log.debug(new_df.head())

        self.record_success()


if __name__ == "__main__":

    program = ConfigBasedProgram("Pickles a file based on configuration file", Pickler)
    program.add_argument("--load_pickle", help="test loading pickle file after writing", action='store_true')
    program.main()
