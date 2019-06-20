"""
Abstract class for command line programs
"""
import pandas as pd
from datetime import datetime
import argparse
import logging
import util.file_util as fu
from util.program_util import Keys, Status, TimedReport, TIME_FORMAT, TRUE_LIST, LOG_FORMAT
import traceback2
import sys


CV_TIME_MIN = "cv_time_min"
MODEL_SAVE_TIME_MIN = "model_save_time_min"
FEATURE_PICKLE_TIME_MIN = "feature_pickle_time_min"


log = logging.getLogger(__name__)


class TimedProgram(object):
    """
    Abstract program iteration.  This represents what to do with each row of configuration file
    """

    def __init__(self, index, config_df, report=None, args=None):
        log.debug(f"ProgramIteration constructor - index: {index}")
        self.index = index
        self.config_df = config_df
        self.args = args
        if report:
            self.report = report
        else:
            self.report = TimedReport()

    def get_config(self, config: str) -> str:
        """
        get string configuration
        :param config:
        :return:
        """
        if pd.notnull(self.config_df[config]):
            return self.config_df[config]
        else:
            return None

    def get_config_int(self, config: str) -> int:
        """
        get int config
        :param config:
        :return:
        """
        if pd.notnull(self.config_df[config]):
            return int(self.config_df[config])
        else:
            return None

    def get_config_bool(self, config) -> bool:
        """
        gets a boolean type configuration
        :param config:
        :return:
        """
        if pd.notnull(self.config_df[config]):
            return self.config_df[config] in TRUE_LIST
        else:
            return False

    def get_config_list(self, config) -> list:
        """
        gets a list type config parameter. config is expected to be a comma separated list
        :param config:
        :return:
        """
        if pd.notnull(self.config_df[config]):
            return self.config_df[config].replace(" ", "").split(",")
        else:
            return []

    def get_report_dict(self):
        return self.report.get_report_dict()

    def record(self, key: str, value: str):
        """
        record key value into report
        :param key:
        :param value:
        :return:
        """
        self.report.record(key, value)

    def start_timer(self, key: str):
        self.report.start_timer(key)

    def stop_timer(self, key: str):
        self.report.end_timer(key)

    def get_infile(self):
        """
        convenience method to get infile
        :return:
        """
        self.report.record("file", self.get_config("data_file"))
        return f'{self.get_config("data_dir")}/{self.get_config("data_file")}'

    def execute(self):
        raise Exception(f"Not yet implemented. {__class__} must implement this method.")


class ConfigBasedProgram(object):

    def __init__(self, description: str, program: TimedProgram):
        """
        :param description:  description of program
        :param program:  class of iteration object
        """
        self.description = description
        self.parser = argparse.ArgumentParser(description=description)
        self.parser.add_argument("config_file", help="file with parameters to drive the permutations")
        self.parser.add_argument("-l", "--loglevel", help="log level ie, DEBUG", default="INFO")

        self.program = program
        self.config_df = None
        self.report_df = None
        self.report_file = None
        self.config_file = None

    def get_report_file_name(self):
        _, config_basename = fu.get_dir_basename(self.config_file)
        reportfile = f'reports/{config_basename}-report.csv'
        log.debug(f'Report file: {reportfile}')
        return reportfile

    def add_argument(self, *args, **kwargs):
        """
        pass parameters to parser
        :param args:
        :return:
        """
        self.parser.add_argument(*args, **kwargs)

    def record_success(self, index, report):
        self.report_df = self.report_df.append(report, ignore_index=True)
        self.report_df.loc[index, Keys.STATUS] = Status.SUCCESS
        self.report_df.loc[index, Keys.STATUS_DATE] = datetime.now().strftime(TIME_FORMAT)
        self.report_df.loc[index, Keys.CONFIG_FILE] = self.config_file
        self.report_df.to_csv(self.report_file, index=False)

        self.config_df.loc[index, Keys.STATUS] = Status.SUCCESS
        self.config_df.loc[index, Keys.STATUS_DATE] = datetime.now().strftime(TIME_FORMAT)

    def record_failure(self, index, err_message):
        self.config_df.loc[index, Keys.STATUS] = Status.FAILED
        self.config_df.loc[index, Keys.STATUS_DATE] = datetime.now().strftime(TIME_FORMAT)
        self.config_df.loc[index, Keys.MESSAGE] = err_message

    def main(self):
        # get command line arguments
        args = self.parser.parse_args()

        # process argument
        if args.loglevel is not None:
            loglevel = getattr(logging, args.loglevel.upper(), None)
        logging.basicConfig(format=LOG_FORMAT, level=loglevel)

        self.config_file = args.config_file
        log.info(f'Reading config file: {self.config_file}')
        self.config_df = pd.read_csv(self.config_file)
        self.report_df = pd.DataFrame()
        self.report_file = self.get_report_file_name()

        for index, row in self.config_df.iterrows():
            try:
                iteration = self.program(index, row, args=args)
                iteration.execute()

                self.record_success(index, iteration.get_report_dict())

            except Exception as e:
                traceback2.print_exc(file=sys.stdout)
                log.error(str(e))
                self.record_failure(index, str(e))

            finally:
                log.info(f'Finished iteration: {index}')
                self.config_df.to_csv(self.config_file, index=False)
