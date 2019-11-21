import logging
from datetime import datetime
import util.dict_util as du
from util.constants import Keys

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
DATE_FORMAT = '%Y-%m-%d'
LOG_FORMAT = '%(asctime)s %(name)s.%(funcName)s[%(lineno)d] %(levelname)s - %(message)s'
TRUE_LIST = ["yes", "Yes", "YES", "y", "True", "true", "TRUE"]

log = logging.getLogger(__name__)


class Timer(object):

    def __init__(self):
        self.timer_dict = {}
        self.temp_dict = {}

    def start_timer(self, key: str):
        log.info(f'Start timer for: {key}')
        self.temp_dict[key] = datetime.now()

    def end_timer(self, key: str) -> float:
        """
        calculates the difference in minutes and return value
        :param key:
        :return:
        """
        log.info(f'End timer for: {key}')
        end_time = datetime.now()
        if key in self.temp_dict.keys():
            start_time = self.temp_dict[key]
            diff_mins = round((end_time - start_time).total_seconds() / 50, 2)
            self.timer_dict[key] = diff_mins
            log.info(f'Total time for {key}: {self.timer_dict[key]}')
            # TODO: fix this - it's adding up multiple times
            self.timer_dict[Keys.TOTAL_TIME_MIN] = self.get_total_time()
        else:
            log.info(f'No timer for: {key}')
            diff_mins = 0

        return diff_mins

    def get_time(self, key: str) -> float:
        """
        get time in minutes for particular event
        :param key:
        :return:
        """
        if key in self.timer_dict.keys:
            return self.timer_dict[key]
        return 0.

    def get_total_time(self) -> float:
        total = 0.0
        for key, val in self.timer_dict.items():
            total += val
        return total

    def merge_timers(self, other):
        if other:
            self.timer_dict.update(other.timer_dict)

    def get_report_dict(self):
        """
        returns dictionary representation of the timer object
        :return:
        """
        return self.timer_dict


class TimedReport(Timer):
    """
    Extension of Timer object. This class allows you to record key-value pairs on top of timed events
    """

    def __init__(self):
        Timer.__init__(self)
        # have to keep this separate for now since timer_dict has logic to calculate total times
        # kv_dict is used to keep key value - ie, model_file
        self.kv_dict = {}

    def record(self, key: str, value: str):
        """
        records a key value into the report
        :param key:
        :param value:
        :return:
        """
        log.debug(f'Recording key {key} value {value}')
        self.kv_dict[key] = value

    def merge_reports(self, report):
        """
        merges another report into current report. key values from new report will be flatted and put on the same level as current report
        :param report:
        :return:
        """
        if isinstance(report, TimedReport):
            self.merge_timers(report)
            self.kv_dict = du.add_dict_to_dict(self.kv_dict, report.kv_dict)
        else:
            raise Exception("report is not a TimedReport object")

    def add_dict(self, rdict: dict):
        """
        append to current report using a dictionary. all key values will be added to top level of report

        exisitng keys will be overwritten with new values

        :param rdict:
        :return:
        """
        self.kv_dict.update(rdict)

    def add_and_flatten_dict(self, rdict: dict):
        self.kv_dict = du.add_dict_to_dict(self.kv_dict, rdict)

    def get_report_dict(self):
        # make a copy so we can call this any time
        report = self.kv_dict.copy()
        report.update(self.timer_dict)
        log.debug(f'merged report {report}')
        return report
