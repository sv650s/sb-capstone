import pytest
import util.file_util as fu


class TestFileUtil(object):

    def test_get_report_filename(self):
        infile = "dataset/amazon/abcd.csv"
        expected = "reports/abcd-report.csv"
        outfile = fu.get_report_filename(infile)
        assert outfile == expected, "mismatch report file"

        infile = "dataset/amazon/abcd.csv"
        expected = "a/outpath/abcd-report.csv"
        outfile = fu.get_report_filename(infile, "a/outpath/")
        assert outfile == expected, "not handling outpath correctly"

        infile = "dataset/amazon/abcd.csv"
        expected = "/a/outpath/abcd-report.csv"
        outfile = fu.get_report_filename(infile, "/a/outpath/")
        assert outfile == expected, "not handling outpath correctly"
