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


    def test_dir_basename(self):
        infile = "dir/dir2/file.csv"
        expected_dir = "dir/dir2"
        expected_file = "file"
        dir, file = fu.get_dir_basename(infile)
        assert expected_dir == dir, "mismatched dir"
        assert expected_file == file, "mismatched file"

        infile = "/dir/dir2/file.csv"
        expected_dir = "/dir/dir2"
        expected_file = "file"
        dir, file = fu.get_dir_basename(infile)
        assert expected_dir == dir, "mismatched dir"
        assert expected_file == file, "mismatched file"

        infile = "/dir2/file.csv"
        expected_dir = "/dir2"
        expected_file = "file"
        dir, file = fu.get_dir_basename(infile)
        assert expected_dir == dir, "mismatched dir"
        assert expected_file == file, "mismatched file"

        infile = "file.csv"
        expected_dir = None
        expected_file = "file"
        dir, file = fu.get_dir_basename(infile)
        assert expected_dir == dir, "mismatched dir"
        assert expected_file == file, "mismatched file"

