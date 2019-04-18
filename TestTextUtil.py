import unittest2
import text_util as tu


class TestTextUtil(unittest2.TestCase):

    def test_make_lowercase(self):
        text = 'ALLUPPER'
        converted = tu.make_lowercase(text)
        self.assertEqual(converted, "allupper",
                         f"{text} not converted properly")

    def test_remove_special_chars(self):
        text = "aren't"
        converted = tu.remove_special_chars(text)
        self.assertEqual(converted, "arent",
                         f'{converted} should be arent')
        text = "a\nb"

        converted = tu.remove_special_chars(text)
        self.assertEqual(converted, "ab",
                         f'{converted} should be ab')

    def test_remove_accent_chars(self):
        text = "souhaité"
        converted = tu.remove_accented_chars(text)
        self.assertEqual(converted, "souhaite",
                         f'{converted} not converted properly')


if __name__ == '__main__':
    unittest.main()
