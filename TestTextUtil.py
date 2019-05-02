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

        text = "[Apple MFi Certified] Charging Cable for iPhone 5 & 6 by OnyxVolt™ - SmartCharge Technology Accelerates Charging and Syncing Speeds to all Your Latest iPads, iPods, & IOS Devices - (2x 1m / 3.2ft Cord) Comes with OnyxVolt™ Unlimited Lifetime Guarantee!"
        converted = tu.remove_special_chars(text)
        self.assertEqual(converted,
                         "Apple MFi Certified Charging Cable for iPhone 5 6 by OnyxVolt SmartCharge Technology Accelerates Charging and Syncing Speeds to all Your Latest iPads iPods IOS Devices 2x 1m 32ft Cord Comes with OnyxVolt Unlimited Lifetime Guarantee",
                         f'({converted}) contains special characters')

    def test_remove_accent_chars(self):
        text = "souhaité"
        converted = tu.remove_accented_chars(text)
        self.assertEqual(converted, "souhaite",
                         f'{converted} not converted properly')

    def test_remove_newlines(self):
        text = "a\nb\rc"
        converted = tu.remove_newlines(text)
        self.assertEqual(converted, "abc",
                         f'{converted} contains newlines')

    def test_remove_amazon_tags(self):
        text = "[[VIDEOID:e8181b266aea9bc5006b93fba5078b2c]] This is a great little speaker more text"
        converted = tu.remove_amazon_tags(text)
        self.assertEqual(converted,
                         "This is a great little speaker more text",
                         f'{converted} contains amazon tags')

        text = "[[VIDEOID:e8181b266aea9bc5006b93fba5078b2c]] This is a great little speaker [[ASIN:e8181b266aea9bc5006b93fba5078b2c]] more text"
        converted = tu.remove_amazon_tags(text)
        self.assertEqual(converted,
                         "This is a great little speaker more text",
                         f'{converted} contains amazon tags')


if __name__ == '__main__':
    unittest.main()
