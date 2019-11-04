import util.text_util as tu
from tools import amazon_review_preprocessor as pa


class TestTextUtil(object):

    def test_make_lowercase(self):
        text = 'ALLUPPER'
        converted = tu.make_lowercase(text)
        assert converted == "allupper", \
                         f"{text} not converted properly"

    def test_remove_special_chars(self):
        text = "aren't"
        converted = tu.remove_special_chars(text)
        assert converted == "aren t", \
                         f'{converted} should be arent'
        text = "a\nb"

        converted = tu.remove_special_chars(text)
        assert converted == "ab", \
                         f'{converted} should be ab'

        text = "[Apple MFi Certified] Charging Cable for iPhone 5 & 6 by OnyxVolt™ - SmartCharge Technology Accelerates Charging and Syncing Speeds to all Your Latest iPads, iPods, & IOS Devices - (2x 1m / 3.2ft Cord) Comes with OnyxVolt™ Unlimited Lifetime Guarantee!"
        converted = tu.remove_special_chars(text)
        assert converted == \
                         "Apple MFi Certified Charging Cable for iPhone 5 6 by OnyxVolt SmartCharge Technology Accelerates Charging and Syncing Speeds to all Your Latest iPads iPods IOS Devices 2x 1m 3 2ft Cord Comes with OnyxVolt Unlimited Lifetime Guarantee", \
                         f'({converted}) contains special characters'

    def test_remove_accent_chars(self):
        text = "souhaité"
        converted = tu.remove_accented_chars(text)
        assert converted == "souhaite", \
                         f'{converted} not converted properly'

    def test_remove_newlines(self):
        text = "a\nb\rc"
        converted = tu.remove_newlines(text)
        assert converted == "abc", \
                         f'{converted} contains newlines'

    def test_remove_amazon_tags(self):
        text = "[[VIDEOID:e8181b266aea9bc5006b93fba5078b2c]] This is a great little speaker more text"
        converted = pa.remove_amazon_tags(text)
        assert converted == "This is a great little speaker more text", \
                         f'{converted} contains amazon tags'

        text = "[[VIDEOID:e8181b266aea9bc5006b93fba5078b2c]] This is a great little speaker [[ASIN:e8181b266aea9bc5006b93fba5078b2c]] more text"
        converted = pa.remove_amazon_tags(text)
        assert converted == "This is a great little speaker more text", \
                         f'{converted} contains amazon tags'


    def test_remove_http_links(self):
        text = "vacation to use it.<br /><br />http://www.amazon.com/gp/product/B00SBCWC3C?redirect=true&ref_=cm_cr_ryp_prd_ttl_sol_0"
        converted = pa.remove_http_links(text)
        assert converted == "vacation to use it.<br /><br />", \
                        f'[{converted}] still contains link'


        text = "James Fletcher.)<br />http://youtu.be/Z0Lan54TL0A<br /><br />So that's it!"
        expected = "James Fletcher.)<br /> /><br />So that's it!"
        converted = pa.remove_http_links(text)
        assert converted == expected, \
            f'expected [{expected}] got [{converted}]'

        text = "http://youtu.be/Z0Lan54TL0A<br /><br />So that's it!  It's a pretty good little radio!  I'll try to update in 5 months when it is winter and see how they"
        expected = " /><br />So that's it!  It's a pretty good little radio!  I'll try to update in 5 months when it is winter and see how they"
        converted = pa.remove_http_links(text)
        assert converted == expected, \
            f'expected [{expected}] got [{converted}]'

        text = "James Fletcher.)<br />https://youtu.be/Z0Lan54TL0A<br /><br />So that's it!"
        expected = "James Fletcher.)<br /> /><br />So that's it!"
        converted = pa.remove_http_links(text)
        assert converted == expected, \
            f'expected [{expected}] got [{converted}]'


    def test_stem_words(self):
        text = "hello running man give sadly"
        converted = tu.stem_text(text)
        assert converted == "hello run man give sadli", \
                         f'{converted} did not stem correctly'

    def test_get_contractions(self):
        # negative test case
        text = "no contractions here"
        matches = tu.get_contractions(text)
        assert len(matches) == 0, \
                        f'sound not find any matches matches: {matches}'

        text = "no contractions here: blah'blah"
        matches = tu.get_contractions(text)
        assert len(matches) == 0, \
                         f'sound not find any matches matches: {matches}'

        # test one match
        text = "one contractions can't here"
        matches = tu.get_contractions(text)
        assert len(matches) == 1, \
                         f'should not one match: {matches}'
        assert matches[0] == "can't", \
                         f"should find can't but found': {matches[0]}"


        # test 2 matches
        text = "two contractions can't here and couldn't"
        matches = tu.get_contractions(text)
        assert len(matches) == 2, \
                         f'should not 2 matches: {matches}'
        assert matches[0] == "can't", \
                         f"should find can't but found': {matches[0]}"
        assert matches[1] == "couldn't", \
                         f"should find can't but found': {matches[0]}"


        # test multiple matches
        text = "multiple contractions: aren't blah 3490mak0; ab'dec can't here and couldn't wouldn't"
        matches = tu.get_contractions(text)
        assert len(matches) == 4, \
                         f'original [{text}] should find 4 matches: {matches}'
        assert matches[0] == "aren't", \
                         f"should find can't but found': {matches[0]}"
        assert matches[1] == "can't", \
                         f"should find can't but found': {matches[0]}"
        assert matches[2] == "couldn't", \
                         f"should find can't but found': {matches[0]}"
        assert matches[3] == "wouldn't", \
                         f"should find can't but found': {matches[0]}"


        # take a line from sample file
        text = "US 50800750 R15V54KBMTQWAY B00XK95RPQ 516894650 Selfie Stick Fiblastiq&trade; Extendable Wireless Bluetooth Selfie Stick with built-in Bluetooth Adjustable Phone Holder Wireless 4 0 0 N N A fun little gadget I’m embarrassed to admit that until recently, I have had a very negative opinion about “selfie sticks” aka “monopods” aka “narcissticks.” But having reviewed a number of them recently, they’re growing on me. This one is pretty nice and simple to set up and with easy instructions illustrated on the back of the box (not sure why some reviewers have stated that there are no instructions when they are clearly printed on the box unless they received different packaging than I did). Once assembled, the pairing via bluetooth and use of the stick are easy and intuitive. Nothing to it.<br /><br />The stick comes with a USB charging cable but arrived with a charge so you can use it immediately, though it’s probably a good idea to charge it right away so that you have no interruption of use out of the box. Make sure the stick is switched to on (it will light up) and extend your stick to the length you desire up to about a yard’s length and snap away.<br /><br />The phone clamp held the phone sturdily so I wasn’t worried about it slipping out. But the longer you extend the stick, the harder it is to maneuver.  But that will happen with any stick and is not specific to this one in particular.<br /><br />Two things that could improve this: 1) add the option to clamp this in portrait orientation instead of having to try and hold the stick at the portrait angle, which makes it feel unstable; 2) add the opening for a tripod so that this can be used to sit upright on a table for skyping and facetime eliminating the need to hold the phone up with your hand, causing fatigue.<br /><br />But other than that, this is a nice quality monopod for a variety of picture taking opportunities.<br /><br />I received a sample in exchange for my honest opinion. 2015-08-31 "
        matches = tu.get_contractions(text)
        assert len(matches) == 5, \
                         f'original [{text}] should find 4 matches: {matches}'

    def test_alphanumerica_words(self):
        text = "iphone 6s"
        expected = "iphone"
        newtext = tu.remove_alphanumeric_words(text)
        assert newtext == expected, "did not remove last word correctly"

        text = "iphone a6s"
        expected = "iphone"
        newtext = tu.remove_alphanumeric_words(text)
        assert newtext == expected, "did not remove number in middle of word"

        text = "iphone 6a6s"
        expected = "iphone"
        newtext = tu.remove_alphanumeric_words(text)
        assert newtext == expected, "did not remove letter in middle of word"

        text = "iphone a6"
        expected = "iphone"
        newtext = tu.remove_alphanumeric_words(text)
        assert newtext == expected, "did not remove last word correctly"

        text = "iphone s6 6s a6b"
        expected = "iphone"
        newtext = tu.remove_alphanumeric_words(text)
        assert newtext == expected, "did not remove different types of mixed words"

        text = "6s iphone"
        expected = "iphone"
        newtext = tu.remove_alphanumeric_words(text)
        assert newtext == expected, "did not remove first word correctly"

        text = "apple 6s iphone"
        expected = "apple iphone"
        newtext = tu.remove_alphanumeric_words(text)
        assert newtext == expected, "did not remove second word correctly"

        text = "GSM12345 apple 6s iphone zt3500"
        expected = "apple iphone"
        newtext = tu.remove_alphanumeric_words(text)
        assert newtext == expected, "did not remove multiple words correctly"

        # TODO: not sure what to do with these standalone numbers yet
        text = "GSM12345 apple 6s 5 55 iphone zt3500"
        expected = "apple 5 iphone"
        newtext = tu.remove_alphanumeric_words(text)
        assert newtext == expected, "did not remove first word correctly"


