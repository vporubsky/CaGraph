import unittest

class CaGraphTestSuite(unittest.TestCase):
    """
    Author: Veronica Porubsky [Github: https://github.com/vporubsky][ORCID: https://orcid.org/0000-0001-7216-3368]

    Test suite to check that the CaGraph library functions as expected during development.
    """

    def test_assertEqual(self):
        """Test raises an AssertionError because True != False. This results in a failure."""
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()