import unittest

class VisualizationTestSuite(unittest.TestCase):
    """
    Author: Veronica Porubsky [Github: https://github.com/vporubsky][ORCID: https://orcid.org/0000-0001-7216-3368]

    Test suite to check that the visualization module functions as expected.
    """

    def test_assertEqual(self):
        """
        Dummy test.
        
        Test passes because True == True."""
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()