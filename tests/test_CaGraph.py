import unittest
import cagraph as cg
DATA_PATH = '/Users/veronica_porubsky/GitHub/CaGraph/datasets/'
TEST_DATA_PATH = DATA_PATH + 'bla_dataset.csv'
INCORRECT_DATA_PATH = DATA_PATH + 'bla_dataset.tsv'

class CaGraphTestSuite(unittest.TestCase):
    """
    Author: Veronica Porubsky [Github: https://github.com/vporubsky][ORCID: https://orcid.org/0000-0001-7216-3368]

    Test suite to check that the CaGraph class functions as expected.
    """

    def test_assertEqual(self):
        """
        Dummy test.

        Test passes because True == True."""
        self.assertEqual(True, True)

    def test_createCaGraphDataTypeError(self):
        """

        :return:
        """
        with self.assertRaises(TypeError):
            cg.CaGraph(data_file=[1, 2, 3])

    def test_createCaGraphFileTypeError(self):
        """

        :return:
        """
        with self.assertRaises(TypeError):
            cg.CaGraph(data_file=INCORRECT_DATA_PATH)



if __name__ == '__main__':
    unittest.main()