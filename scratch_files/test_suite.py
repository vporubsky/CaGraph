"""
Author: Veronica Porubsky [Github: https://github.com/vporubsky][ORCID: https://orcid.org/0000-0001-7216-3368]
Description: Test suite for DGNetworkGraph class, built upon test suite for parent NeuronalNetworkGraph class.
"""
from neuronal_network_graph import DGNetworkGraph as nng
import unittest

# Implement class of helper functions for unit test suite
class NeuronalNetworkGraphTestSuiteHelper:
    """
    Test suite helper functions for NeuronalNetworkGraph.
    """

    def __init__(self, test_network_data):
        self.test_network = test_network_data

    # Check for complex eigen values
    def is_extension_csv(self):
        """
        Function to check if network data is supplied as a .csv file.

        :return: bool
        """
        filename = TEST_NETWORK_DATA
        return filename.endswith('.csv')

    def generates_DGNetworkGraph_object(self):

        return


    # Add more helper functions to class as needed


# Implement unit test suite
class NeuronalNetworkGraphTestSuite(unittest.TestCase):
    """
    Test suite for NeuronalNetworkGraph.

    To set up the test suite, the user must supply a .csv data file containing calcium
    imaging data for a network of neurons.
    """

    def setUp(self):
        self.test_network_data = TEST_NETWORK_DATA
        self.test_network_object = nng(csv_file=TEST_NETWORK_DATA)

    def test_file_format(self):
        """
        Check if model system has complex eigenvalues after timecourse simulation.
        """
        self.assertTrue(NeuronalNetworkGraphTestSuiteHelper(test_network_data=self.test_network_data).is_extension_csv())

    def test_file_load(self):
        """

        """
        self.assertTrue(NeuronalNetworkGraphTestSuiteHelper().generates_DGNetworkGraph_object())



if __name__ == "__main__":
    # Load network data
    TEST_NETWORK_DATA = '2-1_D1_smoothed_calcium_traces.csv'

    # Run unit test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(NeuronalNetworkGraphTestSuite)
    _ = unittest.TextTestRunner().run(suite)