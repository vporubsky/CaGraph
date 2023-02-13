import unittest
from ca_graph import CaGraph
import os

class NeuronalNetworkGraphTestSuite(unittest.TestCase):
    """
    Author: Veronica Porubsky [Github: https://github.com/vporubsky][ORCID: https://orcid.org/0000-0001-7216-3368]

    Test suite to check that the CaGraph library functions as expected during development.
    """
    @classmethod
    def setUpClass(cls):
        """Runs before any tests have been executed."""
        cls.nng = CaGraph(data_file=os.getcwd() + '/LC-DG-FC-data/2-1_D1_smoothed_calcium_traces.csv')
        pass

    @classmethod
    def tearDownClass(cls):
        "Runs after all tests have been executed."
        pass

    def setUp(self):
        """Runs before each test. Creates CaGraph object which can be used in each test."""

        pass

    def tearDown(self):
        """Runs after each test."""
        pass

    def test_assertEqual(self):
        """Test raises an AssertionError because True != False. This results in a failure."""
        self.assertEqual(True, False)

    def test_assertIsNotNone(self):
        """Test raises an AssertionError because the value passed is None. This results in a failure."""
        self.assertIsNotNone(None)

    def test_assertIs(self):
        """Test raises an AssertionError because the values passed are not the same. This results in a failure."""
        self.assertIs(1,2)

    def test_dt(self):
        """Test raises an AssertionError if the interval between time points is not 0.1 seconds."""
        TIME = self.nng.time
        dt = TIME[1] - TIME[0]
        self.assertEqual(0.1, dt)

    def test_getNetworkGraph(self):
        """"""
        self.nng.get_network_graph()

    def test_plotCDF(self):
        """Test """
        self.nng.plot_CDF(data=self.nng.get_clustering_coefficient())

    def test_generateThreshold(self):
        """Test that the CaGraph.generate_threshold() function performs as expected when threshold is not set."""
        self.assertEqual(0.3, self.nng.threshold)



if __name__ == '__main__':
    unittest.main()