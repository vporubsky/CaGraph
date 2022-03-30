import unittest
from neuronal_network_graph import NeuronalNetworkGraph
import os

class NeuronalNetworkGraphTestSuite(unittest.TestCase):
    """
    Author: Veronica Porubsky [Github: https://github.com/vporubsky][ORCID: https://orcid.org/0000-0001-7216-3368]

    Test suite to check that the NeuronalNetworkGraph library functions as expected during development.
    """
    @classmethod
    def setUpClass(cls):
        """Runs before any tests have been executed."""
        cls.nng = NeuronalNetworkGraph(data_file=os.getcwd() + '/LC-DG-FC-data/caData_day1_Th/2-1_D1_smoothed_calcium_traces.csv')
        pass

    @classmethod
    def tearDownClass(cls):
        "Runs after all tests have been executed."
        pass

    def setUp(self):
        """Runs before each test. Creates NeuronalNetworkGraph object which can be used in each test."""

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


if __name__ == '__main__':
    unittest.main()