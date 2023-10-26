import unittest
import cagraph as cg

# Todo: need to add test dataset, and specify that some tests are specific to that dataset
# Todo: make sure test data is hidden to user when package uploaded to PyPi

class CaGraphTestSuite(unittest.TestCase):
    """
    Author: Veronica Porubsky [Github: https://github.com/vporubsky][ORCID: https://orcid.org/0000-0001-7216-3368]

    Test suite to check that the CaGraph class functions as expected.
    """
    @classmethod
    def setUpClass(cls):
        """Runs before any tests have been executed to set up the testing class."""
        cls.DATA_PATH = './datasets/'
        cls.TEST_DATA_PATH = cls.DATA_PATH + 'bla_dataset.csv'
        cls.INCORRECT_DATA_PATH = cls.DATA_PATH + 'bla_dataset.tsv'
        cls.cg_graph = cg.CaGraph(data=cls.TEST_DATA_PATH)
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

    def test_dt(self):
        """Test raises an AssertionError if the interval between time points is not 0.1 seconds."""
        self.assertEqual(0.1, self.cg_graph.dt)

    def test_getGraph(self):
        """"""
        self.cg_graph.get_graph()

    def test_assertEqual(self):
        """
        Dummy test.

        Test passes because True == True."""
        self.assertEqual(True, True)

    def test_createCaGraphDataAsListTypeError(self):
        """
        Test that data cannot be passed as a list.
        """
        with self.assertRaises(TypeError):
            cg.CaGraph(data=[1, 2, 3])

    def test_createCaGraphFileTypeError(self):
        """
        Test that the CaGraph object creation method does not accept TSV files.
        """
        with self.assertRaises(TypeError):
            cg.CaGraph(data=self.INCORRECT_DATA_PATH)

    def test_changeThreshold(self):
        """
        Test that the threshold can be updated by the user.
        """
        cg_obj = cg.CaGraph(data=self.TEST_DATA_PATH, threshold=0.2)
        init_threshold = cg_obj.threshold
        cg_obj.threshold = 0.5
        self.assertNotEqual(cg_obj.threshold, init_threshold)

    def test_resetGraph(self):
        """
        Test that calling the reset method resets the threshold to the intial state when the data was loaded.
        """
        cg_obj = cg.CaGraph(data=self.TEST_DATA_PATH, threshold=0.2)
        init_threshold = cg_obj.threshold
        cg_obj.threshold = 0.5
        cg_obj.reset()
        self.assertEqual(cg_obj.threshold,init_threshold)

    def test_changeThresholdTypeError(self):
        """
        Test that the threshold cannot be updated to a non-float value.
        """
        with self.assertRaises(TypeError):
            self.cg_graph.threshold = '0.5'

    def test_changeThresholdValueError(self):
        """
        Test that the threshold cannot be updated to a value outside the range [0,1].
        """
        with self.assertRaises(ValueError):
            self.cg_graph.threshold = 1.5

    def test_changeThresholdValueError2(self):
        """
        Test that the threshold cannot be updated to a value outside the range [0,1].
        """
        with self.assertRaises(ValueError):
            self.cg_graph.threshold = -0.5


    def test_inputTypeError(self):
        """
        Test that the input data must be of type string.
        """
        with self.assertRaises(TypeError):
            cg.CaGraph(data=1)

    def test_inputFileNotFoundError(self):
        """
        Test that the input data must be a valid file path.
        """
        with self.assertRaises(FileNotFoundError):
            cg.CaGraph(data='bla.csv')

    def test_inputFileTypeError(self):
        """
        Test that the input data must be a valid file type.
        """
        with self.assertRaises(TypeError):
            cg.CaGraph(data=self.INCORRECT_DATA_PATH)
            
    # Todo: test that input data must be in right format - error when file not found or wrong filetype passed




if __name__ == '__main__':
    unittest.main()