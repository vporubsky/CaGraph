import pytest
from neuronal_network_graph import neuronal_network_graph as nng
from neuronal_network_graph import insufficient_data

# create nng object instance
#nn = nng('14-0_D1_smoothed_calcium_traces.npy')

def test_raises_insufficient_data():
    with pytest.raises(insufficient_data):
        tmp = 0 # write some test that raises the insufficient_data warning

def test_neuronal_network_graph():
    with pytest.raises(TypeError):
        nng("14-0_D1_smoothed_calcium_traces.csv")

if __name__ == '__main__':
    #pytest.main() # this attempts to run all modules in the working dir which start with 'test'
    print('Running test_nng.py as main method.')