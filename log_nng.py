import logging
from neuronal_network_graph import neuronal_network_graph as nng

# set up logger
logging.basicConfig(filename='log_nng.log',format='%(asctime)s %(name)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', filemode='w')
_log = logging.getLogger('__neuronal_network_graph_log__')

# create nng object instance
nn = nng('1055-2_D1_smoothed_calcium_traces.csv')

def gen_log():
    _log.info('Logging started.')
    nn.get_context_A_graph(threshold = 0.3)
    nn.get_context_B_graph(threshold = 0.3)
    nn.get_null_context_A_graph(threshold = 0.3)
    nn.get_null_context_B_graph(threshold=0.3)
    _log.info('Logging complete.')

if __name__ == '__main__':
    gen_log()