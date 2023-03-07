import logging
from cagraph import CaGraph as cg

# set up logger
logging.basicConfig(filename='log_cg.log',format='%(asctime)s %(name)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', filemode='w')
_log = logging.getLogger('__CaGraph_log__')

# create nng object instance
graph = cg('/datasets/1055-2_D1_smoothed_calcium_traces.csv')

def gen_log():
    _log.info('Logging started.')
    graph.get_network_graph(threshold = 0.3)
    _log.info('Logging complete.')

if __name__ == '__main__':
    gen_log()