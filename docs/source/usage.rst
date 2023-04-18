Usage
=====

.. _installation:

Installation
------------

To use cagraph, first install it using pip:

.. code-block:: console

   (.venv) $ pip install cagraph

Generating CaGraph object
----------------

To create a CaGraph object which can be used to build and analyze graphs
you can use the ``cagraph.CaGraph()`` Class:

.. autofunction:: cagraph.CaGraph()

Generating graphs
----------------

To construct a graph, you can use the ``CaGraph(data='data.csv').get_graph()`` function:

.. autofunction:: CaGraph(data='data.csv').get_graph()

The ``data`` parameter should be either a numpy.ndarray, CSV file, or NWB file. Otherwise, :py:func:`CaGraph(data='data.csv')`
will raise an exception.

.. autoexception::

For example:

>>> from cagraph import CaGraph
>>> cg = CaGraph(data='data.csv')
>>> cg_graph = cg.get_graph()

