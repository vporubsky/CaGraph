<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/vporubsky/CaGraph">
    <img src="figures/icon.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">CaGraph</h3>

  <p align="center">
    Graph theory analysis and visualization package for calcium imaging timeseries data. The package makes it simple for experimental researchers to generate graphs 
    of functional networks and inspect their topology, using NetworkX to compute useful graph theory metrics. Bokeh is used to assist with interactive plotting of graph networks.
    <br />
    <a href="https://cagraph.readthedocs.io/en/latest/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/vporubsky/CaGraph">View Demo</a>
    ·
    <a href="https://github.com/vporubsky/CaGraph/issues">Report Bug</a>
    ·
    <a href="https://github.com/vporubsky/CaGraph/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#features">Features</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![CaGraph Screen Shot][product-screenshot]](https://raw.githubusercontent.com/vporubsky/CaGraph/main/figures/figure_1.png)


## 🚀 Features

- **Graph-theoretic analysis of calcium imaging time-series**  
  - Pearson, Spearman, cross-correlation, mutual information, partial correlation, Granger causality  
  - Flexible thresholding strategies (static, statistical, shuffled baselines)

- **Preprocessing utilities**  
  - Data validation and cleaning  
  - Event-based shuffling to create null distributions  
  - Threshold estimation for network edge significance  
  - (Recommended) works best with **CNMF-deconvolved data** (e.g. [CalmAn CNMF-E](https://github.com/flatironinstitute/CaImAn))

- **Visualization**  
  - Publication-ready plots with Matplotlib/Seaborn  
  - Interactive Bokeh plots for exploring networks in a browser  
  - Customizable node attributes (size, color by degree, communities, etc.)  
  - Adjustable edge transparency, thresholding, and layouts  

- **Extensible design**  
  - Drop-in modules for new metrics or workflows  
  - Easy integration with Pandas, NetworkX, and scientific Python stack  


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

Ensure you have Python (3.7+).

Here is a list of software required for the CaGraph project and how to install the packages.

* networkx
  ```sh
  pip install networkx
  ```
* bokeh
  ```sh
  pip install bokeh
  ```  
* numpy
  ```sh
  pip install numpy
  ```  
* matplotlib
  ```sh
  pip install matplotlib
  ```  
* pynwb
  ```sh
  pip install pynwb
  ```  
* scipy
  ```sh
  pip install scipy
  ```  
* seaborn
  ```sh
  pip install seaborn
  ```  
* statsmodels
  ```sh
  pip install statsmodels
  ```  
### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/vporubsky/CaGraph.git
   ```
2. Install via PyPI
   ```sh
   pip install cagraph
   ```



<!-- USAGE EXAMPLES -->
## Usage

Examples of best practices for using the package are discussed. Additional screenshots, code examples and demos will be included formally in documentation.

###Recommended preprocessing

For most use cases, we recommend deconvolving calcium imaging data using CNMF (e.g., CNMF-E via CaImAn
) before constructing graphs with CaGraph. This reduces noise and improves the biological interpretability of inferred networks.

### Modules Overview

CaGraph is organized into three main modules:

1. cagraph (Core Graph Analysis)

* Construct graphs from calcium imaging signals
* Compute graph metrics (degree, clustering coefficient, modularity, path length, assortativity, etc.)
* Run community detection
* Export results into reports (pandas.DataFrame)

Example:
~~~python
    from cagraph.cagraph import CaGraph
    cg = CaGraph(data=deconvolved_traces, sampling_rate=10.0)
    cg.compute_graph(method="pearson", threshold=0.3)
    metrics = cg.compute_metrics()
    cg.get_report()
~~~

2. preprocess (Data Preparation & Utilities)

* Validation: check calcium traces for NaNs, size mismatches, or invalid frames
* Deconvolution interface: supports importing CNMF outputs
* Event-based shuffling: create null distributions for statistical comparisons
* Threshold estimation: derive correlation thresholds from shuffled data

Example:

~~~python
import cagraph.preprocess as prep

shuffled = prep.shuffle_events(deconvolved_traces)
threshold = prep.generate_threshold(shuffled)
~~~

3. visualization (Static & Interactive Plotting)

* Matplotlib / Seaborn: for static figures (heatmaps, adjacency plots, distributions)
* Bokeh: interactive network exploration with zoom, hover, and filtering
* Customize node sizes, colors, and edge transparency based on metrics or metadata

Example:

~~~python
import cagraph.visualization as viz

# Interactive network with neuron coordinates
viz.interactive_network(cagraph_obj=cg, position=neuron_coordinates)
~~~

### Quickstart

~~~python
import numpy as np
from cagraph import CaGraph
import cagraph.preprocess as prep
import cagraph.visualization as viz

# Example: simulated calcium traces
time = np.linspace(0, 100, 1000)       # 1000 time points (s)
traces = np.random.randn(50, 1000)     # 50 neurons × 1000 time points
data = np.vstack([time, traces])       # shape = (51, 1000)

# Preprocess (shuffle & thresholds)
shuffled = prep.generate_event_shuffle(data)
threshold = prep.generate_threshold(shuffled)

# Build graph
cg = CaGraph(data=data)

# Visualize
viz.interactive_network(cg.graph)
~~~


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Veronica Porubksy - verosky@uw.edu

Project Link: [https://github.com/vporubsky/CaGraph](https://github.com/vporubsky/CaGraph)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [Sauro Lab at the University of Washington](https://sites.google.com/uw.edu/systems-biology-lab/home?authuser=1)
* [Bruchas Lab at the University of Washington](http://www.bruchaslab.org/)






<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/vporubsky/CaGraph.svg?style=for-the-badge
[contributors-url]: https://github.com/vporubsky/CaGraph/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/vporubsky/CaGraph.svg?style=for-the-badge
[forks-url]: https://github.com/vporubsky/CaGraph/network/members
[stars-shield]: https://img.shields.io/github/stars/vporubsky/CaGraph.svg?style=for-the-badge
[stars-url]: https://github.com/vporubsky/CaGraph/stargazers
[issues-shield]: https://img.shields.io/github/issues/vporubsky/CaGraph.svg?style=for-the-badge
[issues-url]: https://github.com/vporubsky/CaGraph/issues
[license-shield]: https://img.shields.io/github/license/vporubsky/CaGraph.svg?style=for-the-badge
[license-url]: https://github.com/vporubsky/CaGraph/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/vporubsky
[product-screenshot]: https://raw.githubusercontent.com/vporubsky/CaGraph/main/figures/figure_1.png
