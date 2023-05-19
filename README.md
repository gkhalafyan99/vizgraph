# Repository README

## Overview

This repository contains a collection of Python scripts used for the "Visualization-Driven 
Graph Sampling Strategies" project. The datasets are derived from a variety of sources, representing different graph structures. The Python scripts include implementations of various sampling algorithms, metrics for quantitative analysis, and utility functions.

## Datasets

The following datasets have been used during the exepriments:

1. **Autonomous Systems AS-733**: The Autonomous Systems AS-733 dataset represents the structure of autonomous systems (AS), collected from BGP logs in 2001 by the Stanford Network Analysis Platform (SNAP). Details can be found at this [link](https://snap.stanford.edu/data/as-733.html).

2. **Condense Matter collaboration network**: The Condensed Matter collaboration network dataset is a collaboration network derived from arXiv's Condensed Matter category, where each node represents an author and each edge represents a co-authorship between two authors. Details can be found at this [link](https://snap.stanford.edu/data/ca-CondMat.html).

3. **CORA**: The Cora dataset comprises 2708 scientific publications classified into one of seven classes. The citation network is depicted by the graph where each node corresponds to a document and each edge signifies a citation. Details can be found at this [link](https://paperswithcode.com/dataset/cora).

4. **Facebook Large Page-Page Network**: The Facebook Large Page-Page Network dataset includes a large network of Facebook pages, where each node denotes a page, and each edge represents a 'like' between pages. Details can be found at this [link](https://snap.stanford.edu/data/facebook-large-page-page-network.html).

5. **Facebook**: The Facebook dataset includes a network of Facebook users, where nodes represent users, and edges represent friendships between users. Details can be found at this [link](https://snap.stanford.edu/data/ego-Facebook.html).

6. **General Relativity and Quantum Cosmology collaboration network**: The General Relativity and Quantum Cosmology collaboration network dataset contains a collaboration network from arXiv's General Relativity and Quantum Cosmology category. Details can be found at this [link](https://snap.stanford.edu/data/ca-GrQc.html).

7. **LastFMAsia**: The LastFMAsia dataset represents the social network of the LastFM music listening service, specifically for users in Asia. Details can be found at this [link](https://snap.stanford.edu/data/feather-lastfm-social.html).

8. **Wikivote**: The Wikivote dataset is a network of Wikipedia voting on promotions to adminship. A directed edge in the network means that a Wikipedia user voted on another user. Details can be found at this [link](https://snap.stanford.edu/data/wiki-Vote.html).


## Python Scripts

- **samplers.py**: This script contains code for various graph sampling algorithms, both established and new ones proposed in this project.
- **metrics.py**: This script contains functions for calculating metrics used for the quantitative analysis of the graphs.
- **utils.py**: This script contains utility classes and functions used for calculations and other support functions in the project.
