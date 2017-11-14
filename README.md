# NetworkLearning

A Julia package for networking learning.

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) 
[![Build Status](https://travis-ci.org/zgornel/NetworkLearning.jl.svg?branch=master)](https://travis-ci.org/zgornel/NetworkLearning.jl) 
[![Coverage Status](https://coveralls.io/repos/github/zgornel/NetworkLearning.jl/badge.svg?branch=master)](https://coveralls.io/github/zgornel/NetworkLearning.jl?branch=master)

## Introduction

NetworkLearning implements a generic framework for network classification. It could in theory be used for other functionality such as regression and density estimation,
provided that appropriate methods for relational learning (i.e. relational variable generation) and collective inference are added. 
The framework is designed to make as little assumptions as possible on the elements involved in the process.  

## Features

For now, only out-of-graph learning is supported i.e the training data network structure is not available in the evaluation of test data; a new network structure 
has to be computed for the tested data - this implies that out-of-graph learning has to be performed in batches in order to be able to calculate the features relating
to the neighbours of each observation.

- **Types of learning**
	- out-of-graph learning
	- **(TODO)** in-graph learning

- **Relational learners**
	- simple relational neighbour
	- weighted/probabilistic relational neighbour
	- naive bayes
	- class distribution

- **Collective inference**
	- relaxation labeling
	- collective classification
	- **(TODO)** gibbs sampling
 
## Documentation

The documentation is provided in Julia's native docsystem. 



## Installation

The package can be installed by running `Pkg.clone("https://github.com/zgornel/NetworkLearning.jl")` in the Julia REPL.



## License

This code has an MIT license and therefore it is free.



## References
[1] S.A. Macskassy, F. Provost "Classification in networked data: A toolkit and a univariate case study", Journal of Machine learning Research 8, 2007, 935-983

[2] P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Gallagher, T. Eliassi-Rad "Collective classification in network data", AI Magazine 29(3), 2008
