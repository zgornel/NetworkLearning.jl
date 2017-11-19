# NetworkLearning

A Julia package for networking learning.

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) 
[![Build Status](https://travis-ci.org/zgornel/NetworkLearning.jl.svg?branch=master)](https://travis-ci.org/zgornel/NetworkLearning.jl) 
[![Coverage Status](https://coveralls.io/repos/github/zgornel/NetworkLearning.jl/badge.svg?branch=master)](https://coveralls.io/github/zgornel/NetworkLearning.jl?branch=master)

## Introduction

NetworkLearning implements a generic framework for network classification. It could in theory be used for other functionality such as regression and density estimation,
provided that appropriate methods for relational learning (i.e. relational variable generation) and collective inference are added. The framework is designed to make as little assumptions as possible on the elements involved in the process.  

Two scenarios for network learning can be distinguished:

- *Observation-based learning*, in which the network structure is pertinent to the observations and consequently, estimates (i.e. class probabilities) are associated to the observations; in the estimation process, relational structures can either make use the training data (in-graph learning) or not (out-of-graph learning). For example, in the case of document classifcation, an observation would correspond to a publication that has to be classified into an arbitrary category, given a representation of its local content-based information as well on the its relational information (links to other documents, citations etc.).  

- *Entity-based learning*, in which observations are pertinent to one or more abstract entities for which estimates are calculated. In entity-based network learning, observations can modify either the local or relational information of one or more entities.

So far, the package supports only observation-based learning.



## Features

- **Relational learners**
	- simple relational neighbour
	- weighted/probabilistic relational neighbour
	- naive bayes
	- class distribution

- **Collective inference**
	- relaxation labeling
	- collective classification

- **Adjacency strucures**
	- matrices
	- graphs
	- tuples containing functions and data from which adjacency matrices or graphs can be computed



## TO DO
	
- *entity-based learning* i.e. observations modify properties of abstract entities

- Gibbs sampling collective inference



## Documentation

The documentation is provided in Julia's native docsystem. 



## Installation

The package can be installed by running `Pkg.clone("https://github.com/zgornel/NetworkLearning.jl")` in the Julia REPL.



## License

This code has an MIT license and therefore it is free.



## References
[1] S.A. Macskassy, F. Provost "Classification in networked data: A toolkit and a univariate case study", Journal of Machine learning Research 8, 2007, 935-983

[2] P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Gallagher, T. Eliassi-Rad "Collective classification in network data", AI Magazine 29(3), 2008
