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


## Usage example

```julia
import DecisionTree
using NetworkLearning, MLDataPattern, MLLabelUtils, LossFunctions

# Download the CORA dataset, and return data and citing/cited papers indices (relative to order in X,y)
(X,y), cited_papers, citing_papers = NetworkLearning.grab_cora_data()
yᵤ = sort(unique(y))
C = length(yᵤ)

# Split data
idx = collect(1:nobs(X));
shuffle!(idx)
p = 0.5
train_idx,test_idx = splitobs(idx,p)
Xtr = X[:,train_idx]
ytr = y[train_idx]
Xts = X[:,test_idx]

# Build adjacency matrices
train_am = NetworkLearning.generate_partial_adjacency(cited_papers, citing_papers, train_idx);

# Construct necessary arguments for training the network learner
fl_train = (X)->  DecisionTree.build_tree(X[2],X[1]')
fl_exec(C) = (m,X)-> DecisionTree.apply_tree_proba(m, X', collect(1:C))'

fr_train = (X)->  DecisionTree.build_tree(X[2],X[1]')
fr_exec(C) = (m,X)-> DecisionTree.apply_tree_proba(m, X', collect(1:C))'

AV = [adjacency(train_am)]

# Train
info("Training ...")
nlmodel = NetworkLearning.fit(NetworkLearnerObs, 
	      Xtr,
	      ytr,
	      AV,
	      fl_train, fl_exec(C),
	      fr_train, fr_exec(C);
	      learner = :wrn,
	      inference = :ic,
	      use_local_data = false, # use only relational variables
	      f_targets = x->targets(indmax,x),
	      normalize = true,
	      maxiter = 100,
	      α = 0.95
	  )



#########################
# Out-of-Graph learning #
#########################

# Add adjacency pertinent to the test data
test_am = NetworkLearning.generate_partial_adjacency(cited_papers, citing_papers, test_idx);
add_adjacency!(nlmodel, [test_am])

# Make predictions
info("Predicting (out-of-graph) ...")
ŷts = predict(nlmodel, Xts)

# Squared loss
yts = convertlabel(LabelEnc.OneOfK(C), y[test_idx], yᵤ)
println("\tL2 loss (out-of-graph):", value(L2DistLoss(), yts, ŷts, AvgMode.Mean()))
println("\tAverage error (out-of-graph):", mean(targets(indmax,yts).!=targets(indmax,ŷts)))



#####################
# In-Graph learning #
#####################

# Initialize output structure
Xo = zeros(C,nobs(X))
Xo[:,train_idx] = DecisionTree.apply_tree_proba(DecisionTree.build_tree(ytr,Xtr'),Xtr',yᵤ)'

# Add adjacency pertinent to the test data
test_am = NetworkLearning.generate_partial_adjacency(cited_papers, citing_papers, collect(1:nobs(X)));
add_adjacency!(nlmodel, [test_am])

# Make predictions
info("Predicting (in-graph) ...")
update_mask = trues(nobs(X));
update_mask[train_idx] = false # estimates for training samples will not be used
predict!(Xo, nlmodel, X, update_mask)
ŷts = Xo[:,test_idx]

# Squared loss
yts = convertlabel(LabelEnc.OneOfK(C), y[test_idx], yᵤ)
println("\tL2 loss (in-graph):", value(L2DistLoss(), yts, ŷts, AvgMode.Mean()))
println("\tAverage error (in-graph):", mean(targets(indmax,yts).!=targets(indmax,ŷts)))
```

The output of the above code is:
```julia
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  163k  100  163k    0     0   163k      0  0:00:01  0:00:01 --:--:--  101k
cora/
cora/README
cora/cora.content
cora/cora.cites
INFO: Training ...
INFO: Predicting (out-of-graph) ...
	L2 loss (out-of-graph):0.11059675847236224
	Average error (out-of-graph):0.45790251107828656
INFO: Predicting (in-graph) ...
	L2 loss (in-graph):0.04396139067718938
	Average error (in-graph):0.17134416543574593
```



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
