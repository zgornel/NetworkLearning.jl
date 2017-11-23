# Network learning
VERSION >= v"0.6" && __precompile__(true)

"""
NetworkLearning implements a generic framework for network classification. It could in theory be used for other functionality such as regression and density estimation,
provided that appropriate methods for relational learning (i.e. relational variable generation) and collective inference are added. The framework is designed to make as little assumptions as possible on the elements involved in the process.  

# References
[1] S.A. Macskassy, F. Provost "Classification in networked data: A toolkit and a univariate case study", Journal of Machine learning Research 8, 2007, 935-983

[2] P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Gallagher, T. Eliassi-Rad "Collective classification in network data", AI Magazine 29(3), 2008
"""
module NetworkLearning
	
	using LearnBase, MLDataPattern, MLLabelUtils, LightGraphs, SimpleWeightedGraphs, Distances

	if VERSION >= v"0.7.0-"
		using DelimitedFiles:readdlm
	end
	
	# Verbosity level, used for debugging: 
	# 0 - off, 1 - minimal, 2 - normal (all messages) 
	global const VERBOSE = 0 

	# Exports
	export  # Adjacencies
		AbstractAdjacency, 
		MatrixAdjacency,
		GraphAdjacency,
		ComputableAdjacency,
		PartialAdjacency,
		EmptyAdjacency,
		
		# Relational learners
		AbstractRelationalLearner,
		SimpleRN, 
		WeightedRN,
		BayesRN,
		ClassDistributionRN,

		# Collective inference
		AbstractCollectiveInferer, 
		RelaxationLabelingInferer,
		IterativeClassificationInferer,
		GibbsSamplingInferer,
		
		# Network learners
	 	AbstractNetworkLearner,
		NetworkLearnerObs,
		NetworkLearnerEnt,

		# Functionality	
		fit, 
		predict, 
		predict!,
		infer!,
		transform, 
		transform!, 
		adjacency,
		add_adjacency!, 
		update_adjacency!,
		strip_adjacency,
		adjacency_matrix,
		adjacency_graph,
		@print_verbose
	
	abstract type AbstractNetworkLearner end
	
	include("utils.jl")									# Small utility functions
	include("adjacency.jl") 								# Adjacency-related structures 
	include("rlearners.jl")									# Relational learners
	include("cinference.jl")								# Collective inference algorithms		
	include("obslearning.jl")								# Observation-based learning
	include("entlearning.jl")								# Entity-based learning

end
