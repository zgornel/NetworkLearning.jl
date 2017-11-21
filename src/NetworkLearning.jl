# Network learning

VERSION >= v"0.6" && __precompile__(true)

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

		# Functionality	
		fit, 
		predict, 
		predict!,
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

end
