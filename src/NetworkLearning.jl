# Network learning
module NetworkLearning
	
	using LearnBase, MLDataPattern, MLLabelUtils, LightGraphs, SimpleWeightedGraphs, Distances
	
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
		NetworkLearnerOutOfGraph,
		NetworkLearnerInGraph,	

		# Functionality	
		fit, 
		transform, 
		transform!, 
		adjacency,
		add_adjacency!, 
		strip_adjacency,
		adjacency_matrix,
		adjacency_graph
	
	abstract type AbstractNetworkLearner end
	
	include("adjacency.jl") 								# Adjacency-related structures 
	include("rlearners.jl")									# Relational learners
	include("cinference.jl")								# Collective inference algorithms		
	include("utils.jl")									# Small utility functions
	include("outlearning.jl")								# Out-of-graph learning
	include("inlearning.jl")								# In-graph learning

end


