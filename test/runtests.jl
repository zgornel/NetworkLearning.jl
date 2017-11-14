using LearnBase, NetworkLearning

if (VERSION > v"0.7-")
	using Test
end

# Test Out-of-graph learning
include("t_networklearner_out_of_graph.jl")
Test.@testset "Network Learning (out-of-graph)" begin 
	t_networklearner_out_of_graph(); 
end

# Test In-graph learning
include("t_networklearner_in_graph.jl")
Test.@testset "Network Learning (in-graph)" begin 
	t_networklearner_in_graph(); 
end
