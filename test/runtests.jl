using NetworkLearning

if (VERSION > v"0.7-")
	using Test
end

# Test observation-based learning
include("t_observation_networklearner.jl")
Test.@testset "Network Learning (observation-based)" begin 
	t_observation_networklearner(); 
end
