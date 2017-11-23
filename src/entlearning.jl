#########################
# Entity-based learning #
#########################

# The state of the network learner: it is defined by estimates for all nodes
# and a mask that specifies which of the estimates are to be update (in case the training
# data is used as well


mutable struct NetworkLearnerState{T<:AbstractArray}
	ê::T			# estimates
	update::BitVector	# which estimates to update

	function NetworkLearnerState(ŷ::T, update::BitVector) where T<:AbstractArray
		@assert  nobs(ŷ) == nobs(update) "Number of NetworkLearner estimates should be equal to the length of the update mask."
		new{T}(ŷ, update)
	end
end

Base.show(io::IO, m::NetworkLearnerState) = print(io, "NetworkLearner state: $(sum(m.update))/$(nobs(m.update)) entities can be updated") 



mutable struct NetworkLearnerEnt{S,V,
				    NS<:NetworkLearnerState,
				    R<:Vector{<:AbstractRelationalLearner},
				    C<:AbstractCollectiveInferer,
				    A<:Vector{<:AbstractAdjacency}} <: AbstractNetworkLearner 			 
	state::NS										# state 
	Mr::S											# relational model
	fr_exec::V										# relational model execution function
	RL::R											# relational learner
	Ci::C											# collective inferer	
	Adj::A											# adjacency information
	size_out::Int										# expected output dimensionality
end



# Printers
Base.show(io::IO, m::NetworkLearnerEnt) = begin 
	println(io,"NetworkLearner, $(m.size_out) estimates, entity-based")
	println(io,"`- state: $(sum(m.state.update))/$(nobs(m.state.update)) entities can be updated"); 
	print(io,"`- relational model: "); println(io, m.Mr)
	print(io,"`- relational learners: "); println(io, m.RL)
	print(io,"`- collective inferer: "); println(io, m.Ci)
	print(io,"`- adjacency: "); println(io, m.Adj)	
end



####################
# Training methods #
####################
function fit(::Type{NetworkLearnerEnt}, Xo::AbstractMatrix, update::BitVector, Adj::A where A<:Vector{<:AbstractAdjacency}, 
	     	fr_train, fr_exec; 
		priors::Vector{Float64}=1/size(Xo,1).*ones(size(Xo,1)), learner::Symbol=:wvrn, inference::Symbol=:rl, 
		normalize::Bool=true, f_targets::Function=x->targets(indmax,x), 
		tol::Float64=1e-6, κ::Float64=1.0, α::Float64=0.99, maxiter::Int=100, bratio::Float64=0.1) 

	# Parse, transform input arguments
	κ = clamp(κ, 1e-6, 1.0)
	α = clamp(α, 1e-6, 1.0-1e-6)
	tol = clamp(tol, 0.0, Inf)
	maxiter = ifelse(maxiter<=0, 1, maxiter)
	bratio = clamp(bratio, 1e-6, 1.0-1e-6)

	@assert all((priors.>=0.0) .& (priors .<=1.0)) "All priors have to be between 0.0 and 1.0."
	
	# Parse relational learner argument and generate relational learner type
	if learner == :rn
		Rl = SimpleRN
	elseif learner == :wrn
		Rl = WeightedRN
	elseif learner == :cdrn 
		Rl = ClassDistributionRN
	elseif learner == :bayesrn
		Rl = BayesRN
	else
		warn("Unknown relational learner. Defaulting to :wrn.")
		Rl = WeightedRN
	end

	# Parse collective inference argument and generate collective inference objects
	if inference == :rl
		Ci = RelaxationLabelingInferer(maxiter, tol, f_targets, κ, α)
	elseif inference == :ic
		Ci = IterativeClassificationInferer(maxiter, tol, f_targets)
	elseif inference == :gs
		Ci = GibbsSamplingInferer(maxiter, tol, f_targets, ceil(Int, maxiter*bratio))
	else
		warn("Unknown collective inferer. Defaulting to :rl.")
		Ci = RelaxationLabelingInferer(maxiter, tol, f_targets, κ, α)
	end
	
	fit(NetworkLearnerEnt, Xo, update, Adj, Rl, Ci, fr_train, fr_exec; priors=priors, normalize=normalize)
end



function fit(::Type{NetworkLearnerEnt}, Xo::T, update::BitVector, Adj::A, Rl::R, Ci::C, fr_train::U, fr_exec::U2; 
		priors::Vector{Float64}=1/size(Xo,1).*ones(size(Xo,1)), normalize::Bool=true, use_local_data::Bool=true) where {
			T<:AbstractMatrix, 
			A<:Vector{<:AbstractAdjacency}, 
			R<:Type{<:AbstractRelationalLearner}, 
			C<:AbstractCollectiveInferer, 
			U, U2
		}
	 
	# Step 0: pre-process input arguments and retrieve sizes
	n = nobs(Xo)										# number of entities
	p = size(Xo,1)										# number of estimates/entity
	m = length(Adj) * p									# number of relational variables

	@assert p == length(priors) "Found $p entities/estimate, the priors indicate $(length(priors))."
	
	# Step 1: Get relational variables by training and executing the relational learner 
	@print_verbose 2 "Calculating relational variables ..."
	mₜ = .!update										# training mask (entities that are not updated
	Xoₜ = Xo[:,mₜ]										#    are considered training or high/certainty samples)
	nₜ = sum(mₜ)										# number of training observations
	yₜ = Ci.tf(Xoₜ)										# get targets
	Adjₜ = [adjacency(adjacency_matrix(Aᵢ)[mₜ,mₜ]) for Aᵢ in Adj]
	RL = [fit(Rl, Aᵢₜ, Xoₜ, yₜ; priors=priors, normalize=normalize) for Aᵢₜ in Adjₜ]	# Train relational learners				

	# Pre-allocate relational variables array	
	Xr = zeros(m,n)
	Xrₜ = Xr[:,mₜ]
	Xrᵢ = zeros(p,nₜ)									# Initialize temporary storage	
	for (i,(RLᵢ,Aᵢₜ)) in enumerate(zip(RL,Adjₜ))		
		
		# Apply relational learner
		transform!(Xrᵢ, RLᵢ, Aᵢₜ, Xoₜ, yₜ)

		# Update relational data output		
		Xrₜ[(i-1)*p+1 : i*p, :] = Xrᵢ									
	end
	

	# Step 2 : train relational model 
	@print_verbose 2 "Training relational model ..."
	Mr = fr_train((Xrₜ,yₜ))

	# Step 3: Apply collective inference
	@print_verbose 2 "Collective inference ..."
	transform!(Xo, Ci, Mr, fr_exec, RL, Adj, 0, Xr, update)	
	
	# Step 3: return network learner 
	return NetworkLearnerEnt(NetworkLearnerState(Xo,update), Mr, fr_exec, RL, Ci, Adj, p)
end



# Function that calls collective inference using the information in contained in the 
# entity-based network learner
function infer!(model::T) where T<:NetworkLearnerEnt
	p = size(model.state.ê,1)								# number of estimates/entity
	m = length(model.Adj) * p								# number of relational variables
	Xr = zeros(m,nobs(model.state.ê))

	transform!(model.state.ê, model.Ci, model.Mr, model.fr_exec, model.RL, model.Adj, 0, Xr, model.state.update) 
end
