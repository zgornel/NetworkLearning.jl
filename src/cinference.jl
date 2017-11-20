# Types
abstract type AbstractCollectiveInferer end

struct RelaxationLabelingInferer <: AbstractCollectiveInferer 
	maxiter::Int
	tol::Float64
	tf::Function
	κ::Float64	
	α::Float64
end

struct IterativeClassificationInferer <: AbstractCollectiveInferer 
	maxiter::Int
	tol::Float64
	tf::Function
end

struct GibbsSamplingInferer <: AbstractCollectiveInferer 
	maxiter::Int
	tol::Float64
	tf::Function
	burniter::Int
end



# Show methods
Base.show(io::IO, ci::RelaxationLabelingInferer) = print(io, "Relaxation labeling, maxiter=$(ci.maxiter), tol=$(ci.tol), κ=$(ci.κ), α=$(ci.α)")
Base.show(io::IO, ci::IterativeClassificationInferer) = print(io, "Iterative classification, maxiter=$(ci.maxiter), tol=$(ci.tol)")
Base.show(io::IO, ci::GibbsSamplingInferer) = print(io, "Gibbs sampling, maxiter=$(ci.maxiter), tol=$(ci.tol), burniter=$(ci.burniter)")
Base.show(io::IO, vci::T) where T<:AbstractVector{S} where S<:AbstractCollectiveInferer = 
	print(io, "$(length(vci))-element Vector{$S} ...")



# Transform methods
function transform!(Xo::T, Ci::RelaxationLabelingInferer, Mr::M, fr_exec::E, RL::R, Adj::A, offset::Int, Xr::S, 
		    update::BitVector=trues(nobs(Xo))) where {
		M, E, 
		T<:AbstractMatrix, R<:Vector{<:AbstractRelationalLearner}, 
		A<:Vector{<:AbstractAdjacency}, S<:AbstractMatrix}
	
	# Initializations
	n = nobs(Xo)										# number of observations
	κ = Ci.κ										# a constant between 0 and 1
	α = Ci.α										# decay
	β = κ											# weight of current iteration estimates
	maxiter = Ci.maxiter									# maximum number of iterations
	tol = Ci.tol										# maximum error 
	f_targets = Ci.tf									# function used to obtain targets
	size_out = size(Xo,1)									# ouput size (corresponds to the number of classes)
	Xl = copy(Xo)										# local estimates
	ŷₗ = f_targets(Xo)									# Obtain first the labels corresponding to the local model
	ŷ = ŷₗ											#   and initialize the current estimates
	ŷₒ = similar(ŷ)										#   and the 'previous' iteration estimates
	AV = adjacency_matrix.(Adj)								# Pre-calculate adjacency matrices
	Xrᵢ = zeros(size_out,n)									# Initialize temporary storage	

	# Iterate
	for it in 1:maxiter
		β = β * α									# Update learning rate
		copy!(ŷₒ, ŷ);									# Update 'previous iteration' estimates 
		
		# Obtain relational dataset for the current iteration
		@inbounds for (i,(RLᵢ,Aᵢ)) in enumerate(zip(RL,AV))		
		
			# Apply relational learner
			transform!(Xrᵢ, RLᵢ, Aᵢ, Xo, ŷ)

			# Update relational data output
			Xr[offset+(i-1)*size_out+1 : offset+i*size_out,:] = Xrᵢ
		end
		
		# Update estimates
		Xo[:,update] = β.*fr_exec(Mr, Xr[:,update]) + (1.0-β).*Xo[:,update] 
		ŷ = f_targets(Xo)

		# Convergence check
		if isequal(ŷ,ŷₒ) || mean(abs.(ŷ-ŷₒ))<=tol
			@print_verbose 1 "Convergence reached at iteration $it."
			break
		else
			@print_verbose 2 "\tIteration $it: $(sum(ŷ.!= ŷₒ)) estimates changed"
	   	end
		
		# Replace non-converging estimates with local estimates
		if (it == maxiter) && (maxiter != 1)
			_nc = ŷ.!=ŷₒ 		# positions of non-converging estimates
			datasubset(Xo, _nc)[:] = datasubset(Xl, _nc)[:]
			@print_verbose 1 "Maximum level of iterations reached, $(sum(_nc)) estimates did not converge."
		end
	end
	
	return Xo
end

function transform!(Xo::T, Ci::IterativeClassificationInferer, Mr::M, fr_exec::E, RL::R, Adj::A, offset::Int, Xr::S,
		    update::BitVector=trues(nobs(Xo))) where {
		M, E, 
		T<:AbstractMatrix, R<:Vector{<:AbstractRelationalLearner}, 
		A<:Vector{<:AbstractAdjacency}, S<:AbstractMatrix}
	
	# Initializations
	n = nobs(Xr)										# number of observations 
	ordering = [i:i for i in find(update)]							# observation estimation order 
	maxiter = Ci.maxiter									# maximum number of iterations
	tol = Ci.tol										# maximum error 
	f_targets = Ci.tf									# function used to obtain targets
	size_out = size(Xo,1)									# ouput size (corresponds to the number of classes)
	Xl = copy(Xo)										# local estimates	
	ŷₗ = f_targets(Xo)									# Obtain first the labels corresponding to the local model
	ŷ = ŷₗ											#   and initialize the current estimates
	ŷₒ = similar(ŷ)										#   and the 'previous' iteration estimates
	AV = adjacency_matrix.(Adj)								# Pre-calculate adjacency matrices
	Xrᵢⱼ = zeros(size_out,1)								# Initialize temporary storage	
	
	# Iterate
	for it in 1:maxiter	
		shuffle!(ordering)								# Randomize observation order
		copy!(ŷₒ, ŷ);									# Update 'previous iteration' estimates 
		
		# Loop over observations and obtain individual estimates
		for rⱼ in ordering		
			
			# Get data subsets pertinent to the current observation 
			Xrⱼ = datasubset(Xr, rⱼ)
			Xoⱼ = datasubset(Xo, rⱼ)
			ŷⱼ = datasubset(ŷ, rⱼ)

			# Obtain relational data for the current observation
			@inbounds for (i,(RLᵢ,Aᵢ)) in enumerate(zip(RL,AV))		

				# Apply relational learner
				transform!(Xrᵢⱼ, RLᵢ, Aᵢ[:,rⱼ], Xo, ŷ) 				# TODO: Find a better compromise for adjacency access; views - slow for sparse matrices
												#	slicing - increases the number of allocations.
				# Update relational data output for the current sample
				Xrⱼ[offset+(i-1)*size_out+1 : offset+i*size_out,:] = Xrᵢⱼ
			end
		
			# Update estimates
			Xoⱼ[:] = fr_exec(Mr, Xrⱼ) 
			ŷⱼ[:] = f_targets(Xoⱼ)
		end

		# Convergence check
		if isequal(ŷ,ŷₒ) || mean(abs.(ŷ-ŷₒ))<=tol
			@print_verbose 1 "Convergence reached at iteration $it."
			break
		else
			@print_verbose 2 "\tIteration $it: $(sum(ŷ.!= ŷₒ)) estimates changed"
	   	end

		# Replace non-converging estimates with local estimates
		if (it == maxiter) && (maxiter != 1)
			_nc = ŷ.!=ŷₒ 		# positions of non-converging estimates
			datasubset(Xo, _nc)[:] = datasubset(Xl, _nc)[:]
			@print_verbose 1 "Maximum level of iterations reached, $(sum(_nc)) estimates did not converge."
		end
	end
	
	return Xo
end

function transform!(Xo::T, Ci::GibbsSamplingInferer, Mr::M, fr_exec::E, RL::R, Adj::A, offset::Int, Xr::S) where {
		M, E, 
		T<:AbstractMatrix, R<:Vector{<:AbstractRelationalLearner}, 
		A<:Vector{<:AbstractAdjacency}, S<:AbstractMatrix}
	
	# warn("Gibbs sampling not implemented, returning input (local model) estimates.")
	return Xo
end
