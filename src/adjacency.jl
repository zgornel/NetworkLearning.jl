#############
# Adjacency #
#############
abstract type AbstractAdjacency end

# Types
mutable struct MatrixAdjacency{T<:AbstractMatrix} <: AbstractAdjacency
	am::T
	function MatrixAdjacency(am::T) where T<:AbstractMatrix
		@assert issymmetric(am) "Adjacency matrix should be symmeric."
		new{T}(am)
	end
end

mutable struct GraphAdjacency{T<:AbstractGraph} <: AbstractAdjacency
	ag::T
end

mutable struct ComputableAdjacency{T,S} <:AbstractAdjacency
	f::T
	data::S
end

mutable struct PartialAdjacency{T<:Function} <: AbstractAdjacency
	f::T
end

mutable struct EmptyAdjacency <: AbstractAdjacency end


# Constructors
adjacency(a::T) where T<:AbstractAdjacency = a
adjacency(a::T, data) where T<:PartialAdjacency = adjacency(a.f(data))
adjacency(am::T) where T<:AbstractMatrix = MatrixAdjacency(am)
adjacency(ag::T) where T<:AbstractGraph = GraphAdjacency(ag)
adjacency(f::T, data::S) where {T,S} = ComputableAdjacency(f,data)
adjacency(t::Tuple{T,S}) where {T,S} = ComputableAdjacency(t[1],t[2])
adjacency(f::T) where T<:Function = PartialAdjacency(f)
adjacency(::Void) = EmptyAdjacency()
adjacency() = EmptyAdjacency()

	

# Show methods
Base.show(io::IO, a::MatrixAdjacency) = print(io, "Matrix adjacency, $(size(a.am,1)) obs")
Base.show(io::IO, a::GraphAdjacency) = print(io, "Graph adjacency, $(nv(a.ag)) obs")
Base.show(io::IO, a::ComputableAdjacency) = print(io, "Computable adjacency")
Base.show(io::IO, a::PartialAdjacency) = print(io, "Partial adjacency, not computable")
Base.show(io::IO, a::EmptyAdjacency) = print(io, "Empty adjacency, not computable")
Base.show(io::IO, va::T) where T<:AbstractVector{S} where S<:AbstractAdjacency = 
	print(io, "$(length(va))-element Vector{$S} ...")



# Methods that to strip the adjacency infomation (used in training of the NetworkLearner); the methods return a Partial Adjacency
strip_adjacency(a::MatrixAdjacency{T}) where T <:AbstractMatrix = adjacency(x->T(x))
strip_adjacency(a::GraphAdjacency{T}) where T<:AbstractGraph = adjacency(x->T(x))
strip_adjacency(a::ComputableAdjacency{T,S}) where {T,S} = adjacency(a.f)
strip_adjacency(a::PartialAdjacency) = a
strip_adjacency(a::EmptyAdjacency) = PartialAdjacency(x->adjacency(x))



# Functions to strip the adjacency infomation (used in training of the NetworkLearner); the methods return a Partial Adjacency
update_adjacency!(a::MatrixAdjacency{T}, f_update) where T <:AbstractMatrix = begin
	sam = size(a.am)
	f_update(a.am) # f_update should be a closure calling an in-place modifier i.e.  x->foo!(x, args...)
	@assert size(a.am) == sam "The dimensions of the adjacency matrix cannot change."
	return a
end

update_adjacency!(a::GraphAdjacency{T}, f_update) where T<:AbstractGraph = begin
	n = nv(a.ag)
	f_update(a.ag) # f_update should be a closure calling an in-place modifier i.e.  x->foo!(x, args...)
	@assert nv(a.ag) == n "The number of vertices in the adjacency graph cannot change."
	return a
end

update_adjacency!(a::ComputableAdjacency{T,S}, f_update) where {T,S} = error("Cannot update a computable adjacency.")
update_adjacency!(a::PartialAdjacency, args...) = error("Cannot update a partial adjacency.")
update_adjacency!(a::EmptyAdjacency, args...) = error("Cannot update an empty adjacency.")



# Return an adjacency graph from Adjacency types
adjacency_graph(a::MatrixAdjacency{T}) where T <:AbstractMatrix = adjacency_graph(a.am)
adjacency_graph(a::GraphAdjacency{T}) where T = adjacency_graph(a.ag)
adjacency_graph(a::ComputableAdjacency{T,S}) where {T,S} = adjacency_graph(a.f, a.data)
adjacency_graph(a::PartialAdjacency) = error("Insufficient information to obtain an adjacency graph.")
adjacency_graph(a::EmptyAdjacency) = error("Insufficient information to obtain an adjacency graph.")
adjacency_graph(am::T) where T <:AbstractMatrix = Graph(am)
adjacency_graph(am::T) where T <:AbstractMatrix{Float64} = SimpleWeightedGraph(am)
adjacency_graph(ag::T) where T <:AbstractGraph = ag
adjacency_graph(f::T, data::S) where {T,S} = adjacency_graph(f(data))


# Return an adjacency matrix from Adjacency types
adjacency_matrix(a::MatrixAdjacency{T}) where T <:AbstractMatrix = adjacency_matrix(a.am)
adjacency_matrix(a::GraphAdjacency{T}) where T = adjacency_matrix(a.ag)
adjacency_matrix(a::ComputableAdjacency{T,S}) where {T,S} = adjacency_matrix(a.f, a.data)
adjacency_matrix(a::PartialAdjacency) = error("Insufficient information to obtain an adjacency matrix.")
adjacency_matrix(a::EmptyAdjacency) = error("Insufficient information to obtain an adjacency matrix.")
adjacency_matrix(am::T) where T <:AbstractMatrix = am
adjacency_matrix(ag::T) where T <:AbstractGraph = sparse(ag)
adjacency_matrix(ag::T) where T <:AbstractSimpleWeightedGraph = weights(ag)
adjacency_matrix(f::T, data::S) where {T,S} = adjacency_matrix(f(data))
