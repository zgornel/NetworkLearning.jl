# Tests for observation-based learning
function t_observation_networklearner()

########################################
# Test adjacency-related functionality #
########################################

# Define adjacency data
A = [							# adjacency matrix		
 0.0  0.0  1.0  1.0  0.0;
 0.0  0.0  1.0  0.0  0.0;
 1.0  1.0  0.0  1.0  1.0;
 1.0  0.0  1.0  0.0  0.0;
 0.0  0.0  1.0  0.0  0.0;
]

Ai = Int.(A)
G = Graph(Ai)

Gw = SimpleWeightedGraph(A)

rows,cols = findn(A)
data = [(i,j) for (i,j) in zip(rows,cols)]		# edges of A
n = 5							# number of vertices
f(n::Int, T=Int) = (data::Vector{Tuple{Int,Int}})->begin
	m = zeros(T,n,n)
	for t in data
		m[t[1],t[2]] = one(T)
	end
	return m
end

# Test functionality
Test.@test adjacency() isa EmptyAdjacency
Test.@test adjacency(nothing) isa EmptyAdjacency

A_Adj = adjacency(Ai)
Test.@test A_Adj isa MatrixAdjacency
Test.@test A_Adj.am == A

G_Adj = adjacency(G)
Test.@test G_Adj isa GraphAdjacency
Test.@test G_Adj.ag == G

Gw_Adj = adjacency(Gw)
Test.@test Gw_Adj isa GraphAdjacency
Test.@test Gw_Adj.ag == Gw


C_Adj = adjacency(f(n), data)
Test.@test C_Adj isa ComputableAdjacency
Test.@test C_Adj.f == f(n)
Test.@test C_Adj.data == data 

C2_Adj = adjacency((f(n), data))
Test.@test C2_Adj isa ComputableAdjacency
Test.@test C2_Adj.f == f(n)
Test.@test C2_Adj.data == data 

P_Adj = adjacency(f(n))
Test.@test P_Adj isa PartialAdjacency
Test.@test P_Adj.f == f(n)
Test.@test adjacency(P_Adj, data).am == A

A_Adj2 = adjacency(A_Adj)
Test.@test A_Adj2 == A_Adj

Test.@test adjacency(strip_adjacency(A_Adj),A).am == A_Adj.am
Test.@test adjacency(strip_adjacency(G_Adj),A).ag == G_Adj.ag
Test.@test adjacency(strip_adjacency(C_Adj),data).am == A_Adj.am
Test.@test adjacency(strip_adjacency(P_Adj),data).am == A_Adj.am

Test.@test adjacency_graph(A_Adj) == G
Test.@test adjacency_graph(G_Adj) == G
Test.@test adjacency_graph(Gw_Adj) == Gw
Test.@test adjacency_graph(C_Adj) == G

Test.@test adjacency_matrix(A_Adj) == Ai
Test.@test adjacency_matrix(G_Adj) == Ai
Test.@test adjacency_matrix(Gw_Adj) == A
Test.@test adjacency_matrix(C_Adj) == A

# Test update_adjacency
A = [0 1 0; 1 0 0; 0 0 0]
Adj_m = adjacency(A)
update_function_m!(X,x,y) = begin
	X[x,y] += 1
	X[y,x] += 1
	return X
end
f_update_m(x,y) = X->update_function_m!(X,x,y)
for i in 1:3
	update_adjacency!(Adj_m, f_update_m(1,3))
end
Test.@test adjacency_matrix(Adj_m) == [0 1 3; 1 0 0; 3 0  0]

Adj_g = adjacency(Graph(A))
f_update_g(x,y) = G->add_edge!(G,x,y)
update_adjacency!(Adj_g, f_update_g(1,3))
Test.@test Matrix(adjacency_matrix(Adj_g)) == [0 1 1; 1 0 0; 1 0  0]



######################################################
# Test fit/transform methods for relational learners #
######################################################
LEARNER = [SimpleRN,
	   WeightedRN,
	   BayesRN,
	   ClassDistributionRN]
N = 5							# number of observations
C = 2; 							# number of classes
A = [							# adjacency matrix		
 0.0  0.0  1.0  1.0  0.0;
 0.0  0.0  1.0  0.0  0.0;
 1.0  1.0  0.0  1.0  1.0;
 1.0  0.0  1.0  0.0  0.0;
 0.0  0.0  1.0  0.0  0.0;
]
Ad = adjacency(A); 

X = [							# local model estimates (2 classes)
 1.0  1.0  1.0  0.0  0.0;
 0.5  1.0  0.0  1.5  0.0
]

y = [1, 1, 1, 2, 2]					# labels

result = [
 [0.5  1.0  0.5  1.0  1.0; 				# validation data for SimpleRN
  0.5  0.0  0.5  0.0  0.0]
, 
 [0.5  1.0  0.5   1.0   1.0;				# validation data for WeightedRN 
  0.75 0.0  0.75  0.25  0.0]
, 		
 [1.60199  1.51083  1.60199  1.51083  1.51083;		# validation data for BayesRN
  1.14384  1.28768  1.14384  1.28768  1.28768]
,
 [0.300463  0.600925  0.300463  0.416667  0.600925; 	# validation data for ClassDistributionRN
  0.800391  0.125     0.800391  0.125     0.125]   
]

Xo = zeros(size(X));					# ouput (same size as X)
Xon = zeros(size(X));					# normalized ouput (same size as X)

tol = 1e-5
for li in 1:length(LEARNER)
	rl = fit(LEARNER[li], Ad, X, y; priors=ones(length(unique(y))),normalize=false)
	rln = fit(LEARNER[li], Ad, X, y; priors=ones(length(unique(y))),normalize=true)
        transform!(Xo, rl, Ad, X, y);
        transform!(Xon, rln, Ad, X, y);
	Test.@test all(abs.(Xo - result[li]) .<= tol);	# external validation
	Test.@test Xon ≈ (Xo./sum(Xo,1))		# normalization validation
end



###################################
# Tests for the utility functions #
###################################
cited =  [1,1,2,2,3,3,4,6,5,6]
citing = [2,3,1,5,2,6,7,1,2,3]
useidx = [1,2,6]

# 1 and 2 cite each other (edge weight of 2), 1 cites 6 one time (edge weight of 1)
Test.@test NetworkLearning.generate_partial_adjacency(cited,citing,useidx) == [0.0 2 1; 2 0 0; 1 0 0]
 


#########################################
# Test observation-based NetworkLearner #
#########################################
Ntrain = 100						# Number of training observations
Ntest = 10						# Number of testing observations					
inferences = [:ic, :rl]					# Collective inferences
rlearners = [:rn, :wrn, :bayesrn, :cdrn]		# Relational learners
nAdj = 2						# Number of adjacencies to generate	
X = rand(1,Ntrain); 					# Training data

nlmodel=[]
for tL in [:regression, :classification]		# Learning scenarios 
	# Initialize data            
	if tL == :regression
		ft=x->vec(x)
		y = vec(sin.(X)); 
		Xo = zeros(1,Ntest)
		
		# Train and test methods for local model 
		fl_train = (x)->mean(x[1]); 
		fl_exec=(m,x)->x.-m;

		# Train and test methods for relational model
		fr_train=(x)->sum(x[1],2);
		fr_exec=(m,x)->sum(x.-m,1)
	else 
		ft=x->vec(x[1,:])
		y = rand([1,2,3],Ntrain) # generate 3 classes 
		C = length(unique(y))
		Xo = zeros(C,Ntest)
		
		# Train and test methods for local model
		fl_train = (x)->zeros(C,1);
		fl_exec=(m,x)->abs.(x.-m);
		
		# Train and test methods for relational model
		fr_train=(x)->sum(x[1],2);
		fr_exec=(m,x)->rand(C,size(x,2))
	end

	amv = sparse.(Symmetric.([sprand(Float64, Ntrain,Ntrain, 0.5) for i in 1:nAdj]));
	adv = adjacency.(amv); 

	for infopt in inferences
		for rlopt in rlearners  
			Test.@test try
				# Train NetworkLearner
				nlmodel=fit(NetworkLearnerObs, X, y, 
				       adv, fl_train, fl_exec,fr_train,fr_exec;
				       learner=rlopt, 
				       inference=infopt,
				       use_local_data=rand(Bool),
				       f_targets=ft,
				       normalize=false, maxiter = 5
				)

				# Test NetworkLearner
				Xtest = rand(1,Ntest)

				# Add adjacency
				amv_t = sparse.(Symmetric.([sprand(Float64, Ntest,Ntest, 0.7) for i in 1:nAdj]));
				adv_t = adjacency.(amv_t); 
				add_adjacency!(nlmodel, adv_t)
				
				#Run NetworkLearner
				predict!(Xo, nlmodel, Xtest); # in-place
				predict(nlmodel, Xtest);      # creates output matrix	
				true
			catch
				false
			end
		end
	end
end



Test.@test try
	show(nlmodel)
	true
catch
	false
end


end
