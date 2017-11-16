# Function that calculates the number of  relational variables / each adjacency structure
get_size_out(y::AbstractVector{T}) where T<:Float64 = 1			# regression case
get_size_out(y::AbstractVector{T}) where T = length(unique(y))::Int	# classification case
get_size_out(y::AbstractArray) = error("Only vectors supported as targets in relational learning.")



# Function that calculates the priors of the dataset
getpriors(y::AbstractVector{T}) where T<:Float64 = [1.0]	
getpriors(y::AbstractVector{T}) where T = [sum(yi.==y)/length(y) for yi in sort(unique(y))]
getpriors(y::AbstractArray) = error("Only vectors supported as targets in relational learning.")



encode_targets(labels::T where T<:AbstractVector{S}) where S = begin
	ulabels::Vector{S} = sort(unique(labels))
	enc = LabelEnc.NativeLabels(ulabels)
	return (enc, label2ind.(labels,enc))
end

encode_targets(labels::T where T<:AbstractVector{S}) where S<:AbstractFloat = begin
	return (nothing, labels)
end

encode_targets(labels::T where T<:AbstractArray{S}) where S = begin
	error("Targets must be in vector form, other arrays not supported.")
end



read_citation_data(content_file::String, cites_file::String) = begin
              
	# Read files
	content = readdlm(content_file,'\t')
	cites = readdlm(cites_file,'\t')
		      
	# Construct datasets
	labels = content[:,end]
	paper_ids = Int.(content[:,1])
	data = Float64.(content[:,2:end-1]')
	content_data = (data, labels)

	# Construct citing/cited paper indices
	cited_papers = indexin(Int.(cites[:,1]), paper_ids)
	citing_papers = indexin(Int.(cites[:,2]), paper_ids)

	return content_data, cited_papers, citing_papers

end



grab_cora_data(tmpdir::String="/tmp") = begin
	
	DATA_FILE = download("https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz")
	run(`tar zxvf $(DATA_FILE) -C $tmpdir`)
	
	cora_data = read_citation_data("$tmpdir/cora/cora.content","$tmpdir/cora/cora.cites")
	
	run(`rm -rf $tmpdir/cora`)
	run(`rm $DATA_FILE`)
	
	return cora_data
end
