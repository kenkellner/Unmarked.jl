#Basic structure for Unmarked data objects
struct UmData
  y::Array{Union{Int,Missing}}
  site_covs::DataFrame
  obs_covs::DataFrame
  UmData(y,site_covs,obs_covs) = new(y, check_data(site_covs, size(y)[1]), 
                                     check_data(obs_covs, prod(size(y))))
end

#Check input data frames for validity
function check_data(dat::Union{DataFrame,Nothing}, exp_rows::Int) 
  if isnothing(dat) return DataFrame(_dummy=ones(exp_rows)) end  
  if any([any(ismissing.(col)) for col = eachcol(dat)])
    error("Covariate DataFrames cannot contain missing values")
  end
  if nrow(dat) != exp_rows
    error("Supplied DataFrame has incorrect number of rows")
  end  
  return dat
end

"Construct Unmarked data objects"
function UmData(y::Union{Array{Int}, Array{Union{Int,Missing}}}; 
                site_covs::Union{DataFrame,Nothing}=nothing,
                obs_covs::Union{DataFrame,Nothing}=nothing)
  N,J = size(y)
  return UmData(y, check_data(site_covs, N), check_data(obs_covs, N*J))
end

#Subset a UmData object
function _subset_obs(data::UmData, i::Union{Array{Int},UnitRange})
  N,J = size(data.y)
  site_idx = repeat(1:N, inner=J)
  keep = in.(site_idx, (i,))
  return keep
end

function Base.getindex(data::UmData, i::Union{Int,Array{Int},UnitRange})
  if typeof(i) == Int i = i:i end
  new_sc = data.site_covs[i,:]
  new_y = data.y[i,:]
  new_oc = data.obs_covs[_subset_obs(data, i),:]
  UmData(new_y,new_sc,new_oc)
end
