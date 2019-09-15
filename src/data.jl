"Basic structure for Unmarked data objects"
struct UmData
  y::Array{Union{Int,Missing}}
  site_covs::DataFrame
  obs_covs::DataFrame
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
