"Basic structure for Unmarked data objects"
struct UmData
  y::Array{Union{Int,Missing}}
  site_covs::DataFrame
  obs_covs::DataFrame
end

