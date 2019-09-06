using DataFrames: DataFrame
using StatsModels: ModelMatrix, ModelFrame, @formula, coefnames

"Structure containg design matrices and related info"
struct design_mats
  params::Array
  coefs::Array
  inds::Array
  mats::Array
end

"""
Build design matrices from vector of formulas and matching 
vector of data frames
"""
function get_design(formulas, covariate_frames)
  
  params = map(x -> string(x.lhs.sym), formulas)
  frames = get_frames(formulas, covariate_frames)
  coefs = coefnames.(frames) 
  mats = map(x -> ModelMatrix(x).m, frames)
  inds = get_inds(mats)

  return design_mats(params, coefs, inds, mats)

end

"Check that all formula variables are present in corresponding data frame"
function check_formula(formula, covariate_frame)
  nms = names(covariate_frame)
 
  #Variables in formula to array
  if(eltype(formula.rhs) == Term)
    vars = collect(map(x -> x.sym, formula.rhs))
  else
    vars = [formula.rhs.sym]
  end
   
  #Check if any are not in covariate frame
  not_in = map(x -> x ∉ nms, vars)
  
  #Error if necessary
  if(any(not_in))
    miss_vars = map(x -> string(x), vars)[not_in]
    error(string("Variable(s) in formula not found in dataset: ", miss_vars))
  end

  return nothing

end

"Get model frames for each formula / data frame combination"
function get_frames(formulas, covariate_frames)
  
  frames = []
  
  for i = 1:length(formulas)
    #Check if formula has variables not in covariate frame
    check_formula(formulas[i], covariate_frames[i])
    #Copy covariate frame and add dummy column of zeros for LHS of formula
    covs = deepcopy(covariate_frames[i])
    covs[!,formulas[i].lhs.sym] = zeros(nrow(covs))
    #Add design frame to output vector
    push!(frames, ModelFrame(formulas[i], covs))
  end

  return frames

end

"Get parameter index ranges for each design matrix"
function get_inds(design_mats)

  np = map(x -> size(x)[2], design_mats)

  out = Array{UnitRange{Int}}(undef, length(np))

  index = 1
  for i in 1:length(np)
    out[i] = index:(index + np[i] - 1)
    index += np[i]
  end
    
  return out

end
