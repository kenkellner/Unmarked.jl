"Structure containg design matrices and related info"
struct UmDesign
  params::Array
  coefs::Array
  inds::Array
  mats::Array
end

"""
Build design matrices from vector of formulas and matching 
vector of data frames
"""
function get_design(formulas::Array, cov_frames::Array)
  
  params = map(x -> string(x.lhs.sym), formulas)
  schemas = get_schemas(formulas, cov_frames)
  coefs = reduce(vcat, coefnames.(schemas)) 
  mats = get_mats(schemas, cov_frames) 
  inds = get_inds(mats)

  return UmDesign(params, coefs, inds, mats)

end

"Check that all formula variables are present in corresponding data frame"
function check_formula(formula::FormulaTerm, covariate_frame::DataFrame)
  nms = names(covariate_frame)
 
  #Variables in formula to array
  if eltype(formula.rhs) == Term
    vars = collect(map(x -> x.sym, formula.rhs))
  else
    vars = [formula.rhs.sym]
  end
   
  #Check if any are not in covariate frame
  not_in = map(x -> x ∉ nms, vars)
  
  #Error if necessary
  if any(not_in)
    miss_vars = map(x -> string(x), vars)[not_in]
    error(string("Variable(s) in formula not found in dataset: ", miss_vars))
  end

  return nothing

end

"Check input data frames for validity"
function check_data(dat::DataFrame)
  if any([any(ismissing.(col)) for  col = eachcol(dat)])
    error("Covariate DataFrames cannot contain missing values")
  end
  return nothing
end

"Get schema for each formula / data frame combination"
function get_schemas(formulas::Array, cov_frames::Array)
  
  out = []
  
  for i = 1:length(formulas)
    #Check if formula has variables not in covariate frame
    #Need to fix to address interactions
    #check_formula(formulas[i], cov_frames[i])
    #Check correspding data frame
    check_data(cov_frames[i])
    #Copy covariate frame and add dummy column of zeros for LHS of formula
    covs = deepcopy(cov_frames[i])
    covs[!,formulas[i].lhs.sym] = zeros(nrow(covs))

    #Buld and apply schema
    sch = schema(formulas[i], covs)
    asch = apply_schema(formulas[i], sch, StatisticalModel)

    #Add rhs of schema to output
    push!(out, asch.rhs)
  end

  return out

end

"Get array of design matrices from array of schemas"
function get_mats(schemas::Array, cov_frames::Array)
  
  mats = []

  for i in 1:length(schemas)
    push!(mats, modelcols(schemas[i], cov_frames[i]))
  end

  mats
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
