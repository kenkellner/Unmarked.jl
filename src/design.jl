#Structure containg design matrix and related info for each submodel"
mutable struct UmDesign
  name::Symbol
  formula::FormulaTerm
  link::Link
  data::DataFrame
  coefnames::Array{String}
  mat::Array{Float64,2}
  idx::Union{UnitRange{Int64}, Nothing}
end

#Check formulas for validity
function check_formula(formula::FormulaTerm, data::DataFrame)
  vars = varnames(formula).rhs
  if isnothing(vars) return nothing end
  covs = names(data)
  for x in vars
    if x ∉ covs error(string(x," not found in data")) end
  end
  return nothing
end

#Add dummy column for response variable to data
add_resp = function(x::DataFrame, f::FormulaTerm)
  lhs = f.lhs.sym
  if lhs in names(x) return x end
  dc = deepcopy(x)
  dc[!,lhs] = zeros(nrow(dc))
  return dc
end

#Build and apply schema
function get_schema(formula::FormulaTerm, data::DataFrame) 
  check_formula(formula, data)
  dat = add_resp(data, formula)
  sch = schema(formula, dat)
  apply_schema(formula, sch, StatisticalModel)
end

#Outer constructor for UmDesign objects"
function UmDesign(name::Symbol, formula::FormulaTerm, link::Link,
                  data::DataFrame)

  asch = get_schema(formula, data)
  cn = coefnames(asch.rhs)
  cn = cn isa String ? [cn] : cn
  mat = modelcols(asch.rhs, data)
  UmDesign(name, formula, link, data, cn, mat, nothing)
end

#Add indexes to map combined coefs back to submodels"
function add_idx!(dm::Array{UmDesign})
  idx = 1
  for x in dm
    np = length(x.coefnames)
    x.idx = idx:(idx+np-1)
    idx += np
  end
  return idx - 1
end

#Transform linear predictor back to response scale"
function transform(dm::UmDesign, coefs::Array)
  lp = dm.mat * coefs[dm.idx]
  invlink(lp, dm.link)
end
