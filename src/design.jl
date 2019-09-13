"Structure containg design matrix and related info for each submodel"
mutable struct UmDesign
  name::Symbol
  formula::FormulaTerm
  link::Link
  data::DataFrame
  coefnames::Union{Array{String}, Nothing}
  mat::Union{Array{Float64,2}, Nothing}
  idx::Union{UnitRange{Int64}, Nothing}
end

"Check input data frames for validity"
function check_data(dat::DataFrame)
  if any([any(ismissing.(col)) for  col = eachcol(dat)])
    error("Covariate DataFrames cannot contain missing values")
  end
  return nothing
end

"Add dummy column for response variable to data"
add_resp = function(x::DataFrame, f::FormulaTerm)
  lhs = f.lhs.sym
  if lhs in names(x) return x end
  dc = deepcopy(x)
  dc[!,lhs] = zeros(nrow(dc))
  return dc
end

"Add design matrix"
function add_dm!(x::UmDesign)
  dat = add_resp(x.data, x.formula)
  sch = schema(x.formula, dat)
  asch = apply_schema(x.formula, sch, StatisticalModel)  
  x.coefnames = coefnames(asch.rhs)
  x.mat = modelcols(asch.rhs, dat)
end

"Outer constructor for UmDesign objects"
function UmDesign(name::Symbol, formula::FormulaTerm, link::Link,
                  data::DataFrame)
  check_data(data)
  out = UmDesign(name, formula, link, data, nothing, nothing, nothing)
  add_dm!(out)
  out
end

"Add indexes to map combined coefs back to submodels"
function add_idx!(dm::Array{UmDesign})

  idx = 1
  for i = 1:length(dm)
    np = length(dm[i].coefnames)
    dm[i].idx = idx:(idx+np-1)
    idx += np
  end

end

"Transform linear predictor back to response scale"
function transform(dm::UmDesign, coefs::Array)
  lp = dm.mat * coefs[dm.idx]
  invlink(lp, dm.link)
end
