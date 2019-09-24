#Extract variable names from a formula before creating schema
function varnames(f::FormulaTerm)
  rhs = varnames(f.rhs)
  if rhs isa Symbol rhs = [rhs] end
  return (lhs=varnames(f.lhs), rhs=rhs)
end

function varnames(f::Term)
  return Symbol(f)
end

function varnames(f::StatsModels.TupleTerm)
  raw = map(x -> varnames(x), f)
  return get_unique(raw)
end

function varnames(f::ConstantTerm)
  return nothing
end

function varnames(f::InteractionTerm)
  return map(x -> varnames(x), f.terms) 
end

function get_unique(f::Tuple)
  out = Symbol[]
  fl(v) = for x in v
    if x isa Tuple fl(x) else push!(out, x) end
  end
  fl(f)
  return unique(out)
end

function combine_formulas(formulas::Union{Array,FormulaTerm}...)
  formulas = map(x -> x isa FormulaTerm ? [x] : x, formulas)
  form_combs = collect(Base.product(formulas...))
  return reshape(form_combs, length(form_combs))
end

"""

    allsub(f::FormulaTerm)

Takes a formula `f` and returns an array containing formulas for all
subsets of the formula terms.
"""
function allsub(f::FormulaTerm)
  if f.rhs isa Union{ConstantTerm,Term,Tuple{Term}} return [f] end
  combs = Tuple.(collect(combinations(Symbol.(f.rhs))))
  terms = map(x -> term.(x), combs)
  forms = map(x -> term(f.lhs.sym) ~ sum(x), terms)
  return [term(f.lhs.sym) ~ term(1); forms]
end
