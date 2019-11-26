#Dot and whisker plots of coefficient values + confidence intervals

#For submodels
"""
    whiskerplot(m::UnmarkedSubmodel; level::Real = 0.95)

Generate a dot-and-whisker plot for all parameters in a given submodel `m`.
Error bars are drawn for given confidence `level`.
"""
function whiskerplot(m::UnmarkedSubmodel; level::Real = 0.95)
  
  #Build data frame of coefficients and upper/lower bounds
  df = DataFrame(names=coefnames(m), coef=coef(m), 
                 se=stderror(m))
  zval = -quantile(Normal(), (1-level)/2)
  df[!,:lower] = df.coef - zval * df.se
  df[!,:upper] = df.coef + zval * df.se

  plot(df, x=:names, y=:coef, ymin=:lower, ymax=:upper,
       yintercept=[0.0],
       style(major_label_font_size=18pt, 
             minor_label_font_size=16pt,
             point_size=5pt,line_width=2pt),
       Guide.xlabel("Parameter"), Guide.ylabel("Value"),
       Guide.title(string(m.name)),
       Geom.point, Geom.errorbar, 
       Geom.hline(color="orange",style=:dash))

end

#For complete model
"""
    whiskerplot(m::UnmarkedModel; level::Real = 0.95)

Generate a dot-and-whisker plot for all parameters in a given model `m`.
Error bars are drawn for given confidence `level`.
"""
function whiskerplot(m::UnmarkedModel; level::Real = 0.95)
  sm = collect(m.submodels)
  vstack(whiskerplot.(sm, level=level))
end

#------------------------------------------------------------------------------

#Partial effects plots for each parameter

#Get array filled with "baseline" value for variable (mean or reference level)
function get_var_baseline(x::Array{<:Number}, nrep::Int)
  return repeat([mean(x)], nrep)
end

function get_var_baseline(x::CategoricalArray, nrep::Int)
  out = categorical(repeat([levels(x)[1]], nrep))
  levels!(out, levels(x))
  return out
end

#Get sequence of values for focal parameter
function get_var_seq(x::Array{<:Number})
  nrow = 100
  out = collect(range(min(x...), stop=max(x...), length=nrow))
  return (nrow, out)
end

function get_var_seq(x::CategoricalArray)
  nrow = length(levels(x))
  out = categorical(levels(x))
  levels!(out, levels(x))
  return (nrow, out)
end

#For a single parameter
"""
    effectsplot(m::UnmarkedSubmodel, param::String; level::Real = 0.95)

Generate a partial effects plot for parameter `param` from submodel `m`.
The response variable is plotted on the original scale. A confidence envelope
is also plotted for the given `level`.
"""
function effectsplot(m::UnmarkedSubmodel, param::String; level::Real = 0.95)

  resp = string(m.formula.lhs.sym)
  psym = Symbol(param)
  dat = deepcopy(m.data)
  
  #Build newdata for predict
  nrow, var_seq = get_var_seq(dat[:, psym])
  nd = aggregate(dat, x -> get_var_baseline(x, nrow))
  names!(nd, names(dat)) 
  nd[!, psym] = var_seq
  
  #Predict values
  pr = DataFrame(predict(m, nd, interval=true, level=level))
  pr[!, psym] = var_seq
  
  #Choose plot type depending on variable type
  estplot = Geom.line; errorplot = Geom.ribbon; lw = 1pt
  if typeof(var_seq) <: CategoricalArray
    estplot = Geom.point; errorplot = Geom.errorbar; lw = 2pt
  end

  plot(pr, x=psym, y=:prediction, ymin=:lower, ymax=:upper,
       estplot, errorplot, Guide.xlabel(param), Guide.ylabel(resp),
       style(major_label_font_size=18pt, minor_label_font_size=16pt,
             point_size=5pt, line_width=lw))
 
end 

#For a submodel
"""
    effectsplot(m::UnmarkedSubmodel; level::Real = 0.95)

Partial effects plot for all parameters in a submodel `m`.
"""
function effectsplot(m::UnmarkedSubmodel; level::Real = 0.95) 
  vars = string.(collect(m.formula.rhs))
  vstack(map(x -> effectsplot(m, x, level=level), vars))
end

#For a complete model
"""
    effectsplot(m::UnmarkedModel; level::Real = 0.95)

Partial effects plot for all parameters in a model `m`.
"""
function effectsplot(m::UnmarkedModel; level::Real = 0.95)
  cols = effectsplot.(collect(m.submodels), level=level)
  hstack(cols[1], cols[2])
end
