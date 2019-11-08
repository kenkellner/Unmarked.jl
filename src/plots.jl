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

#For a single parameter
"""
    effectsplot(m::UnmarkedSubmodel, param::String; level::Real = 0.95)

Generate a partial effects plot for parameter `param` from submodel `m`.
The response variable is plotted on the original scale. A confidence envelope
is also plotted for the given `level`.
"""
function effectsplot(m::UnmarkedSubmodel, param::String; level::Real = 0.95)
  #Build newdata for predict
  psym = Symbol(param)
  dat = deepcopy(m.data)
  param_dat = dat[:, psym]
  val_rng = collect(range(min(param_dat...), stop=max(param_dat...),
                          length=100))
  mns = colwise(mean, dat)  
  nd = DataFrame()
  for i in 1:length(mns) nd[!,names(dat)[i]] = repeat([mns[i]],100) end 
  nd[!, psym] = val_rng
  
  #Predict values
  pr = DataFrame(predict(m, nd, interval=true, level=level))
  pr[!, psym] = val_rng
  
  #Plot line with CI ribbon
  resp = string(m.formula.lhs.sym)
  plot(pr, x=psym, y=:prediction, ymin=:lower, ymax=:upper,
              Geom.line, Geom.ribbon,
              style(major_label_font_size=18pt, 
                    minor_label_font_size=16pt),
              Guide.xlabel(param), Guide.ylabel(resp))

end 

#For a submodel
"""
    effectsplot(m::UnmarkedSubmodel; level::Real = 0.95)

Partial effects plot for all parameters in a submodel `m`.
"""
function effectsplot(m::UnmarkedSubmodel; level::Real = 0.95) 
  vars = string.(names(m.data)) 
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
