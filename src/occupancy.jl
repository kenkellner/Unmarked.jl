struct Occu <: UnmarkedModel
  data::UmData
  opt::UnmarkedOpt
  submodels::NamedTuple
end

"Fit single-season occupancy models"
function occu(ψ_formula::FormulaTerm, p_formula::FormulaTerm, data::UmData)
  
  occ = UmDesign(:Occupancy, ψ_formula, LogitLink(), data.site_covs)
  det = UmDesign(:Detection, p_formula, LogitLink(), data.obs_covs)
  add_idx!([occ, det])
  np = det.idx.stop

  y = data.y
  N, J = size(y)
  nd = ndetects(y)

  function loglik(β::Array)
    
    ψ = transform(occ, β)
    p = transform(det, β)

    ll = zeros(eltype(β), N)

    idx = 0
    for i = 1:N
      if all(ismissing.(y[i,:]))
        idx += J
        continue
      end
      cp = 1.0
      for j = 1:J
        idx += 1
        if ismissing(y[i,j])
          continue
        end
        cp *= p[idx]^y[i,j] * (1-p[idx])^(1-y[i,j])
      end
      ll[i] = log(ψ[i] * cp + (1-ψ[i]) * (nd[i]==0))
    end

    return -sum(ll)
  end

  opt = optimize_loglik(loglik, np)
  Occu(data, opt, (occ=UnmarkedSubmodel(occ,opt),
                   det=UnmarkedSubmodel(det,opt)))
end

#Simulations
"Simulate new dataset from a fitted model"
function simulate(fit::Occu)
  
  ydims = collect(size(fit.data.y))
  ψ = predict(occupancy(fit))
  p = predict(detection(fit))
  
  y = _sim_y_occu(ψ, p, ydims)
  
  UmData(y, fit.data.site_covs, fit.data.obs_covs)
end

"Simulate new dataset from provided formulas"
function simulate(::Type{Occu}, ψ_formula::FormulaTerm, 
                  p_formula::FormulaTerm, ydims::Array{Int}, coef::Array)
 
  sc = gen_covs(ψ_formula, ydims[1])
  oc = gen_covs(p_formula, ydims[1] * ydims[2])

  occ = UmDesign(:Occ, ψ_formula, LogitLink(), sc)
  det = UmDesign(:Det, p_formula, LogitLink(), oc)
  add_idx!([occ, det])

  np = det.idx.stop
  if np != length(coef)
    error(string("Coef array must be length ",np))
  end

  ψ = transform(occ, coef)
  p = transform(det, coef)

  y = _sim_y_occu(ψ, p, ydims)

  UmData(y, sc, oc)
end

#Simulate y matrix from provided psi and p vectors"
function _sim_y_occu(ψ::Array, p::Array, ydims::Array)
  
  N,J = ydims
  z = map(x -> rand(Binomial(1, x))[1], ψ)
  y = Array{Int}(undef, N, J)
  idx = 0
  for i = 1:N
    for j = 1:J
      idx += 1
      y[i,j] = rand(Binomial(1, p[idx] * z[i]))[1]
    end
  end

  return y
end
