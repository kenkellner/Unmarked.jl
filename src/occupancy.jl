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
  
  UmFit(opt.vcov, opt.AIC,
        (occ=UmModel(occ,opt), det=UmModel(det,opt)))

end
