@testset "Occu GOF" begin
  
  #Test separating cohorts by NA pattern
  Random.seed!(123)
  umd1 = simulate(Occu, @formula(ψ~elev+forest), @formula(p~wind+precip),
                  (10,3), [0,0,0,0,0,0])
  fit1 = occu(@formula(ψ~elev), @formula(p~wind), umd1)
  
  gc = Unmarked.get_cohorts(fit1)
  @test isequal(length(gc), 1)
  @test isequal(gc[1].y, umd1.y)

  yna = Array{Union{Int,Missing}}(deepcopy(umd1.y))
  yna[1,:] = fill(missing,3)
  yna[2,1] = missing
  yna[3,[1,3]] = fill(missing,2)
  umd2 = UmData(yna, umd1.site_covs, umd1.obs_covs)
  fit2 = occu(@formula(ψ~elev), @formula(p~wind), umd2)
  
  gc = Unmarked.get_cohorts(fit2)
  @test isequal(length(gc), 3)
  @test isequal(gc[1].y, umd2[4:10].y)
  @test isequal(gc[2].y, umd2[3].y)
  @test isequal(gc[3].y, umd2[2].y)

  #Test encounter history probability
  psi = 0.5
  pval = [0.2, 0.8, 0.3]
  eh1 = "0 1 1"
  ehp1 = Unmarked.get_prob_eh(Occu, eh1, pval, psi)
  @test isequal(ehp1, 0.5*prod(pval .* [0,1,1] 
                               + (1 .- pval) .* (1 .- [0,1,1])))
  eh2 = "0 missing 1"
  ehp2 = Unmarked.get_prob_eh(Occu, eh2, pval, psi)
  @test isequal(ehp2, 0.5*prod([0.2,0.3] .* [0,1] 
                               + (1 .- [0.2,0.3]) .* (1 .- [0,1])))

  #Test getting obs and expected counts
  oc = Unmarked.get_obs_counts(fit1, umd1)
  @test isequal(length(oc), 5)
  @test isequal(sum(collect(values(oc))), 10) 
  @test isequal(oc["0 0 0"], 6)
  ec = Unmarked.get_exp_counts(fit1, umd1)
  @test isequal(keys(oc), keys(ec))
  @test isequal(round(ec["1 1 0"],digits=6), 0.468793)

  #Test chi-square calc
  @test isequal(round(Unmarked.mb_chisq(fit1),digits=4), 3.0741)
  #with missing values
  @test isequal(round(Unmarked.mb_chisq(fit2),digits=4), 3.6675)

  #Test parametric bootstrap
  Random.seed!(123)
  gf = gof(fit1)
  @test isequal(length(gf.tstar), 50)
  @test isequal(mean(gf.t0 .< gf.tstar), 0.72)
  gf2 = gof(fit2)
  @test isequal(mean(gf2.t0 .< gf2.tstar), 0.86)
  
end
