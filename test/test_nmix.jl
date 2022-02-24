@testset "Nmix GOF" begin

  #Test separating cohorts by NA pattern
  Random.seed!(123)
  umd1 = simulate(Nmix, @formula(ψ~elev+forest), @formula(p~wind+precip),
                  (50,3), [0,0,0,0,0,0])
  fit1 = nmix(@formula(ψ~elev), @formula(p~wind), umd1)

  yna = Array{Union{Int,Missing}}(deepcopy(umd1.y))
  yna[1,:] = fill(missing,3)
  yna[2,1] = missing
  yna[3,[1,3]] = fill(missing,2)
  umd2 = UmData(yna, umd1.site_covs, umd1.obs_covs)
  fit2 = nmix(@formula(ψ~elev), @formula(p~wind), umd2)

  #Test fitted
  ft = Unmarked.fitted(fit1)
  @test ft isa Array
  @test length(ft) == 150
  @test round(sum(ft),digits=4) == 56.8432

  #Test residuals
  rs = Unmarked.residuals(fit1)
  @test rs isa Array
  @test length(rs) == 150
  @test round(sum(rs), digits=4) == -0.1568

  #Test chi-square calc
  @test isequal(round(Unmarked.chisq(fit1),digits=4), 144.1886)
  #with missing values
  @test isequal(round(Unmarked.chisq(fit2),digits=4), 138.6249)

  #Test parametric bootstrap
  Random.seed!(123)
  gf = gof(fit1)
  @test isequal(length(gf.tstar), 30)
  @test isequal(round(mean(gf.t0 .< gf.tstar), digits=4), 0.3667)
  gf2 = gof(fit2)
  @test isequal(mean(gf2.t0 .< gf2.tstar), 0.5000)
end
