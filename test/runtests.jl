using Test, Unmarked, StatsModels, StatsFuns, DataFrames

@testset "UmData" begin

  y = [[1 0] [0 1]] 
  sc = DataFrame(a=[1,2], b=[3,4])
  oc = DataFrame(c=[5,6], d=[7,8])
  dat = UmData(y, sc, oc)
  @test isequal(dat.y, y)
  @test isequal(dat.site_covs, sc)
  @test isequal(dat.obs_covs, oc)

  yna = [[missing 0] [0 1]]
  dat2 = UmData(yna, sc, oc)
  @test isequal(dat2.y, yna)

  scna = DataFrame(a=[1,missing],b=[3,4])
  @test_throws ErrorException Unmarked.check_data(scna)

end

@testset "UmDesign" begin

  y = [[1 0] [0 1]] 
  sc = DataFrame(a=[1,2], b=[3,4])
  oc = DataFrame(c=[5,6], d=[7,8])
  dat = UmData(y, sc, oc)

  umd = Unmarked.UmDesign(:test, @formula(ψ~a+b), 
                          Unmarked.LogitLink(), dat.site_covs,
                          nothing, nothing, nothing)
 
  @test isequal(umd.name, :test)
  @test isequal(umd.formula, @formula(ψ~a+b))
  @test isequal(umd.link, Unmarked.LogitLink())
  @test isequal(umd.data, dat.site_covs)
  @test all(map(x -> x == nothing, [umd.coefnames,umd.mat,umd.idx]))

  #Test add dummy response colum
  df_test = deepcopy(sc)
  df_test[!,:ψ] = zeros(2)
  new_sc = Unmarked.add_resp(umd.data, umd.formula)
  @test isequal(new_sc, df_test)
  @test isequal(names(sc), [:a, :b])
  @test isequal(names(umd.data), [:a, :b])
  @test !(new_sc === sc)
  @test umd.data === sc
 
  #Add design matrix
  Unmarked.add_dm!(umd)
  @test isequal(umd.mat, [[1,1] [1,2] [3,4]])

  #Test outer constructor
  umd2 = Unmarked.UmDesign(:test2, @formula(p~c), Unmarked.LogitLink(),
                          dat.obs_covs)
  @test isequal(umd2.mat, [[1.0, 1.0] [5.0, 6.0]])
  @test umd2.data === oc

  #Test getting coefficient indices
  Unmarked.add_idx!([umd, umd2])
  @test isequal(umd.idx, 1:3)
  @test isequal(umd2.idx, 4:5)

  #Test transform
  tr = logistic.(umd.mat * [0,0.3,0.5])
  @test isequal(Unmarked.transform(umd, [0,0.3,0.5]), tr)

end

@testset "Link functions" begin
  
  #Logit link
  @test isequal(Unmarked.invlink(0.5, Unmarked.LogitLink()), logistic(0.5))
  @test isequal(Unmarked.invlink([0.3,0.5], Unmarked.LogitLink()),
                logistic.([0.3,0.5]))
  @test isequal(Unmarked.grad(0.5, Unmarked.LogitLink()),
                exp(-0.5)/(1+exp(-0.5))^2)
  @test isequal(Unmarked.grad([0.3,0.5], Unmarked.LogitLink()),
                map(x -> exp(-x)/(1+exp(-x))^2, [0.3,0.5]))

  #Log link
  @test isequal(Unmarked.invlink(0.5, Unmarked.LogLink()), exp(0.5))
  @test isequal(Unmarked.invlink([0.3,0.5], Unmarked.LogLink()),
                exp.([0.3,0.5]))
  @test isequal(Unmarked.grad(0.5, Unmarked.LogLink()), exp(0.5))
  @test isequal(Unmarked.grad([0.3,0.5], Unmarked.LogLink()),
                exp.([0.3,0.5]))

end
