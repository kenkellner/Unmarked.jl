@testset "UmData" begin

  #Test dataframe checker
  df1 = DataFrame(a=[1,2],b=[3,4])
  @test isequal(df1, Unmarked.check_data(df1,2))
  @test_throws ErrorException Unmarked.check_data(df1,3) 
  df2 = DataFrame(a=[1,missing],b=[3,4])
  @test_throws ErrorException Unmarked.check_data(df2,2)
  df3 = DataFrame(_dummy=[1,1])
  @test isequal(Unmarked.check_data(nothing, 2), df3)

  #Test basic constructor
  y = [[1 0]; [0 1]; [1 1]] 
  sc = DataFrame(a=[1,2,3], b=[3,4,5])
  oc = DataFrame(c=collect(1:6), d=collect(1:6))
  dat = UmData(y, sc, oc)
  @test isequal(dat.y, y)
  @test isequal(dat.site_covs, sc)
  @test isequal(dat.obs_covs, oc)
  sc2 = DataFrame(a=[1,2], b=[3,4])
  scno = DataFrame(_dummy=[1.0,1.0,1.0])
  @test_throws ErrorException UmData(y, sc2, oc)
  dat2 = UmData(y, nothing, oc)
  @test isequal(dat2.site_covs, scno)
  
  #Test missing values
  yna = [[missing 0]; [0 1]; [1 1]]
  dat2 = UmData(yna, sc, oc)
  @test isequal(dat2.y, yna)

  #Test outer constructor
  dat3 = UmData(y, obs_covs=oc, site_covs=sc)
  @test isequal(dat3.site_covs, sc)
  dat4 = UmData(y, obs_covs=oc)
  @test isequal(dat4.site_covs, DataFrame(_dummy=[1.0,1.0,1.0]))
  @test_throws ErrorException UmData(y, site_covs=oc, obs_covs=oc)
  dat5 = UmData(yna)
  @test isequal(dat5.y, yna)

  #Test subsetting
  ds1 = dat[1]
  @test isequal(size(ds1.y), (1,2))
  @test isequal(ds1.y[1,:], y[1,:])
  @test isequal(ds1.site_covs, sc[1:1,:])
  @test isequal(ds1.obs_covs, oc[1:2,:])

  ds2 = dat[2:3]
  ds3 = dat[[2,3]]
  @test isequal(ds2.y, ds3.y)
  @test isequal(ds2.y, y[2:3,:])
  @test isequal(ds2.site_covs, sc[2:3,:])
  @test isequal(ds2.obs_covs, oc[3:6,:])

end
