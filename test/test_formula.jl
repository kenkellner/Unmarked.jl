@testset "Formulas" begin

  #Test extracting covariate names from formulas with no schema
  f_int = @formula(y~1)
  @test isequal(Unmarked.varnames(f_int), (lhs=:y, rhs=nothing))  
  f1 = @formula(y~a)
  f2 = @formula(y~a+b)
  f3 = @formula(y~a*b+c)
  @test isequal(Unmarked.varnames(f1).rhs, [:a])
  @test isequal(Unmarked.varnames(f2).rhs, [:a, :b])
  @test isequal(Unmarked.varnames(f3).rhs, [:a, :b, :c])

  #Test building all subsets of formulas
  @test isequal(allsub(f_int), [f_int])
  @test isequal(allsub(f1), [f1])
  all_forms = [@formula(y~1),@formula(y~a),@formula(y~b),@formula(y~a+b)]
  @test isequal(string.(allsub(f2)), string.(all_forms))

end
