#export JULIA_NUM_THREADS=3 #in bash
Threads.nthreads() #should be >1

using Distributed, DataFrames
addprocs(3);
@everywhere using Unmarked



ψ_formula = @formula(ψ~elev+forest);
p_formula = @formula(p~precip+wind);
β_truth = [0, -0.5, 1.2, -0.2, 0, 0.7];

umd = simulate(Occu, ψ_formula, p_formula, [1000, 5], β_truth);

fit = occu(ψ_formula, p_formula, umd);


#bootstrap

@everywhere function calc_stat(i, fit::Unmarked.UnmarkedModel)
  s = simulate(fit)
  sum(s.y)
end

@everywhere calc_stat2 = calc_stat2

@time y = map(x -> calc_stat(x, fit), 1:5000);

@time y2 = pmap(x -> calc_stat(x, fit), 1:5000);

