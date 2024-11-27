include("bench_gpu_mcmc.jl")
using StatsPlots
using PairPlots

n_rounds = 1

result = run_bench(; 
                n_rounds, 
                seed = 1,
                model_type = LogisticRegression,
                scheme_type = SAIS,
                elt_type = Float64,
                max_particle_exponent = 5
            )




backend = backends()[1]
target = build_target(backend, LogisticRegression)
model_approx_gcb = approx_gcb(target)
approx_log_Z = ref_log_Z(target)
result = parallel_mcmc(target; 
    seed = 1, 
    N = 1, 
    L = 1000,
    backend = backend, 
    elt_type = Float64, 
    show_report = false)

α = 0.05
filtered_samples = mapslices(result.chains[:,[1,2],1]; dims=1) do c
    qs = quantile(c, (α, 1-α))
    filter(x -> first(qs) ≤ x ≤ last(qs), c)
end
pairplot(filtered_samples)


###############################################################
target = build_target(CPU(), LogisticRegression)
s = scheme(MCMC, 1000, 1.0)
a = ais(target, s; seed=1, N=2^8, backend=CPU(), elt_type=Float64, show_report = false)

maximum(nested_Rhat.(eachrow(a.particles.states)))
nRhat_from_ESS(100,2^8)


α = 0.05
filtered_samples = mapslices(a.particles.states[[1,2],:]'; dims=1) do c
    qs = quantile(c, (α, 1-α))
    filter(x -> first(qs) ≤ x ≤ last(qs), c)
end
pairplot(filtered_samples)


res = run_bench_mcmc(; init_len = 100, seed = 1, model_type = LogisticRegression,
scheme_type = MCMC, elt_type = Float64, max_chain_exponent = 7,min_chain_exponent = 7)