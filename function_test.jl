include("bench_gpu_mcmc.jl")
include("bench_gpu_particles.jl")
using StatsPlots
using PairPlots

n_rounds = 6

result = run_bench(; 
                n_rounds, 
                seed = 1,
                model_type = LogisticRegression,
                scheme_type = SAIS,
                elt_type = Float64,
                max_particle_exponent = 5
            )

result = run_bench(; 
            n_rounds = 6, 
            seed = 1,
            model_type = LogisticRegression,
            scheme_type = SAIS,
            elt_type = Float32,
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
y_vec = [1, 1, 1, 0, 1, 1, 1, 1, 0, 0]
x_mat = [1.0 -0.22431898850947196; 1.0 1.7394207065134153; 
        1.0 1.7007639471686942; 1.0 -0.831201924337179; 
        1.0 -0.39854651439527594; 1.0 -0.03729529996432233; 
        1.0 0.40501534544028; 1.0 -0.7439535958170874; 
        1.0 -0.5159742079044226; 1.0 -1.0939094681946309]
target = LogisticRegression(10,2,x_mat,y_vec)
model_approx_gcb = approx_gcb(target)
s = scheme(SAIS, 6, model_approx_gcb)
a = ais(target, s; seed=1, N=2^10, backend=CPU(), elt_type=Float64, show_report = false)

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


###################################################################
res_CGGibbs = DataFrame(CSV.File("nextflow/deliverables/bench_mcmc/aggregated/bench_speedup.csv"))

df = res_CGGibbs
n_rounds = 9
df = df[(df.n_rounds .== n_rounds),:]

StatsPlots.boxplot(log2.(df.N), log2.(df.ESS ./ df.time), title = "n_rounds = $n_rounds")
StatsPlots.xlabel!("log₂(particles)")
StatsPlots.ylabel!("log₂(ESS/s)")
StatsPlots.hline!