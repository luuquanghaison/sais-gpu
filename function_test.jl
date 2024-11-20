include("bench_gpu_particles.jl")
include("parallel_mcmc.jl")


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

