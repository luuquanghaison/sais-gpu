include("bench_variance_utils.jl")
include("parallel_mcmc.jl")
using CairoMakie
using AlgebraOfGraphics

function run_bench_mcmc(; seed, model_type, elt_type, max_chain_exponent = 10)
    result = DataFrame(
                N=Int[], 
                time=Float64[], 
                model = String[],
                type = Symbol[],
                backend=Symbol[],
                elt_type = String[],
                )

    for backend in backends()
        @show backend
        target = build_target(backend, model_type)
                        
        # warm-up 
        m = parallel_mcmc(target; seed, N = 1, L = 10, backend, elt_type, show_report = false)
        println("Warm up: $(m.full_timing.time)") 

        # actual
        for N in map(i -> 2^i, (0:max_chain_exponent))
            @show N
            m = parallel_mcmc(target; seed, N, L = 1000, backend, elt_type, show_report = false)
            push!(result, (; 
                N, 
                time = m.full_timing.time,
                model = string(model_type),
                type = Symbol("None"),
                backend = backend_label(backend),
                elt_type = string(elt_type)
            ))
        end
    end
    return result
end

plot_gpu_chains(result) =
    data(result) * 
        visual(Lines) *
        mapping(
            :N => "Number of chains", 
            :time => "Wallclock time (s)", 
            color = :backend) 

function plot_gpu_chains(result, to_file) 
    p = plot_gpu_chains(result) 
    axis = (width = 225, height = 225, xscale = log2, yscale = log2)
    fg = draw(p; axis)
    save(to_file, fg)
end

