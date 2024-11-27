include("bench_variance_utils.jl")
using CairoMakie
using AlgebraOfGraphics

function run_bench_mcmc(; init_len, seed, model_type, scheme_type, elt_type, 
    max_chain_exponent = 15, min_chain_exponent = 10, ESS_thres = 100)
    result = DataFrame(
                N=Int[], 
                time=Float64[], 
                model = String[],
                type = Symbol[],
                backend=Symbol[],
                elt_type = String[],
                chain_length = Int[],
                max_Rhat = Float64[],
                med_Rhat = Float64[],
                max_Rhat2 = Float64[],
                med_Rhat2 = Float64[]
                )

    for backend in backends()
        @show backend
        target = build_target(backend, model_type)
        s = scheme(scheme_type, init_len, 1.0)
                        
        # warm-up 
        a = ais(target, s; seed, N = 1, backend, elt_type, show_report = false)
        println("Warm up: $(a.full_timing.time)") 

        # actual
        for N in map(i -> 2^i, (min_chain_exponent:max_chain_exponent))
            len = init_len
            @show N
            s = scheme(scheme_type, len, 1.0)
            a = ais(target, s; seed, N, backend, elt_type, show_report = false)
            while (maximum(nested_Rhat.(eachrow(a.particles.states))) > nRhat_from_ESS(ESS_thres,N)) & (len < 100000)
                @show maximum(nested_Rhat.(eachrow(a.particles.states)))
                @show nRhat_from_ESS(ESS_thres,N)
                len *= 2
                @show len
                s = scheme(scheme_type, len, 1.0)
                a = ais(target, s; seed, N, backend, elt_type, show_report = false)
            end
            ess_vec = 
            push!(result, (; 
                N, 
                time = a.full_timing.time,
                model = string(model_type),
                type = Symbol(scheme_type),
                backend = backend_label(backend),
                elt_type = string(elt_type),
                chain_length = len,
                max_Rhat = maximum(nested_Rhat.(eachrow(a.particles.states))),
                med_Rhat = median(nested_Rhat.(eachrow(a.particles.states))),
                max_Rhat2 = maximum(nested_Rhat.(eachrow(a.particles.states .^ 2))),
                med_Rhat2 = median(nested_Rhat.(eachrow(a.particles.states .^ 2)))
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

