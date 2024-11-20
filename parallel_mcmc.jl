using CUDA
using Pigeons
include("SplitRandom.jl") 
include("mh.jl")
include("Particles.jl")
include("barriers.jl")
include("kernels.jl")

@auto struct MCMC 
    chains # store current states of chains 
    backend
    timing # for the kernel only
    full_timing # for the full function call
end

Base.show(io::IO, m::MCMC) = print(io, "MCMC(backend=$(typeof(m.backend)), N=$(size(m.chains)[3]), time=$(m.timing.time)s)")


function parallel_mcmc(
    problem_obj; 
    backend::Backend = CPU(), 
    N::Int = backend isa CPU ? 2^5 : 2^10, # number of chains
    L::Int = 1000, # length of each chain
    seed = 1,
    multi_threaded = true,
    explorer = default_explorer(problem_obj), 
    elt_type::Type{E} = Float64,
    show_report::Bool = true
    ) where {E}

    full_timing = @timed begin
    
        @assert multi_threaded || backend isa CPU
        
        rngs = SplitRandomArray(N; backend, seed) 
        D = dimensionality(problem_obj)
        chains = KernelAbstractions.zeros(backend, E, L, D, N)

        # initialization: iid sampling from reference
        init_(backend, cpu_args(multi_threaded, N, backend)...)(rngs, problem_obj, chains, ndrange = N) 
        KernelAbstractions.synchronize(backend)

        # parallel sampling 
        buffers = KernelAbstractions.zeros(backend, E, buffer_size(explorer, problem_obj), N) 
        sample! = sample_(backend, cpu_args(multi_threaded, N, backend)...)
        timing = @timed begin
            sample!(rngs, problem_obj, explorer, chains, buffers, L, ndrange = N)
            KernelAbstractions.synchronize(backend)
            nothing
        end 

        nothing
    end
    return MCMC(chains, backend, timing, full_timing)
end

# workaround counter intuitive behaviour of KA on CPUs
cpu_args(multi_threaded::Bool, N::Int, ::CPU) = multi_threaded ? 1 : N
# the above is not needed for GPUs
cpu_args(_, _, _) = ()