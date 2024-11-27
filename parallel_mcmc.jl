using CUDA
using Pigeons
include("SplitRandom.jl") 
include("mh.jl")
include("Particles.jl")
include("barriers.jl")
include("kernels.jl")

@auto struct MCMC 
    len # chain length
end
MCMC() = MCMC(1000)
ais(path, s::MCMC; kwargs...) = ais(path, fill(1, s.len); kwargs...)
