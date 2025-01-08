include("sais.jl")
include("zja.jl")
include("parallel_mcmc.jl")
include("simple_mixture.jl")
include("logistic_regression.jl")
include("toy_unid.jl")
using DataFrames 
using CSV

build_target(backend, ::Type{Unid}) = Unid(10^11, 10^10) 
build_target(backend, ::Type{SimpleMixture}) = SimpleMixture(backend)
build_target(backend, ::Type{LogisticRegression}) = LogisticRegression(; backend, file = "data/ALLAML.mat", requested_dim = 64)

# Based on a large run:
# seed 1 => -782.1570317771477
# seed 2 => -781.9446689097091
ref_log_Z(::SimpleMixture) = -782
approx_gcb(::SimpleMixture) = 7.0 

# Based on a large run ( 5.24e+05   1.02e+03):
# seed 1 => -24.6 
# seed 2 => -24.4
ref_log_Z(::Unid) = -24.5
approx_gcb(::Unid) = 17.0


ref_log_Z(::LogisticRegression) = -27.5
approx_gcb(::LogisticRegression) = 9.9


scheme(::Type{SAIS}, n_rounds, Λ) = SAIS(n_rounds)
scheme(::Type{FixedSchedule}, n_rounds, Λ) = FixedSchedule(2^(n_rounds-1)) 
scheme(::Type{ZJA}, n_rounds, Λ) = ZJA((Λ / 2^(n_rounds-1))^2)
scheme(::Type{MCMC}, len, Λ) = MCMC(len) # specify chain length instead of n_rounds

function med_mc_err(lr::LogisticRegression, p::Particles) # return log2 of median Monte Carlo error
    d = lr.p
    N = length(p.probabilities)
    err = Float64[]
    for i in 1:d
        weighted_vals = p.states[i,:] .* p.probabilities
        first_moment = sum(weighted_vals)^2/N
        second_moment = sum(x -> x^2, weighted_vals)
        push!(err, second_moment-first_moment)
    end

    return log2(median(err))
end
