using KernelAbstractions
using MCMCChains

"""
Normalize a vector of non-negative weights stored in 
log-scale. Returns the log normalization. 
Works on both CPU and GPU. 

I.e., given input `log_weights` (an array), 
perform in-place, numerically stable implementation of: 
```
prs = exp.(log_weights)
log_weights .= prs/sum(prs)
```

Moreover, returns a numerically stable version of 
```
log(sum(prs))
```
"""
function exp_normalize!(log_weights)
    m = maximum(log_weights)
    log_weights .= exp.(log_weights .- m) 
    return m + log(normalize!(log_weights))
end 

function normalize!(weights) 
    s = sum(weights)
    weights .= weights ./ s 
    return s
end

# vectorized log_sum_exp
function log_sum_exp(log_weights; dims...) 
    m = maximum(log_weights; dims...)
    m .= m .+ log.(sum(exp.(log_weights .- m); dims...))
    return m
end

# create a copy to CPU of an arbitrary array
copy_to_cpu(array) = Array(array)

ensure_to_cpu(array::Array) = array 
ensure_to_cpu(array) = copy_to_cpu(array)

# create a copy to device of an arbitrary array
copy_to_device(array::AbstractArray{E, N}, backend) where {E, N} = 
    copy_to_device(array, backend, E)
function copy_to_device(array::AbstractArray{E, N}, backend, ::Type{F}) where {E, N, F}
    result = KernelAbstractions.zeros(backend, F, size(array)) 
    copyto!(result, array)
    return result
end

dummy = ""
function gpu_available() 
    try 
        # dummy needed to avoid compiler removing that line!
        global dummy = CUDA.driver_version()
        return true
    catch 
        return false
    end
end

backends() = gpu_available() ? [CUDABackend()] : [CPU()]

backend_label(::CPU) = :CPU 
backend_label(_)     = :GPU


# superchain split function
function superchain_split(N) 
    @assert isinteger(log2(N)) # expect N = 2^i
    M = 2^(ceil(Int,log2(N)/2)) # set number of chains M per superchain close to √N
    K = Int(N/M) # number of superchains
    return M, K
end

# nested R hat
function nested_Rhat(particles)
    N = length(particles)
    M, K = superchain_split(N)
    superchains = reshape(particles, (M, K))
    Bhat_nu = var(vec(mean(superchains,dims=1)))
    Btilde_nu = vec(var(superchains,dims=1))
    #Wtilde_nu = fill(0,K)
    What_nu = mean(Btilde_nu) #.+ Wtilde_nu)
    return sqrt(1+Bhat_nu/What_nu)
end

# nested R hat threshold based on ESS and number of chains
function nRhat_from_ESS(ess_val,N) 
    M, K = superchain_split(N)
    return sqrt(1+1/M+3/(ess_val))
end