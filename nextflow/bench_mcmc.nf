include { crossProduct; collectCSVs; deliverables; } from './utils.nf'
include { instantiate; precompile_gpu; } from "./nf-nest/pkg_gpu.nf"
include { activate; } from "./nf-nest/pkg.nf"

params.dryRun = false
def julia_env = file('julia_env')
def data_dir = file('../data')

def experiment_jl = "utils.jl,toy_unid.jl,simple_mixture.jl,SplitRandom.jl,report.jl,sais.jl,zja.jl,mh.jl,bench_variance_utils.jl,ais.jl,Particles.jl,kernels.jl,barriers.jl,bench_gpu_mcmc.jl,parallel_mcmc.jl,logistic_regression.jl,logistic_regression_data.jl,bench_gpu_particles.jl".split(",").collect{file("../" + it)}
def plot_jl = "utils.jl,toy_unid.jl,simple_mixture.jl,SplitRandom.jl,report.jl,bench_speedup_plot.jl,sais.jl,zja.jl,mh.jl,bench_variance_utils.jl,ais.jl,Particles.jl,kernels.jl,barriers.jl,parallel_mcmc.jl".split(",").collect{file("../" + it)}

def deliv = deliverables(workflow)

def variables = [
    job_seed: (1..10),
    n_rounds: (5..10),
    job_model: ["LogisticRegression"],
    job_scheme_types: ["SAIS"],
    job_elt_type: ["Float64"],
]
// def variables = [
//     job_seed: (1..10),
//     job_model: ["Unid", "SimpleMixture"],
//     job_scheme_types: ["SAIS", "ZJA"],
//     job_elt_type: ["Float64", "Float32"],
// ]

workflow  {
    compiled_env = instantiate(julia_env) | precompile_gpu
    args = crossProduct(variables, params.dryRun)
    results = run_experiment(compiled_env, data_dir, experiment_jl, args, params.dryRun) | collectCSVs
    //plot(julia_env, plot_jl, results)
}

process run_experiment {
    debug true
    label 'gpu'
    time 400.min
    memory = 30.GB
    errorStrategy 'ignore'
    //clusterOptions '--nodes 1', '--account st-alexbou-1-gpu', '--gpus 1'
    input:
        path julia_env
        path data_dir
        path jl_files
        val arg
        val dryRun
    output:
        tuple val(arg), path('csvs')
    """
    ${activate(julia_env)}

    n_rounds = ${arg.n_rounds}
    include(pwd() * "/bench_gpu_particles.jl")
    result = run_bench(; 
                n_rounds, 
                seed = ${arg.job_seed},
                model_type = ${arg.job_model},
                scheme_type = ${arg.job_scheme_types},
                elt_type = ${arg.job_elt_type},
            )
    
    mkdir("csvs")
    CSV.write("csvs/bench_speedup.csv", result; quotestrings = true)
    """
}

process plot {
    debug true
    time 5.min
    memory = 16.GB
    input:
        path julia_env
        // path toml_files
        path jl_files
        path aggregated 
    output:
        path "bench_mcmc_speedup.png"
    publishDir { deliverables(workflow) }, mode: 'copy', overwrite: true
    
    """ 
    ${activate(julia_env)}

    include(pwd() * "/bench_speedup_plot.jl")
    result = DataFrame(CSV.File("aggregated/bench_mcmc_speedup.csv"))

    fg = create_speedup_fig(result)
    save("bench_mcmc_speedup.png", fg)
    """
}