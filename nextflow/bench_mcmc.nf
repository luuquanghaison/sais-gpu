include { crossProduct; collectCSVs; deliverables; setupPigeons; } from './utils.nf'

params.dryRun = false
def julia_depot_dir = file("/scratch/st-alexbou-1/lqhson/sais-gpu/.depot")
// def julia_env_dir = file("..")
// def julia_depot_dir = file(".depot")
def toml_files = file("../*.toml")

def experiment_jl = "utils.jl,toy_unid.jl,simple_mixture.jl,SplitRandom.jl,report.jl,sais.jl,zja.jl,mh.jl,bench_variance_utils.jl,ais.jl,Particles.jl,kernels.jl,barriers.jl,bench_gpu_mcmc.jl,parallel_mcmc.jl".split(",").collect{file("../" + it)}
def plot_jl = "utils.jl,toy_unid.jl,simple_mixture.jl,SplitRandom.jl,report.jl,bench_variance_plot.jl,sais.jl,zja.jl,mh.jl,bench_variance_utils.jl,ais.jl,Particles.jl,kernels.jl,barriers.jl,parallel_mcmc.jl".split(",").collect{file("../" + it)}

def deliv = deliverables(workflow)

def variables = [
    job_seed: (1..10),
    job_model: ["LogisticRegression"],
    job_elt_type: ["Float64", "Float32"],
]

workflow  {
    args = crossProduct(variables, params.dryRun)
    // julia_env = setupPigeons(julia_depot_dir, julia_env_dir)
    results = run_experiment(julia_depot_dir, toml_files, experiment_jl, args, params.dryRun) | collectCSVs    
    plot(julia_depot_dir, toml_files, plot_jl, results)
}

process run_experiment {
    debug false
    time 400.min
    memory = 16.GB
    errorStrategy 'ignore'
    scratch true 
    clusterOptions '--nodes 1', '--account st-alexbou-1-gpu', '--gpus 1'
    input:
        env JULIA_DEPOT_PATH
        // path julia_env
        path toml_files
        path jl_files
        val arg
        val dryRun
    output:
        tuple val(arg), path('csvs')
    """
    #!/usr/bin/env -S julia

    include(joinpath("/scratch/st-alexbou-1/lqhson/sais-gpu", "bench_gpu_mcmc.jl"))

    result = run_bench_mcmc(; 
        seed = ${arg.job_seed}, 
        model_type = ${arg.job_model}, 
        elt_type = ${arg.job_elt_type})
    
    mkdir("csvs")
    CSV.write("csvs/bench_mcmc_speedup.csv", result; quotestrings = true)
    """
}

process plot {
    debug true
    time 5.min
    memory = 16.GB
    input:
        env JULIA_DEPOT_PATH
        // path julia_env
        path toml_files
        path jl_files
        path aggregated 
    output:
        path "bench_mcmc_speedup.png"
    publishDir { deliverables(workflow) }, mode: 'copy', overwrite: true
    
    """ 
    #!/usr/bin/env -S julia --project=@.

    include(joinpath("/scratch/st-alexbou-1/lqhson/sais-gpu", "bench_speedup_plot.jl"))
    result = DataFrame(CSV.File("aggregated/bench_mcmc_speedup.csv"))

    fg = create_speedup_fig(result)
    save("bench_mcmc_speedup.png", fg)
    """
}