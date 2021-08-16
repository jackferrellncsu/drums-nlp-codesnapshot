# Helper functions
include("routine_blstm.jl")

# Seed for random
Random.seed!(24)

# ----------------------- Loading data in for output ----------------------- #

# Loads in data from "datagen_blstm.jl"
blstm_results = JLD.load("out/out_blstm_results/run_blstm_output.jld")
plot_requirements = blstm_results["plot_requirements"]
t1_data = blstm_results["t1_data"]
t2_data = blstm_results["t2_data"]

α_i = get(plot_requirements, "α_i", 0)
pvals = get(plot_requirements, "pvals", 0)
actWords = get(plot_requirements, "actWords", 0)
sizes_99 = get(plot_requirements, "sizes_99", 0)

# ------------------------ Saving plots and figures ------------------------ #

# Finds and plots the distribution of nonconformity scores
nonconf_dist = histogram(α_i, color = :red, leg = false, grid = false,
                                                            linecolor = :white)

savefig(nonconf_dist, "out/out_blstm_results/blstm_nonconf_dist.png")

# Finds and plots the empirical vs proposed validity
valid, confs = empirical_proposed(pvals, actWords)
emp_prop = plot(confs, valid, label = "Empirical")
emp_prop = plot!(confs, confs, label = "Proposed")

savefig(emp_prop, "out/out_blstm_results/blstm_emp_prop.png")

set_sizes99 = histogram(sizes_99, leg = false, color = :red, grid = false,
                                                            linecolor = :white)

savefig(set_sizes99, "out/out_blstm_results/blstm_set_sizes99.png")

# ---------------------- Saving results for the tables ---------------------- #

t1 = Table(t1_data)
CSV.write("out/out_blstm_results/blstm_table1.CSV", t1)

t2 = Table(t2_data)
CSV.write("out/out_blstm_results/blstm_table2.CSV", t2)
