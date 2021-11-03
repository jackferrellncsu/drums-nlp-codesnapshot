# Helper functions
include("routine_blstm.jl")

# ----------------------- Loading data in for output ----------------------- #

# Loads in data from "datagen_blstm.jl"
blstm_results = JLD.load("out/out_blstm_results/run_blstm_output.jld")
plot_requirements = blstm_results["plot_requirements"]
t1_data = blstm_results["t1_data"]
t2_data = blstm_results["t2_data"]

# Loads all data needed from "plot_requirements" dictionary
α_i = get(plot_requirements, "α_i", 0)
nominal = get(plot_requirements, "nominal", 0)
confs = get(plot_requirements, "confs", 0)
sizes_99 = get(plot_requirements, "sizes_99", 0)
sizes_999 = get(plot_requirements, "sizes_999", 0)

# ------------------------ Saving plots and figures ------------------------ #

# Finds and plots the distribution of nonconformity scores
nonconf_dist = histogram(α_i, bins = 0:.022:1, color = :black, leg = false,
                            grid = false, linecolor = false,
                            yticks = ([0:20000:100000;],
                            ["0", "20000", "40000", "60000", "80000", "100000"]))

savefig(nonconf_dist, "out/out_blstm_results/blstm_nonconf_dist.png")

# Finds and plots the closer distribution of nonconformity scores
ranger = ranged_α(α_i)
close_nonconf_dist = histogram(ranger, bins = 0:8.547008547008547e-7:.0002,
                                color = :black, leg = false, grid = false,
                                linecolor = false, yticks = ([0:2000:8000;],
                                ["0", "2000", "4000", "6000", "8000"]),
                                xticks = ([0:.0002:.0002;], ["0", "0.0002"]))

savefig(close_nonconf_dist, "out/out_blstm_results/blstm_nonconf_dist_close.png")

# Finds and plots the empirical vs proposed validity
emp_prop = plot(confs, nominal, leg = false, grid = false, color = :black)
emp_prop = plot!(confs, confs, color = :black, linestyle = :dash)

savefig(emp_prop, "out/out_blstm_results/blstm_emp_prop.png")

# Finds and plots the distribution of set sizes for 99.9% confidence
set_sizes999 = histogram(sizes_999, leg = false, bins = 141, color = :black, grid = false,
                            linecolor = false, yticks = ([0:20000:40000;],
                            ["0", "20000", "40000"]), xticks = ([0:20:140;],
                            ["0", "20", "40", "60", "80", "100", "120", "140"]))

savefig(set_sizes999, "out/out_blstm_results/blstm_set_sizes99.png")

# ---------------------- Saving results for the tables ---------------------- #

t1 = Table(t1_data)
CSV.write("out/out_blstm_results/blstm_table1.CSV", t1)

t2 = Table(t2_data)
CSV.write("out/out_blstm_results/blstm_table2.CSV", t2)
