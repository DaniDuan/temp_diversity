# =============================================================================
# summary_temp_div.jl
#
# Post-processing script for cluster use. 
# Run this on the cluster after all sim_div.jl job array replicates have completed. 
# To collect and summarise their per-replicate JLD2 outputs into a single file per condition.
# =============================================================================

include(joinpath(@__DIR__, "../Base/sim_frame.jl"))

using ProgressMeter, RCall, Glob, ColorSchemes, KernelDensity, Colors

# =============================================================================
# Parameters
# =============================================================================
N          = 50
M          = 50
L          = fill(0.3, N)
num_temps  = 31                 # temperature steps from 0 to 30°C
ρ_t        = [-0.3500 -0.3500]  # Realistic trade-off
# ρ_t      = [ 0.0000  0.0000]  # Minimal trade-off
# ρ_t      = [-0.9999 -0.9999]  # Maximal trade-off
Tr         = 273.15 + 10        # Reference temperature (K)
Ed         = 3.5                # High-temperature deactivation energy (eV)
input_type = ["constant", "leaching", "chemostat", "self-renewing"]

# =============================================================================
# Condition definitions
# Each entry is a (label, glob path) pair. The label is used only to name the output variables. 
# Add new conditions here without changing anything else.
# =============================================================================
conditions = [
    # (label = "0",  path = "../../results/20260325/temp_rich_0"),
    (label = "re", path = "../../results/20260325/temp_rich"),
    # (label = "1",  path = "../../results/20260325/temp_rich_1"),
]

# =============================================================================
# Function init_collectors(): returns a fresh NamedTuple of empty collector vectors, matching the structure saved by sim_div.jl
# =============================================================================
function init_collectors()
    (
        rich    = Vector{Vector{Float64}}(),
        Shannon = Vector{Vector{Float64}}(),
        Simpson = Vector{Vector{Float64}}(),
        sur     = Vector{Vector{Vector{Float64}}}(),
        ext     = Vector{Vector{Vector{Float64}}}(),
        r_inv   = Vector{Vector{Vector{Float64}}}(),
        ϵ       = Vector{Vector{Vector{Float64}}}(),
        u       = Vector{Vector{Matrix{Float64}}}(),
        m       = Vector{Vector{Vector{Float64}}}(),
        Eu      = Vector{Vector{Vector{Float64}}}(),
        Em      = Vector{Vector{Vector{Float64}}}(),
        Tpu     = Vector{Vector{Vector{Float64}}}(),
        Tpm     = Vector{Vector{Vector{Float64}}}(),
    )
end

# =============================================================================
# Main loading loop
# Outer loop: conditions (e.g. trade-off levels)
# Middle loop: temperature steps j = 1:num_temps
# Inner loop:  replicate files (one per SLURM array task)
#
# For each temperature step, results are collected across all replicates into flat vectors (rich, Shannon, Simpson) or vectors-of-vectors (sur, ext, r_inv, ϵ, u, m, Eu, Em, Tpu, Tpm), then appended to the condition-level output. 
# Final structure:
#   results["0"].rich[j]    → Vector{Float64} of richness values across replicates at temperature j
#   results["0"].r_inv[j]   → Vector{Vector{Float64}} of invasion rates across replicates at temperature j
# =============================================================================
results = Dict(c.label => init_collectors() for c in conditions)

for c in conditions
    paths = glob("iters_*", c.path)
    @assert length(paths) > 0 "No result files found at $(c.path)"

    out      = results[c.label]
    progress = Progress(length(paths) * num_temps; desc = "Loading $(c.label): ")

    @time for j in 1:num_temps

        # per-temperature collectors — reset at each temperature step
        col_rich    = Float64[];                  col_Shannon = Float64[];              col_Simpson = Float64[]
        col_sur     = Vector{Vector{Float64}}();  col_ext     = Vector{Vector{Float64}}()
        col_r_inv   = Vector{Vector{Float64}}();  col_ϵ       = Vector{Vector{Float64}}()
        col_u       = Vector{Matrix{Float64}}();  col_m       = Vector{Vector{Float64}}()
        col_Eu      = Vector{Vector{Float64}}();  col_Em      = Vector{Vector{Float64}}()
        col_Tpu     = Vector{Vector{Float64}}();  col_Tpm     = Vector{Vector{Float64}}()

        for path in paths
            # load all variables saved by sim_div.jl for this replicate
            @load path all_rich all_Shannon all_Simpson all_sur all_ext all_r_inv all_ϵ all_u all_m all_Eu all_Em all_Tpu all_Tpm

            # extract the j-th temperature step from this replicate and collect
            append!(col_rich,    all_rich[j])
            append!(col_Shannon, all_Shannon[j])
            append!(col_Simpson, all_Simpson[j])
            push!(col_sur,   all_sur[j]);   push!(col_ext,  all_ext[j])
            push!(col_r_inv, all_r_inv[j]); push!(col_ϵ,    all_ϵ[j])
            push!(col_u,     all_u[j]);     push!(col_m,    all_m[j])
            push!(col_Eu,    all_Eu[j]);    push!(col_Em,   all_Em[j])
            push!(col_Tpu,   all_Tpu[j]);   push!(col_Tpm,  all_Tpm[j])

            next!(progress)
        end

        # append this temperature's collected data to the condition-level output
        push!(out.rich,    col_rich);    push!(out.Shannon, col_Shannon); push!(out.Simpson, col_Simpson)
        push!(out.sur,     col_sur);     push!(out.ext,     col_ext)
        push!(out.r_inv,   col_r_inv);   push!(out.ϵ,       col_ϵ)
        push!(out.u,       col_u);       push!(out.m,       col_m)
        push!(out.Eu,      col_Eu);      push!(out.Em,      col_Em)
        push!(out.Tpu,     col_Tpu);     push!(out.Tpm,     col_Tpm)
    end

    # R"library(beepr); beep(sound = 4, expr = NULL)"
end

# =============================================================================
# Save collected results for each condition to JLD2
# One file per condition: summary_0.jld2, summary_re.jld2, summary_1.jld2
# =============================================================================
save_dir = "../../results/20260327/"
mkpath(save_dir)
 
for c in conditions
    out  = results[c.label]
    path = joinpath(save_dir, "summary_$(c.label).jld2")
 
    rich   = out.rich
    Shannon = out.Shannon
    Simpson = out.Simpson
    sur    = out.sur
    ext    = out.ext
    r_inv  = out.r_inv
    ϵ      = out.ϵ
    u      = out.u
    m      = out.m
    Eu     = out.Eu
    Em     = out.Em
    Tpu    = out.Tpu
    Tpm    = out.Tpm

    # @save path rich Shannon Simpson sur ext r_inv ϵ u m Eu Em Tpu Tpm 
    println("Saved $(c.label) → $path")
end
 
# =============================================================================
# Summary statistics: mean and variance across replicates at each temperature
#
# For each condition and each temperature step j, compute:
#   - richness, Shannon, Simpson      : mean and variance across replicates
#   - u_sum (total uptake per species): mean and variance across all species
#                                       in all replicates at that temperature
#   - m, ϵ, Eu, Em, Tpu, Tpm        : mean and variance pooled across species
#                                       and replicates at that temperature
#   - r_inv                           : mean and variance pooled across species
#                                       and replicates
#
# Output structure (vectors of length num_temps):
#   stats["re"].rich_mean[j]    — mean richness at temperature j
#   stats["re"].u_sum_mean[j]   — mean total uptake at temperature j
#   etc.
# =============================================================================

function summarise(out, num_temps)
    # helpers: pool a vector-of-vectors into one flat vector then summarise
    pool(vv)  = filter(isfinite, vcat(vv...))
    vmean(vv)     = mean(pool(vv))
    vvar(vv)      = var(pool(vv))

    # u_sum: row sums of each N×M uptake matrix → total uptake per species
    u_sum_all(col_u) = vcat([vec(sum(U, dims=2)) for U in col_u]...)
    r_inv_pool(vv) = filter(x -> isfinite(x) && abs(x) < 500.0, vcat(vv...))
    (
        rich_mean    = [mean(out.rich[j])            for j in 1:num_temps],
        rich_var     = [var(out.rich[j])             for j in 1:num_temps],
        Shannon_mean = [mean(out.Shannon[j])         for j in 1:num_temps],
        Shannon_var  = [var(out.Shannon[j])          for j in 1:num_temps],
        Simpson_mean = [mean(out.Simpson[j])         for j in 1:num_temps],
        Simpson_var  = [var(out.Simpson[j])          for j in 1:num_temps],
        u_sum_mean   = [mean(u_sum_all(out.u[j]))    for j in 1:num_temps],
        u_sum_var    = [var(u_sum_all(out.u[j]))     for j in 1:num_temps],
        m_mean       = [vmean(out.m[j])              for j in 1:num_temps],
        m_var        = [vvar(out.m[j])               for j in 1:num_temps],
        ϵ_mean       = [vmean(out.ϵ[j])              for j in 1:num_temps],
        ϵ_var        = [vvar(out.ϵ[j])               for j in 1:num_temps],
        r_inv_mean = [mean(r_inv_pool(out.r_inv[j])) for j in 1:num_temps],
        r_inv_var  = [var(r_inv_pool(out.r_inv[j]))  for j in 1:num_temps],
        Eu_mean      = [vmean(out.Eu[j])             for j in 1:num_temps],
        Eu_var       = [vvar(out.Eu[j])              for j in 1:num_temps],
        Em_mean      = [vmean(out.Em[j])             for j in 1:num_temps],
        Em_var       = [vvar(out.Em[j])              for j in 1:num_temps],
        Tpu_mean     = [vmean(out.Tpu[j])            for j in 1:num_temps],
        Tpu_var      = [vvar(out.Tpu[j])             for j in 1:num_temps],
        Tpm_mean     = [vmean(out.Tpm[j])            for j in 1:num_temps],
        Tpm_var      = [vvar(out.Tpm[j])             for j in 1:num_temps],
    )
end

stats = Dict(c.label => summarise(results[c.label], num_temps) for c in conditions)

# Save summary statistics — one file per condition
for c in conditions
    s    = stats[c.label]
    path = joinpath(save_dir, "stats_$(c.label).jld2")

    jldsave(path;
        rich_mean    = s.rich_mean,    rich_var    = s.rich_var,
        Shannon_mean = s.Shannon_mean, Shannon_var = s.Shannon_var,
        Simpson_mean = s.Simpson_mean, Simpson_var = s.Simpson_var,
        u_sum_mean   = s.u_sum_mean,   u_sum_var   = s.u_sum_var,
        m_mean       = s.m_mean,       m_var       = s.m_var,
        ϵ_mean       = s.ϵ_mean,       ϵ_var       = s.ϵ_var,
        r_inv_mean   = s.r_inv_mean,   r_inv_var   = s.r_inv_var,
        Eu_mean      = s.Eu_mean,      Eu_var      = s.Eu_var,
        Em_mean      = s.Em_mean,      Em_var      = s.Em_var,
        Tpu_mean     = s.Tpu_mean,     Tpu_var     = s.Tpu_var,
        Tpm_mean     = s.Tpm_mean,     Tpm_var     = s.Tpm_var,
    )
    println("Saved stats $(c.label) → $path")
end

# =============================================================================
# Save raw richness values only — one file per condition
# Structure: rich_all[j] is a Vector{Float64} of all replicate richness values
#            at temperature step j, ready for distribution-level comparison
# =============================================================================
for c in conditions
    out  = results[c.label]
    path = joinpath(save_dir, "richness_$(c.label).jld2")
    rich_all = out.rich   # Vector{Vector{Float64}}, length = num_temps
    jldsave(path; rich_all)
    println("Saved richness $(c.label) → $path")
end

# =============================================================================
# Results are accessed by condition label, e.g.:
#   results["0"].rich[j]     — richness across replicates at temperature step j
#   results["re"].r_inv[j]   — invasion growth rates at temperature step j
#   results["1"].Eu[j]       — uptake activation energies at temperature step j
#
# To reload a condition:
#   @load "../../results/20260327/summary_re.jld2" rich Shannon Simpson sur ext r_inv ϵ u m Eu Em Tpu Tpm
#
# To reload summary statistics:
#   @load "../../results/20260327/stats_re.jld2" rich_mean rich_var Shannon_mean Shannon_var Simpson_mean Simpson_var u_sum_mean u_sum_var m_mean m_var ϵ_mean ϵ_var r_inv_mean r_inv_var Eu_mean Eu_var Em_mean Em_var Tpu_mean Tpu_var Tpm_mean Tpm_var
# =============================================================================