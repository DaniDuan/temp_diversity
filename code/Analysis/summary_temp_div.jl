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
    (label = "0",  path = "../../results/20260325/temp_rich_0"),
    (label = "re", path = "../../results/20260325/temp_rich"),
    (label = "1",  path = "../../results/20260325/temp_rich_1"),
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

    R"library(beepr); beep(sound = 4, expr = NULL)"
end

# =============================================================================
# Save collected results for each condition to JLD2
# One file per condition: summary_0.jld2, summary_re.jld2, summary_1.jld2
# =============================================================================
save_dir = "../../results/20260325/"
mkpath(save_dir)
 
for c in conditions
    out  = results[c.label]
    path = joinpath(save_dir, "summary_$(c.label).jld2")
 
    @save path out.rich out.Shannon out.Simpson out.sur out.ext out.r_inv out.ϵ out.u out.m out.Eu out.Em out.Tpu out.Tpm
 
    println("Saved $(c.label) → $path")
end
 
# =============================================================================
# Results are accessed by condition label, e.g.:
#   results["0"].rich[j]     — richness across replicates at temperature step j
#   results["re"].r_inv[j]   — invasion growth rates at temperature step j
#   results["1"].Eu[j]       — uptake activation energies at temperature step j
#
# To reload a condition:
#   @load "../../results/20260325/summary_re.jld2" rich Shannon Simpson sur ext r_inv ϵ u m Eu Em Tpu Tpm
# =============================================================================