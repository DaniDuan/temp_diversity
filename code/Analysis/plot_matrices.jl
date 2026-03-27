# =============================================================================
# plot_matrices.jl
#
# Generates clean heatmaps and TPC curves matching the uploaded SVG style:
#   - Uptake matrix U (N×M): grayscale heatmap, no labels
#   - Leakage matrix L[i] (M×M): grayscale heatmap, no labels
#   - TPC curves: one per species, orange curve, black border, no labels
# =============================================================================

using Distributions, Random
using LinearAlgebra
using Parameters
using StatsBase
using CairoMakie

include(joinpath(@__DIR__, "Base/sim_frame.jl"))

# ── Parameters ────────────────────────────────────────────────────────────────
N            = 5
M            = 5
T            = 15.0 + 273.15
Tr           = 10.0 + 273.15
Ed           = 3.5
ρ_t          = [-0.35 -0.35]
L            = fill(0.3, N)
species_show = 1

Random.seed!(4321)
mkpath(joinpath(@__DIR__, "../results"))

p = generate_params(N, M;
    f_u          = modular_uptake,
    f_l          = modular_leakage,
    f_m          = F_m,
    f_ρ          = F_ρ,
    f_ω          = F_ω,
    f_Kc         = F_Kc,
    N_modules    = 2,
    s_ratio      = 50.0,
    n_byproducts = 2:3,
    s_ratio_l    = 50.0,
    L = L, T = T, ρ_t = ρ_t, Tr = Tr, Ed = Ed,
    input_type   = "constant",
    ω            = fill(0.0, M)
)

# ── Helper: grayscale heatmap ─────────────────────────────────────────────────
function grayscale_heatmap(mat; filepath)
    nrow, ncol = size(mat)
    fig = Figure(size = (600, 600), backgroundcolor = :white)
    ax  = Axis(fig[1, 1],
        aspect             = DataAspect(),
        yreversed          = true,
        leftspinevisible   = true,
        rightspinevisible  = true,
        topspinevisible    = true,
        bottomspinevisible = true,
        xticksvisible      = false,
        yticksvisible      = false,
        xticklabelsvisible = false,
        yticklabelsvisible = false,
        xgridvisible       = false,
        ygridvisible       = false,
    )
    vmin, vmax = minimum(mat), maximum(mat)
    mat_norm   = vmax > vmin ? (mat .- vmin) ./ (vmax - vmin) : zeros(size(mat))
    heatmap!(ax, 1:ncol, 1:nrow, mat_norm', colormap = :grays, colorrange = (0, 1))
    save(filepath, fig)
    println("Saved: $filepath")
    return fig
end

# ── Helper: Sharpe-Schoolfield TPC ───────────────────────────────────────────
function sharpe_schoolfield(T_K, B0, E, Ed, Tp, Tr)
    k = 0.0000862
    B0 * exp((-E / k) * (1/T_K - 1/Tr)) /
        (1 + (E / (Ed - E)) * exp(Ed / k * (1/Tp - 1/T_K)))
end

# ── Helper: TPC curve plot (one species, matching SVG style) ──────────────────
# pre-compute shared y-axis range across all species so all plots are comparable
temps_K = range(273.15, 273.15 + 60, length = 200)   # 0–40°C in Kelvin

all_u_curves = [
    max.(
        [sharpe_schoolfield(T_K, p.B[i,1], p.E[i,1], Ed, p.Tp[i,1], Tr) for T_K in temps_K],
        0.0
    )
    for i in 1:N
]
u_ymax = maximum(maximum.(all_u_curves)) * 1.05   # shared y-axis upper bound
u_ymin = 0.0

function tpc_plot(i, u_curve; filepath)
    fig = Figure(size = (800, 600), backgroundcolor = :white)
    ax  = Axis(fig[1, 1],
        leftspinevisible   = true,
        rightspinevisible  = true,
        topspinevisible    = true,
        bottomspinevisible = true,
        xticksvisible      = false,
        yticksvisible      = false,
        xticklabelsvisible = false,
        yticklabelsvisible = false,
        xgridvisible       = false,
        ygridvisible       = false,
        limits             = (0, 60, u_ymin, u_ymax)   # x in °C, y shared
    )

    # convert temps to °C for x-axis
    lines!(ax, collect(temps_K) .- 273.15, u_curve,
        color     = colorant"#FA8328",
        linewidth = 10
    )

    save(filepath, fig)
    println("Saved: $filepath")
    return fig
end

# ── Plot 1: Uptake matrix ─────────────────────────────────────────────────────
N_show = min(M, N)
fig_u  = grayscale_heatmap(p.u[1:N_show, :],
    filepath = joinpath(@__DIR__, "../results/uptake_matrix.svg")
)

# ── Plot 2: Leakage matrix (species 1) ───────────────────────────────────────
fig_l = grayscale_heatmap(p.l[species_show, :, :],
    filepath = joinpath(@__DIR__, "../results/leakage_matrix.svg")
)

# ── Plot 3: TPC curves — one file per species, shared y-axis ─────────────────
for i in 1:N
    tpc_plot(i, all_u_curves[i],
        filepath = joinpath(@__DIR__, "../results/tpc_species_$(i).svg")
    )
end

println("\nAll done. Files saved to ../results/")
