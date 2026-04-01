# =============================================================================
# plot_tradeoff_richness.jl
#
# Generates a four-panel publication-quality figure (2 rows × 2 cols):
#
# Row 1 (panels a, b) : minimal trade-off  (ρ_t = [0, 0])
# Row 2 (panels c, d) : maximal trade-off  (ρ_t = [-0.9999, -0.9999])
#   Left  col : B0/E scatter + TPC curves
#   Right col : richness + fraction r_inv > 0
#
# Loads:
#   metabolic_0.jld2, stats_0.jld2   — minimal trade-off
#   metabolic_1.jld2, stats_1.jld2   — maximal trade-off
# =============================================================================

using CairoMakie, JLD2, Statistics, Distributions, LinearAlgebra

# =============================================================================
# Load simulation results
# =============================================================================
@load "../results/metabolic_0.jld2" frac_pos
@load "../results/stats_0.jld2" rich_mean rich_var
frac_pos_0  = frac_pos
rich_mean_0 = rich_mean
rich_se_0   = sqrt.(rich_var) ./ sqrt(999)

@load "../results/metabolic_1.jld2" frac_pos
@load "../results/stats_1.jld2" rich_mean rich_var
frac_pos_1  = frac_pos
rich_mean_1 = rich_mean
rich_se_1   = sqrt.(rich_var) ./ sqrt(999)

n_reps    = 999
num_temps = 31
T_axis    = collect(range(0, num_temps - 1, length = num_temps))

# =============================================================================
# Panel (a): B0/E scatter + TPC curves
# Parameters identical to randtemp_param in temp.jl
# =============================================================================
N_tpc = 50            # number of illustrative species
L_v   = 0.3           # mean leakage (matches L = fill(0.3, N) in simulations)
k     = 0.0000862     # Boltzmann constant (eV/K)
Tr    = 273.15 + 10   # reference temperature (K)
Ed    = 3.5           # high-temperature deactivation energy (eV)
ρ_t   = [0.0000 0.0000]  # maximal B0-E trade-off (matches U_R_var_0 scenario)

# --- TPC parameter distributions (identical to randtemp_param) ---------------
B0_m   = -1.4954
B0_CUE = 0.1953
B0_u   = log(exp(B0_m) / (1 - L_v - B0_CUE))
B0     = [B0_u  B0_m]
B0_var = 0.17 .* abs.(B0)
E_mean = [0.8146  0.5741]
E_var  = 0.1364 .* E_mean
cov_xy = ρ_t .* B0_var .^ 0.5 .* E_var .^ 0.5
meanv  = [B0; E_mean]                          # (2×2): row1 = B0, row2 = E_mean
cov_u  = [B0_var[1] cov_xy[1]; cov_xy[1] E_var[1]]
cov_m  = [B0_var[2] cov_xy[2]; cov_xy[2] E_var[2]]

# temperature axis — full range in Kelvin for TPC evaluation
T_range   = range(273.15, 273.15 + num_temps - 1, length = num_temps)
Temp_rich = collect(range(0, num_temps - 1, length = num_temps))

# --- Sample N_tpc species (same approach as randtemp_param) ------------------
# sample B0 and E jointly for uptake and maintenance
allu = rand(MvNormal(meanv[:, 1], cov_u), N_tpc)   # (2 × N_tpc): row1 = log(B0_u), row2 = E_u
allm = rand(MvNormal(meanv[:, 2], cov_m), N_tpc)   # (2 × N_tpc): row1 = log(B0_m), row2 = E_m
B    = [exp.(allu[1, :])  exp.(allm[1, :])]         # (N_tpc × 2): [B0_u  B0_m]
E    = [allu[2, :]         allm[2, :]]               # (N_tpc × 2): [E_u   E_m]

# peak temperatures: maintenance peaks 3°C above uptake (matches temp.jl)
Tpu = 273.15 .+ rand(Normal(35, 5), N_tpc)
Tpm = Tpu .+ 3
Tp  = [Tpu  Tpm]   # (N_tpc × 2): [Tp_u  Tp_m]

# --- Evaluate Sharpe-Schoolfield TPC across temperature range ----------------
# trait(T) = B * exp(-E/k * (1/T - 1/Tr)) / (1 + E/(Ed-E) * exp(Ed/k * (1/Tp - 1/T)))
# returns log-scaled values for plotting (matching original figure style)
function eval_tpc_log(B_col, E_col, Tp_col, T_range, k, Tr, Ed)
    # B_col, E_col, Tp_col: vectors of length N_tpc
    # returns (N_tpc × num_temps) matrix of log trait values
    log.(
        B_col .* exp.((-E_col ./ k) .* ((1 ./ T_range') .- (1 / Tr))) ./
        (1 .+ (E_col ./ (Ed .- E_col)) .* exp.(Ed / k .* (1 ./ Tp_col .- 1 ./ T_range')))
    )
end

tpc_u = eval_tpc_log(B[:, 1], E[:, 1], Tp[:, 1], T_range, k, Tr, Ed)
tpc_m = eval_tpc_log(B[:, 2], E[:, 2], Tp[:, 2], T_range, k, Tr, Ed)

# --- Maximal trade-off TPC sample (ρ_t = -0.9999) ----------------------------
ρ_t1    = [-0.9999 -0.9999]
cov_xy1 = ρ_t1 .* B0_var .^ 0.5 .* E_var .^ 0.5
cov_u1  = [B0_var[1] cov_xy1[1]; cov_xy1[1] E_var[1]]
cov_m1  = [B0_var[2] cov_xy1[2]; cov_xy1[2] E_var[2]]
allu1   = rand(MvNormal(meanv[:, 1], cov_u1), N_tpc)
allm1   = rand(MvNormal(meanv[:, 2], cov_m1), N_tpc)
B1      = [exp.(allu1[1, :])  exp.(allm1[1, :])]
E1      = [allu1[2, :]         allm1[2, :]]
Tpu1    = 273.15 .+ rand(Normal(35, 5), N_tpc)
Tp1     = [Tpu1  Tpu1 .+ 3]
tpc_u1  = eval_tpc_log(B1[:, 1], E1[:, 1], Tp1[:, 1], T_range, k, Tr, Ed)
tpc_m1  = eval_tpc_log(B1[:, 2], E1[:, 2], Tp1[:, 2], T_range, k, Tr, Ed)

# =============================================================================
# Figure layout: two columns
#   col 1 : panel (a) — B0/E scatter + TPC curves (2×2 sub-grid)
#   col 2 : panel (b) — richness + frac_pos
# =============================================================================
fig = Figure(fontsize = 30, size = (2400, 1800));

colsize!(fig.layout, 1, Relative(0.52))
colsize!(fig.layout, 0, Fixed(30))
# ── Panel (a): 2×2 sub-grid of scatter + TPC axes ─────────────────────────
# Row 1: uptake  (B0 scatter | TPC curves)
# Row 2: maintenance (B0 scatter | TPC curves)
ax1 = Axis(fig[1, 1][1, 1],
    xlabel     = L"\log(B_{0,u})",  ylabel = L"E_u",
    xlabelsize = 30, ylabelsize = 30,
    rightspinevisible = false, topspinevisible = false,
)
ax2 = Axis(fig[1, 1][1, 2],
    xlabel     = "Temperature (°C)", ylabel = L"Uptake rate $log(u_i)$",
    xlabelsize = 30, ylabelsize = 30,
    rightspinevisible = false, topspinevisible = false,
)
ax3 = Axis(fig[1, 1][2, 1],
    xlabel     = L"\log(B_{0,m})",  ylabel = L"E_m",
    xlabelsize = 30, ylabelsize = 30,
    rightspinevisible = false, topspinevisible = false,
)
ax4 = Axis(fig[1, 1][2, 2],
    xlabel     = "Temperature (°C)", ylabel = L"Maintenance rate $log(m_i)$",
    xlabelsize = 30, ylabelsize = 30,
    rightspinevisible = false, topspinevisible = false,
)

for i in 1:N_tpc
    scatter!(ax1, [log(B[i, 1])], [E[i, 1]], color = ("#FA8328", 1.0), markersize = 20)
    lines!(  ax2, Temp_rich, tpc_u[i, :],    color = ("#FA8328", 0.75), linewidth = 2)
    scatter!(ax3, [log(B[i, 2])], [E[i, 2]], color = ("#015845", 1.0), markersize = 20)
    lines!(  ax4, Temp_rich, tpc_m[i, :],    color = ("#015845", 0.75), linewidth = 2)
end

Label(fig[1, 0, Top()], "(a)", fontsize = 38, font = :bold, padding = (0, 0, 8, 0))
Label(fig[1, 0], "Minimal Trade-off",
    fontsize  = 40,
    # font      = :italic,
    rotation  = pi/2,
    padding   = (0, 0, 0, 0),
)

# ── Right: richness + fraction r_inv > 0 ──────────────────────────────────
ax1 = Axis(fig[1, 2],
    xlabel             = "Temperature (°C)",
    ylabel             = "Community richness",
    xlabelsize         = 30, ylabelsize = 30,
    ygridvisible       = true, xgridvisible = true,
    rightspinevisible  = false, topspinevisible = false,
)
ax2 = Axis(fig[1, 2],
    ylabel                 = L"Fraction $r_{inv} > 0$",
    ylabelsize             = 30,
    yaxisposition          = :right,
    yticklabelalign        = (:left, :center),
    xticklabelsvisible     = false, xlabelvisible = false, xticksvisible = false,
    leftspinevisible       = false, bottomspinevisible = false, topspinevisible = false,
    xgridvisible           = false, ygridvisible = false,
)
linkxaxes!(ax1, ax2)

lines!(ax1, T_axis, rich_mean_0, color = ("#6B8EDE", 0.9), linewidth = 5, linestyle = :dash)
band!(ax1,  T_axis, rich_mean_0 .- rich_se_0, rich_mean_0 .+ rich_se_0, color = ("#6B8EDE", 0.2))

lines!(ax2, T_axis, frac_pos_0, color = ("#EF8F8C", 0.9), linewidth = 5)

l1 = LineElement(color = ("#6B8EDE", 0.9), linewidth = 5)
l2 = LineElement(color = ("#EF8F8C", 0.9), linewidth = 5)
Legend(fig[1, 2], [l1, l2], ["Richness", L"Fraction $r_{inv} > 0$"],
    tellheight = false, tellwidth = false,
    halign = :right, valign = :top, framevisible = false, labelsize = 30)
Label(fig[1, 2, TopLeft()], "(b)", fontsize = 38, font = :bold, padding = (0, 0, 8, 0))


# =============================================================================
# Row 2: maximal trade-off (ρ_t = -0.9999)
# =============================================================================
ax5 = Axis(fig[2, 1][1, 1],
    xlabel     = L"\log(B_{0,u})",  ylabel = L"E_u",
    xlabelsize = 30, ylabelsize = 30,
    rightspinevisible = false, topspinevisible = false,
)
ax6 = Axis(fig[2, 1][1, 2],
    xlabel     = "Temperature (°C)", ylabel = L"Uptake rate $log(u_i)$",
    xlabelsize = 30, ylabelsize = 30,
    rightspinevisible = false, topspinevisible = false,
)
ax7 = Axis(fig[2, 1][2, 1],
    xlabel     = L"\log(B_{0,m})",  ylabel = L"E_m",
    xlabelsize = 30, ylabelsize = 30,
    rightspinevisible = false, topspinevisible = false,
)
ax8 = Axis(fig[2, 1][2, 2],
    xlabel     = "Temperature (°C)", ylabel = L"Maintenance rate $log(m_i)$",
    xlabelsize = 30, ylabelsize = 30,
    rightspinevisible = false, topspinevisible = false,
)

for i in 1:N_tpc
    scatter!(ax5, [log(B1[i, 1])], [E1[i, 1]], color = ("#FA8328", 1.0), markersize = 20)
    lines!(  ax6, Temp_rich, tpc_u1[i, :],     color = ("#FA8328", 0.75), linewidth = 2)
    scatter!(ax7, [log(B1[i, 2])], [E1[i, 2]], color = ("#015845", 1.0), markersize = 20)
    lines!(  ax8, Temp_rich, tpc_m1[i, :],     color = ("#015845", 0.75), linewidth = 2)
end

Label(fig[2, 0, Top()], "(c)", fontsize = 38, font = :bold, padding = (0, 0, 8, 0))
Label(fig[2, 0], "Maximal Trade-off",
    fontsize  = 40,
    rotation  = pi/2,
    padding   = (0, 0, 0, 0),
)

# ── Row 2 right: richness + fraction r_inv > 0 ────────────────────────────
ax9  = Axis(fig[2, 2],
    xlabel             = "Temperature (°C)",
    ylabel             = "Community richness",
    xlabelsize         = 30, ylabelsize = 30,
    ygridvisible       = true, xgridvisible = true,
    rightspinevisible  = false, topspinevisible = false,
)
ax10 = Axis(fig[2, 2],
    ylabel                 = L"Fraction $r_{inv} > 0$",
    ylabelsize             = 30,
    yaxisposition          = :right,
    yticklabelalign        = (:left, :center),
    xticklabelsvisible     = false, xlabelvisible = false, xticksvisible = false,
    leftspinevisible       = false, bottomspinevisible = false, topspinevisible = false,
    xgridvisible           = false, ygridvisible = false,
)
linkxaxes!(ax9, ax10)

lines!(ax9,  T_axis, rich_mean_1, color = ("#6B8EDE", 0.9), linewidth = 5, linestyle = :dash)
band!(ax9,   T_axis, rich_mean_1 .- rich_se_1, rich_mean_1 .+ rich_se_1, color = ("#6B8EDE", 0.2))
lines!(ax10, T_axis, frac_pos_1,  color = ("#EF8F8C", 0.9), linewidth = 5)

l1 = LineElement(color = ("#6B8EDE", 0.9), linewidth = 5)
l2 = LineElement(color = ("#EF8F8C", 0.9), linewidth = 5)
Legend(fig[2, 2], [l1, l2], ["Richness", L"Fraction $r_{inv} > 0$"],
    tellheight = false, tellwidth = false,
    halign = :right, valign = :top, framevisible = false, labelsize = 30)
Label(fig[2, 2, TopLeft()], "(d)", fontsize = 38, font = :bold, padding = (0, 0, 8, 0))

# Fix both row heights after all content is placed
rowsize!(fig.layout, 1, Fixed(750))
rowsize!(fig.layout, 2, Fixed(750))

# =============================================================================
# Save
# =============================================================================
save("../results/tradeoff_richness.pdf", fig)
println("Saved → ../results/tradeoff_richness.pdf")
fig
