# =============================================================================
# plot_metabolic_strategies.jl
#
# Thermal metabolic strategies drive community assembly
#
# All panels use condition "0" (no B0-E correlation).
# r_inv values capped at |r_inv| < 200 to exclude numerical artefacts.
#
# Loads:
#   metabolic_0.jld2  — all derived stats (produced by summary_temp_div.jl)
#   stats_0.jld2      — richness mean/se
# =============================================================================

using CairoMakie, JLD2, Statistics

N         = 50
num_temps = 31
T_axis    = collect(range(0, num_temps - 1, length = num_temps))

# =============================================================================
# Load derived data
# =============================================================================
@load "../results/metabolic_0.jld2" u_sur_mean u_sur_se u_ext_mean u_ext_se m_sur_mean m_sur_se m_ext_mean m_ext_se r_inv_sur_mean r_inv_sur_se r_inv_ext_mean r_inv_ext_se frac_pos r_inv_T0 r_inv_T15 r_inv_T30 sc_usum_T15 sc_m_T15 sc_rinv_T15
@load "../results/stats_0.jld2" rich_mean rich_var

n_reps  = 999
rich_se = sqrt.(rich_var) ./ sqrt(n_reps)

T_keys   = [0, 15, 30]
colors_T = ["#4575B4", "#6B8EDE", "#D73027"]

# =============================================================================
# Figure layout: 2×2
# =============================================================================
fig = Figure(fontsize = 28, size = (2700, 800));

# ── Panel A: uᵢ and mᵢ survivors vs extinct ───────────────────────────────
axA1 = Axis(fig[1, 1],
    xlabel             = "Temperature (°C)",
    ylabel             = L"Mean uptake rate $u_i$",
    xlabelsize         = 40, ylabelsize = 40,
    ygridvisible       = false, xgridvisible = false,
    rightspinevisible  = false, topspinevisible = false,
)
axA2 = Axis(fig[1, 1],
    ylabel                 = L"Mean maintenance rate $m_i$",
    ylabelsize             = 40,
    yaxisposition          = :right,
    yticklabelalign        = (:left, :center),
    xticklabelsvisible     = false, xlabelvisible = false, xticksvisible = false,
    leftspinevisible       = false, bottomspinevisible = false, topspinevisible = false,
    xgridvisible           = false, ygridvisible = false,
)
linkxaxes!(axA1, axA2)

lines!(axA1, T_axis, u_sur_mean, color = ("#FA8328", 1.0), linewidth = 5)
band!(axA1,  T_axis, u_sur_mean .- u_sur_se, u_sur_mean .+ u_sur_se, color = ("#FA8328", 0.2))
lines!(axA1, T_axis, u_ext_mean, color = ("#FA8328", 0.4), linewidth = 5)
band!(axA1,  T_axis, u_ext_mean .- u_ext_se, u_ext_mean .+ u_ext_se, color = ("#FA8328", 0.15))

lines!(axA2, T_axis, m_sur_mean, color = ("#015845", 1.0), linewidth = 5)
band!(axA2,  T_axis, m_sur_mean .- m_sur_se, m_sur_mean .+ m_sur_se, color = ("#015845", 0.2))
lines!(axA2, T_axis, m_ext_mean, color = ("#015845", 0.4), linewidth = 5)
band!(axA2,  T_axis, m_ext_mean .- m_ext_se, m_ext_mean .+ m_ext_se, color = ("#015845", 0.15))

l_us = LineElement(color = ("#FA8328", 1.0), linewidth = 5)
l_ue = LineElement(color = ("#FA8328", 0.4), linewidth = 5)
l_ms = LineElement(color = ("#015845", 1.0), linewidth = 5)
l_me = LineElement(color = ("#015845", 0.4), linewidth = 5)
Legend(fig[1, 1], [l_us, l_ue, l_ms, l_me],
    ["Survivor uptake", "Extinct uptake", "Survivor maintenance", "Extinct maintenance"],
    tellheight = false, tellwidth = false,
    halign = :left, valign = :top, framevisible = false, labelsize = 30)
Label(fig[1, 1, TopLeft()], "(a)", fontsize = 40, font = :bold, padding = (0, 0, 8, 0))

# Filled area above zero (r_inv > 0) = species that can invade = survivors
# ── Panel B: u_sum vs m scatter at 15°C, coloured by r_inv sign ────────────
# Blue = r_inv > 0 (can invade), orange = r_inv <= 0 (excluded)
# Shows how the balance of uptake and maintenance determines invasion outcome
# at peak richness temperature (15°C).

col_pos = "#EF8F8C"   # can invade — survivor colour
col_neg = "#4F363E"   # excluded — extinct colour

axB = Axis(fig[1, 2],
    title             = "15°C",
    titlesize         = 40,
    xlabel            = L"Uptake rate $u_i$",
    ylabel            = L"Maintenance rate $m_i$",
    xlabelsize        = 40, ylabelsize = 40,
    ygridvisible      = false, xgridvisible = false,
    topspinevisible   = false, rightspinevisible = false,
)

valid = isfinite.(sc_rinv_T15)
pos   = valid .& (sc_rinv_T15 .> 0)
neg   = valid .& (sc_rinv_T15 .<= 0)

scatter!(axB, sc_usum_T15[neg], sc_m_T15[neg],
    color = (col_neg, 0.2), markersize = 14, label = L"$r_{inv} \leq 0$")
scatter!(axB, sc_usum_T15[pos], sc_m_T15[pos],
    color = (col_pos, 0.7), markersize = 14, label = L"$r_{inv} > 0$")

axislegend(axB, position = :rt, framevisible = false, labelsize = 35)
Label(fig[1, 2, TopLeft()], "(b)", fontsize = 40, font = :bold, padding = (0, 0, 8, 0))

# ── Panel C: mean r_inv survivors vs extinct ───────────────────────────────
axC = Axis(fig[1, 3],
    xlabel            = "Temperature (°C)",
    ylabel            = L"Mean $r_{inv}$",
    xlabelsize        = 40, ylabelsize = 40,
    ygridvisible      = false, xgridvisible = false,
    topspinevisible   = false, rightspinevisible = false,
)
hlines!(axC, [0.0], color = (:black, 0.4), linewidth = 1.5, linestyle = :dash)

lines!(axC, T_axis, r_inv_sur_mean, color = ("#EF8F8C", 1.0), linewidth = 5, label = "Survivors")
band!(axC,  T_axis, r_inv_sur_mean .- r_inv_sur_se, r_inv_sur_mean .+ r_inv_sur_se,
      color = ("#EF8F8C", 0.2))
lines!(axC, T_axis, r_inv_ext_mean, color = ("#4F363E", 0.6), linewidth = 5, label = "Extinct")
band!(axC,  T_axis, r_inv_ext_mean .- r_inv_ext_se, r_inv_ext_mean .+ r_inv_ext_se,
      color = ("#4F363E", 0.2))

axislegend(axC, position = :lt, framevisible = false, labelsize = 30)
Label(fig[1, 3, TopLeft()], "(c)", fontsize = 40, font = :bold, padding = (0, 0, 8, 0))

# =============================================================================
# Save
# =============================================================================
save("../results/metabolic_strategies.pdf", fig)
println("Saved → ../results/metabolic_strategies.pdf")
fig
