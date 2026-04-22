# =============================================================================
# plot_diversity_temp.jl
#
# Plots MiCRM richness, Shannon, and Simpson diversity against temperature.
# Loads:
#   stats_re.jld2      — richness/Shannon/Simpson mean/var
# =============================================================================

using CairoMakie, JLD2, Statistics

# =============================================================================
# Load MiCRM summary statistics
# =============================================================================
@load "../results/stats_re.jld2" rich_mean rich_var Shannon_mean Shannon_var Simpson_mean Simpson_var u_sum_mean u_sum_var m_mean m_var ϵ_mean ϵ_var r_inv_mean r_inv_var Eu_mean Eu_var Em_mean Em_var Tpu_mean Tpu_var Tpm_mean Tpm_var

n_reps    = 999
num_temps = 31
T_axis    = range(0, num_temps - 1, length = num_temps)  # 0–30°C

rich_se    = sqrt.(rich_var)./ sqrt(n_reps)
Shannon_se = sqrt.(Shannon_var)./ sqrt(n_reps)
Simpson_se = sqrt.(Simpson_var)./ sqrt(n_reps)

# =============================================================================
# Figure layout: main axis (col 1) | right outer Simpson label (col 2)
# =============================================================================
fig = Figure(fontsize = 30, size = (1000, 700))

# ── Main axis: Richness (left y) ───────────────────────────────────────────
ax1 = Axis(fig[1, 1],
    xlabel            = "Temperature (°C)",
    ylabel            = "MiCRM Richness",
    xlabelsize        = 40,
    ylabelsize        = 40,
    ygridvisible      = true,
    xgridvisible      = true,
    rightspinevisible = false,
    topspinevisible   = false,
)

# ── Overlay axis: Shannon (right y) ───────────────────────────────────────
ax2 = Axis(fig[1, 1],
    ylabel                 = "Shannon Diversity",
    ylabelsize             = 40,
    yaxisposition          = :right,
    yticklabelalign        = (:left, :center),
    xticklabelsvisible     = false,
    xlabelvisible          = false,
    xticksvisible          = false,
    leftspinevisible       = false,
    bottomspinevisible     = false,
    topspinevisible        = false,
    xgridvisible           = false,
    ygridvisible           = false,
)

# ── Overlay axis: Simpson data only (ticks/label on ax3b) ─────────────────
ax3 = Axis(fig[1, 1],
    yaxisposition          = :right,
    xticklabelsvisible     = false,
    xlabelvisible          = false,
    xticksvisible          = false,
    leftspinevisible       = false,
    bottomspinevisible     = false,
    topspinevisible        = false,
    rightspinevisible      = false,
    yticksvisible          = false,
    yticklabelsvisible     = false,
    xgridvisible           = false,
    ygridvisible           = false,
)

# ── Right outer axis: Simpson ticks + label ───────────────────────────────
ax3b = Axis(fig[1, 2],
    ylabel                 = "Simpson Diversity",
    ylabelsize             = 40,
    yaxisposition          = :right,
    yticklabelalign        = (:left, :center),
    leftspinevisible       = true,
    bottomspinevisible     = false,
    topspinevisible        = false,
    rightspinevisible      = false,
    xticklabelsvisible     = false,
    xlabelvisible          = false,
    xticksvisible          = false,
    xgridvisible           = false,
    ygridvisible           = false,
    yticklabelpad          = 4f0,
)
colsize!(fig.layout, 2, Fixed(0))

# Link axes
linkyaxes!(ax3, ax3b)
linkxaxes!(ax1, ax2, ax3)

# =============================================================================
# Plot MiCRM lines
# =============================================================================
lines!(ax1, T_axis, rich_mean,
    color = ("#6B8EDE", 0.9), linewidth = 5)
band!(ax1,  T_axis, rich_mean.- rich_se, rich_mean.+ rich_se,
    color = ("#6B8EDE", 0.15))

lines!(ax2, T_axis, Shannon_mean,
    color = ("#4F363E", 0.9), linewidth = 5)
band!(ax2,  T_axis, Shannon_mean.- Shannon_se, Shannon_mean.+ Shannon_se,
    color = ("#4F363E", 0.15))

lines!(ax3,  T_axis, Simpson_mean,
    color = ("#7BA9BE", 0.9), linewidth = 5)
band!(ax3,   T_axis, Simpson_mean.- Simpson_se, Simpson_mean.+ Simpson_se,
    color = ("#7BA9BE", 0.15))

# Invisible sync line so ax3b adopts the same y range as ax3
lines!(ax3b, T_axis, Simpson_mean,
    color = (:white, 0.0), linewidth = 0)

# =============================================================================
# Legend
# =============================================================================
l1 = LineElement(color = ("#6B8EDE", 0.9), linewidth = 5)
l2 = LineElement(color = ("#4F363E", 0.9), linewidth = 5)
l3 = LineElement(color = ("#7BA9BE", 0.9), linewidth = 5)

Legend(fig[1, 1], [l1, l2, l3],
    ["Richness", "Shannon", "Simpson"],
    tellheight = false, tellwidth = false,
    halign = :right, valign = :top, framevisible = false)

# # =============================================================================
# # Panel label
# # =============================================================================
# Label(fig[1, 1, TopLeft()], "(a)", fontsize = 36, font = :bold, padding = (0, 0, 8, 0))

# =============================================================================
# Save
# =============================================================================
save("../results/diversity_temp_all.pdf", fig)
println("Saved →../results/diversity_temp_all.pdf")
fig