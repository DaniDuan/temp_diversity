# =============================================================================
# plot_diversity_temp.jl
#
# Plots MiCRM richness, Shannon, Simpson, and EMP richness against temperature.
# Loads:
#   stats_re.jld2      — richness mean/se

# =============================================================================

using CairoMakie, JLD2, Statistics, CSV, DataFrames, StatsBase, Interpolations

# =============================================================================
# Load MiCRM summary statistics
# =============================================================================
@load "../results/stats_re.jld2" rich_mean rich_var Shannon_mean Shannon_var Simpson_mean Simpson_var u_sum_mean u_sum_var m_mean m_var ϵ_mean ϵ_var r_inv_mean r_inv_var Eu_mean Eu_var Em_mean Em_var Tpu_mean Tpu_var Tpm_mean Tpm_var

N = 50; M = 50
n_reps    = 999
num_temps = 31
T_axis    = range(0, num_temps - 1, length = num_temps)  # 0–30°C

rich_se    = sqrt.(rich_var)    ./ sqrt(n_reps)
Shannon_se = sqrt.(Shannon_var) ./ sqrt(n_reps)
Simpson_se = sqrt.(Simpson_var) ./ sqrt(n_reps)

# =============================================================================
# Load and bin EMP data (same processing as EMP_rich.jl)
# =============================================================================
EMP_data = CSV.read("../data/EMP_filtered.csv", DataFrame, header = true)
EMP_data.round_Temp = round.(Int, EMP_data.Temp)

bins        = sort(unique(EMP_data.round_Temp))
EMP_meanerr = zeros(Float64, length(bins), 3)  # [Temp, mean, SE]

for (n, bin) in enumerate(bins)
    bin_data           = filter(row -> row.round_Temp == bin, EMP_data)
    EMP_meanerr[n, :] = [bin,
                         mean(bin_data.Richness),
                         std(bin_data.Richness) / sqrt(nrow(bin_data))]
end

EMP_meanerr = DataFrame(EMP_meanerr, ["Temp", "mean", "err"])
sort!(EMP_meanerr, :Temp)

# =============================================================================
# Figure layout:  col 0 (narrow, EMP label) | col 1 (main plot) | col 2 (narrow, Simpson label)
# =============================================================================
fig = Figure(fontsize = 30, size = (1400, 900));

# ── Left outer: EMP richness ticks + label ────────────────────────────────
ax0b = Axis(fig[1, 1],
    ylabel                 = "Richness (EMP)",
    ylabelsize             = 40,
    yaxisposition          = :left,
    yticklabelalign        = (:right, :center),
    rightspinevisible      = false,
    bottomspinevisible     = false,
    topspinevisible        = false,
    xticklabelsvisible     = false,
    xlabelvisible          = false,
    xticksvisible          = false,
    xgridvisible           = false,
    ygridvisible           = false,
    yticklabelpad          = 4f0,
)
colsize!(fig.layout, 1, Fixed(0))

# ── Main plot axes ─────────────────────────────────────────────────────────
ax1 = Axis(fig[1, 2],
    xlabel             = "Temperature (°C)",
    ylabel             = "Richness (MiCRM)",
    xlabelsize         = 40,
    ylabelsize         = 40,
    ygridvisible       = true,
    xgridvisible       = true,
    rightspinevisible  = false,
    topspinevisible    = false,
)

ax2 = Axis(fig[1, 2],
    ylabel                 = "Shannon Diversity (MiCRM)",
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

# ax3: Simpson data only (ticks/label handled by ax3b)
ax3 = Axis(fig[1, 2],
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

# ax0: EMP data only (ticks/label handled by ax0b)
ax0 = Axis(fig[1, 2],
    yaxisposition          = :left,
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

# ── Right outer: Simpson ticks + label ────────────────────────────────────
ax3b = Axis(fig[1, 3],
    ylabel                 = "Simpson Diversity (MiCRM)",
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
colsize!(fig.layout, 3, Fixed(0))

# Link axes
linkyaxes!(ax3,  ax3b)
linkyaxes!(ax0,  ax0b)
linkxaxes!(ax1, ax2, ax3, ax0)

# =============================================================================
# Plot MiCRM lines
# =============================================================================
lines!(ax1, T_axis, rich_mean,    color = ("#6B8EDE", 0.9), linewidth = 5)
band!(ax1,  T_axis, rich_mean .- rich_se, rich_mean .+ rich_se, color = ("#6B8EDE", 0.15))

lines!(ax2, T_axis, Shannon_mean, color = ("#4F363E", 0.9), linewidth = 5)
band!(ax2,  T_axis, Shannon_mean .- Shannon_se, Shannon_mean .+ Shannon_se, color = ("#4F363E", 0.15))

lines!(ax3,  T_axis, Simpson_mean, color = ("#7BA9BE", 0.9), linewidth = 5)
band!(ax3,   T_axis, Simpson_mean .- Simpson_se, Simpson_mean .+ Simpson_se, color = ("#7BA9BE", 0.15))
lines!(ax3b, T_axis, Simpson_mean, color = (:white, 0.0), linewidth = 0)

# =============================================================================
# Plot EMP scatter + SE bars
# =============================================================================
scatter!(ax0, EMP_meanerr.Temp, EMP_meanerr.mean,
    color = ("#E17542", 1.0), markersize = 14)

cap = 0.15
for (x, y, e) in zip(EMP_meanerr.Temp, EMP_meanerr.mean, EMP_meanerr.err)
    lines!(ax0, [x, x],             [y - e, y + e],         color = "#E17542", linewidth = 2)
    lines!(ax0, [x - cap, x + cap], [y - e, y - e],         color = "#E17542", linewidth = 2)
    lines!(ax0, [x - cap, x + cap], [y + e, y + e],         color = "#E17542", linewidth = 2)
end

# invisible sync line so ax0b adopts the same y range
scatter!(ax0b, EMP_meanerr.Temp, EMP_meanerr.mean,
    color = (:white, 0.0), markersize = 0)

# =============================================================================
# Legend
# =============================================================================
l1  = LineElement(color = ("#6B8EDE", 0.9), linewidth = 5)
l2  = LineElement(color = ("#4F363E", 0.9), linewidth = 5)
l3  = LineElement(color = ("#7BA9BE", 0.9), linewidth = 5)
s_e = MarkerElement(color = ("#E17542", 1.0), markersize = 14, marker = :circle)
Legend(fig[1, 2], [s_e, l1, l2, l3,],
    ["Richness (EMP)", "Richness (MiCRM)", "Shannon (MiCRM)", "Simpson (MiCRM)"],
    tellheight = false, tellwidth = false,
    halign = :right, valign = :top, framevisible = false)

# =============================================================================
# Peak temperature markers on ax1 and ax0
# =============================================================================
T_peak_micrm = Float64(collect(T_axis)[argmax(rich_mean)])
T_peak_emp   = Float64(EMP_meanerr.Temp[argmax(EMP_meanerr.mean)])

vlines!(ax1, [T_peak_micrm],
    color = ("#6B8EDE", 0.9), linewidth = 2.5, linestyle = :dash)
vlines!(ax0, [T_peak_emp],
    color = ("#E17542", 0.9), linewidth = 2.5, linestyle = :dash)

text!(ax1, T_peak_micrm + 0.3, minimum(rich_mean) + 0.05 * (maximum(rich_mean) - minimum(rich_mean)),
    text = "$(Int(T_peak_micrm))°C", color = "#6B8EDE", fontsize = 24, align = (:left, :bottom))
text!(ax0, T_peak_emp - 0.3, minimum(EMP_meanerr.mean) + 0.05 * (maximum(EMP_meanerr.mean) - minimum(EMP_meanerr.mean)),
    text = "$(Int(T_peak_emp))°C", color = "#E17542", fontsize = 24, align = (:right, :bottom))

# # =============================================================================
# # Normalise mean richness curves for scatter panel
# # =============================================================================
# emp_itp     = linear_interpolation(EMP_meanerr.Temp, EMP_meanerr.mean,
#                                    extrapolation_bc = NaN)
# emp_err_itp = linear_interpolation(EMP_meanerr.Temp, EMP_meanerr.err,
#                                    extrapolation_bc = NaN)
# T_all       = collect(T_axis)
# emp_interp  = emp_itp.(T_all)
# emp_err_int = emp_err_itp.(T_all)

# overlap     = .!isnan.(emp_interp)
# T_ov        = T_all[overlap]
# micrm_ov    = rich_mean[overlap]
# micrm_se_ov = rich_se[overlap]
# emp_ov      = emp_interp[overlap]
# emp_se_ov   = emp_err_int[overlap]

# # Normalise means — SE rescaled by same range so bars stay proportional
# micrm_range = maximum(micrm_ov) - minimum(micrm_ov)
# emp_range   = maximum(emp_ov)   - minimum(emp_ov)

# norm01(x)   = (x .- minimum(x)) ./ (maximum(x) .- minimum(x))
# micrm_norm  = norm01(micrm_ov)
# emp_norm    = norm01(emp_ov)
# micrm_se_norm = micrm_se_ov ./ micrm_range
# emp_se_norm   = emp_se_ov   ./ emp_range

# r_pearson   = cor(micrm_norm, emp_norm)
# r_spearman  = corspearman(micrm_norm, emp_norm)

# # =============================================================================
# # Panel B: Scatter — per-temperature means + SE bars, coloured by temperature
# # blue (cold) → white (mid) → red (hot)
# # =============================================================================
# axC = Axis(fig[1, 4],
#     xlabel        = "EMP richness (normalised)",
#     ylabel        = "MiCRM richness (normalised)",
#     xlabelsize    = 36,
#     ylabelsize    = 36,
#     ygridvisible  = false,
#     xgridvisible  = false,
#     topspinevisible   = false,
#     rightspinevisible = false,
#     aspect        = 1,
# )

# xlims!(axC, -0.05, 1.1)
# ylims!(axC, -0.05, 1.1)

# cmap_sc = cgrad(["#4575B4", "#D73027"])
# pt_colors = [cmap_sc[(t - minimum(T_ov)) / (maximum(T_ov) - minimum(T_ov))]
#              for t in T_ov]

# # Error bars
# cap = 0.005   # half-width of caps — proportional to left plot (0.15/30)
# for (x, y, xe, ye, col) in zip(emp_norm, micrm_norm, emp_se_norm, micrm_se_norm, pt_colors)
#     # horizontal bar (EMP SE) + caps
#     lines!(axC, [x - xe, x + xe], [y,      y     ], color = (col, 0.7), linewidth = 1.5)
#     lines!(axC, [x - xe, x - xe], [y - cap, y + cap], color = (col, 0.7), linewidth = 1.5)
#     lines!(axC, [x + xe, x + xe], [y - cap, y + cap], color = (col, 0.7), linewidth = 1.5)
#     # vertical bar (MiCRM SE) + caps
#     lines!(axC, [x,      x     ], [y - ye, y + ye], color = (col, 0.7), linewidth = 1.5)
#     lines!(axC, [x - cap, x + cap], [y - ye, y - ye], color = (col, 0.7), linewidth = 1.5)
#     lines!(axC, [x - cap, x + cap], [y + ye, y + ye], color = (col, 0.7), linewidth = 1.5)
# end

# scatter!(axC, emp_norm, micrm_norm,
#     color       = T_ov,
#     colormap    = cmap_sc,
#     colorrange  = (minimum(T_ov), maximum(T_ov)),
#     markersize  = 14,
#     strokecolor = :grey50,
#     strokewidth = 0.5)

# # 1:1 line — no legend entry
# lines!(axC, [0.0, 1.05], [0.0, 1.05],
#     color = (:black, 0.4), linewidth = 2, linestyle = :dash)

# Colorbar(fig[1, 5],
#     limits     = (minimum(T_ov), maximum(T_ov)),
#     colormap   = cmap_sc,
#     label      = "Temperature (°C)",
#     labelsize  = 26,
#     vertical   = true,
#     tellheight = false)

# text!(axC, 1.08, 0.03,
#     text     = "Pearson r = $(round(r_pearson, digits=3))\nSpearman ρ = $(round(r_spearman, digits=3))",
#     color    = :black,
#     fontsize = 26,
#     align    = (:right, :bottom))

# # =============================================================================
# # Panel labels
# # =============================================================================
# Label(fig[1, 2, TopLeft()], "(a)", fontsize = 36, font = :bold, padding = (0, 0, 8, 0))
# Label(fig[1, 4, TopLeft()], "(b)", fontsize = 36, font = :bold, padding = (0, 0, 8, 0))

# =============================================================================
# Save
# =============================================================================
save("../results/diversity_temp.pdf", fig)
println("Saved → ../results/diversity_temp.pdf")
fig
