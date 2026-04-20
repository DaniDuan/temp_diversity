# =============================================================================
# plot_all_div.jl
#
# Plots MiCRM richness, Shannon, Simpson, and ATLAS richness against temperature.
# Loads:
#   stats_re.jld2         — MiCRM richness mean/se
#   summary_atlas.csv     — ATLAS empirical richness data
# =============================================================================

using CairoMakie, JLD2, Statistics, CSV, DataFrames, StatsBase, Interpolations, LsqFit, LinearAlgebra

# =============================================================================
# Load MiCRM summary statistics
# =============================================================================
@load "../results/stats_re.jld2" rich_mean rich_var Shannon_mean Shannon_var Simpson_mean Simpson_var u_sum_mean u_sum_var m_mean m_var ϵ_mean ϵ_var r_inv_mean r_inv_var Eu_mean Eu_var Em_mean Em_var Tpu_mean Tpu_var Tpm_mean Tpm_var

N = 50; M = 50
n_reps    = 999
num_temps = 31
T_axis    = range(0, num_temps - 1, length = num_temps)  # 0–30°C

rich_se    = sqrt.(rich_var)./ sqrt(n_reps)
Shannon_se = sqrt.(Shannon_var)./ sqrt(n_reps)
Simpson_se = sqrt.(Simpson_var)./ sqrt(n_reps)

# =============================================================================
# Load and bin ATLAS data
# =============================================================================
ATLAS_data = CSV.read("../data/summary_atlas.csv", DataFrame, header = true)
ATLAS_data.round_Temp = round.(Int, ATLAS_data.Temperature_C_mean)

bins          = sort(unique(ATLAS_data.round_Temp))
ATLAS_meanerr = zeros(Float64, length(bins), 3)  # [Temp, mean, SE]

for (n, bin) in enumerate(bins)
    bin_data             = filter(row -> row.round_Temp == bin, ATLAS_data)
    ATLAS_meanerr[n, :] = [bin,
                            mean(bin_data.Richness),
                            std(bin_data.Richness) / sqrt(nrow(bin_data))]
end

ATLAS_meanerr = DataFrame(ATLAS_meanerr, ["Temp", "mean", "err"])
sort!(ATLAS_meanerr, :Temp)

# =============================================================================
# Fit Gaussian to ATLAS mean richness — with 95% CI
# =============================================================================
gaussian(x, p) = p[1].* exp.(.-((x.- p[2]).^2)./ (2 .* p[3].^2))

p0_gauss  = [maximum(ATLAS_meanerr.mean),
             ATLAS_meanerr.Temp[argmax(ATLAS_meanerr.mean)],
             5.0]

gauss_fit = curve_fit(gaussian, Float64.(ATLAS_meanerr.Temp),
                      Float64.(ATLAS_meanerr.mean), p0_gauss)

T_peak_atlas_gauss = gauss_fit.param[2]

println("=== Gaussian Fit (ATLAS Mean Richness) ===")
println("  Peak richness (a):        ", round(gauss_fit.param[1], digits=2))
println("  Optimal temperature (mu): ", round(T_peak_atlas_gauss, digits=2), " °C")
println("  Thermal breadth (sigma):  ", round(gauss_fit.param[3], digits=2), " °C")

# Smooth prediction sequence
pred_x        = collect(range(minimum(ATLAS_meanerr.Temp), maximum(ATLAS_meanerr.Temp), length = 300))
gaussian_pred = gaussian(pred_x, gauss_fit.param)

# 95% CI via Monte Carlo parameter sampling from covariance matrix
covar         = estimate_covar(gauss_fit)
n_samples     = 1000
param_samples = [gauss_fit.param.+ cholesky(Hermitian(covar)).L * randn(3) for _ in 1:n_samples]
pred_matrix   = hcat([gaussian(pred_x, p) for p in param_samples]...)
gauss_ci_lo   = [quantile(pred_matrix[i, :], 0.025) for i in 1:length(pred_x)]
gauss_ci_hi   = [quantile(pred_matrix[i, :], 0.975) for i in 1:length(pred_x)]

# =============================================================================
# Figure layout: col 0 (narrow, ATLAS label) | col 1 (main plot) | col 2 (narrow, Simpson label)
# =============================================================================
fig = Figure(fontsize = 30, size = (2400, 900));

# ── Left outer: ATLAS richness ticks + label ──────────────────────────────
ax0b = Axis(fig[1, 1],
    ylabel                 = "Richness (ATLAS)",
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

# ax0: ATLAS data + Gaussian fit (ticks/label handled by ax0b)
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
band!(ax1,  T_axis, rich_mean.- rich_se, rich_mean.+ rich_se, color = ("#6B8EDE", 0.15))

lines!(ax2, T_axis, Shannon_mean, color = ("#4F363E", 0.9), linewidth = 5)
band!(ax2,  T_axis, Shannon_mean.- Shannon_se, Shannon_mean.+ Shannon_se, color = ("#4F363E", 0.15))

lines!(ax3,  T_axis, Simpson_mean, color = ("#7BA9BE", 0.9), linewidth = 5)
band!(ax3,   T_axis, Simpson_mean.- Simpson_se, Simpson_mean.+ Simpson_se, color = ("#7BA9BE", 0.15))
lines!(ax3b, T_axis, Simpson_mean, color = (:white, 0.0), linewidth = 0)

# =============================================================================
# Plot ATLAS scatter + SE bars + Gaussian fit + 95% CI band
# =============================================================================
scatter!(ax0, ATLAS_meanerr.Temp, ATLAS_meanerr.mean,
    color = ("#E17542", 1.0), markersize = 14)

cap = 0.15
for (x, y, e) in zip(ATLAS_meanerr.Temp, ATLAS_meanerr.mean, ATLAS_meanerr.err)
    lines!(ax0, [x, x],             [y - e, y + e],     color = "#E17542", linewidth = 2)
    lines!(ax0, [x - cap, x + cap], [y - e, y - e],     color = "#E17542", linewidth = 2)
    lines!(ax0, [x - cap, x + cap], [y + e, y + e],     color = "#E17542", linewidth = 2)
end

# 95% CI band on Gaussian fit
band!(ax0, pred_x, gauss_ci_lo, gauss_ci_hi,
    color = ("#E17542", 0.15))

# Gaussian fit line
lines!(ax0, pred_x, gaussian_pred,
    color = ("#E17542", 0.9), linewidth = 3, linestyle = :solid)

# invisible sync line so ax0b adopts the same y range
scatter!(ax0b, ATLAS_meanerr.Temp, ATLAS_meanerr.mean,
    color = (:white, 0.0), markersize = 0)

# =============================================================================
# Peak temperature markers — MiCRM on ax1, ATLAS Gaussian peak on ax0
# =============================================================================
T_peak_micrm = Float64(collect(T_axis)[argmax(rich_mean)])

vlines!(ax1, [T_peak_micrm],
    color = ("#6B8EDE", 0.9), linewidth = 2.5, linestyle = :dash)
vlines!(ax0, [T_peak_atlas_gauss],
    color = ("#E17542", 0.9), linewidth = 2.5, linestyle = :dash)

text!(ax1, T_peak_micrm + 0.3, minimum(rich_mean) + 0.05 * (maximum(rich_mean) - minimum(rich_mean)),
    text = "$(Int(round(T_peak_micrm)))°C", color = "#6B8EDE", fontsize = 24, align = (:left, :bottom))
text!(ax0, T_peak_atlas_gauss + 0.3, minimum(ATLAS_meanerr.mean) + 0.15 * (maximum(ATLAS_meanerr.mean) - minimum(ATLAS_meanerr.mean)),
    text = "$(round(T_peak_atlas_gauss, digits=1))°C", color = "#E17542", fontsize = 24, align = (:left, :bottom))

# =============================================================================
# Legend
# =============================================================================
l1  = LineElement(color = ("#6B8EDE", 0.9), linewidth = 5)
l2  = LineElement(color = ("#4F363E", 0.9), linewidth = 5)
l3  = LineElement(color = ("#7BA9BE", 0.9), linewidth = 5)
l4  = LineElement(color = ("#E17542", 0.9), linewidth = 3, linestyle = :solid)
Legend(fig[1, 2], [l4, l1, l2, l3],
    ["Richness (ATLAS)", "Richness (MiCRM)", "Shannon (MiCRM)", "Simpson (MiCRM)"],
    tellheight = false, tellwidth = false,
    halign = :right, valign = :top, framevisible = false)

# =============================================================================
# Normalise mean richness curves for scatter panel
# =============================================================================
atlas_itp     = linear_interpolation(ATLAS_meanerr.Temp, ATLAS_meanerr.mean,
                                     extrapolation_bc = NaN)
atlas_err_itp = linear_interpolation(ATLAS_meanerr.Temp, ATLAS_meanerr.err,
                                     extrapolation_bc = NaN)
T_all         = collect(T_axis)
atlas_interp  = atlas_itp.(T_all)
atlas_err_int = atlas_err_itp.(T_all)

overlap       =.!isnan.(atlas_interp)
T_ov          = T_all[overlap]
micrm_ov      = rich_mean[overlap]
micrm_se_ov   = rich_se[overlap]
atlas_ov      = atlas_interp[overlap]
atlas_se_ov   = atlas_err_int[overlap]

micrm_range   = maximum(micrm_ov)  - minimum(micrm_ov)
atlas_range   = maximum(atlas_ov)  - minimum(atlas_ov)

norm01(x)     = (x.- minimum(x))./ (maximum(x).- minimum(x))
micrm_norm    = norm01(micrm_ov)
atlas_norm    = norm01(atlas_ov)
micrm_se_norm = micrm_se_ov./ micrm_range
atlas_se_norm = atlas_se_ov./ atlas_range

r_pearson  = cor(micrm_norm, atlas_norm)
r_spearman = corspearman(micrm_norm, atlas_norm)

# =============================================================================
# Panel B: Scatter — MiCRM on x, ATLAS on y
# =============================================================================
axC = Axis(fig[1, 4],
    xlabel        = "MiCRM richness (normalised)",
    ylabel        = "ATLAS richness (normalised)",
    xlabelsize    = 36,
    ylabelsize    = 36,
    ygridvisible  = false,
    xgridvisible  = false,
    topspinevisible   = false,
    rightspinevisible = false,
    aspect        = 1,
)

xlims!(axC, -0.05, 1.1)
ylims!(axC, -0.05, 1.1)

cmap_sc   = cgrad(["#4575B4", "#D73027"])
pt_colors = [cmap_sc[(t - minimum(T_ov)) / (maximum(T_ov) - minimum(T_ov))]
             for t in T_ov]

cap = 0.005
for (x, y, xe, ye, col) in zip(micrm_norm, atlas_norm, micrm_se_norm, atlas_se_norm, pt_colors)
    lines!(axC, [x - xe, x + xe], [y,       y      ], color = (col, 0.7), linewidth = 1.5)
    lines!(axC, [x - xe, x - xe], [y - cap, y + cap], color = (col, 0.7), linewidth = 1.5)
    lines!(axC, [x + xe, x + xe], [y - cap, y + cap], color = (col, 0.7), linewidth = 1.5)
    lines!(axC, [x,       x      ], [y - ye, y + ye], color = (col, 0.7), linewidth = 1.5)
    lines!(axC, [x - cap, x + cap], [y - ye, y - ye], color = (col, 0.7), linewidth = 1.5)
    lines!(axC, [x - cap, x + cap], [y + ye, y + ye], color = (col, 0.7), linewidth = 1.5)
end

scatter!(axC, micrm_norm, atlas_norm,
    color       = T_ov,
    colormap    = cmap_sc,
    colorrange  = (minimum(T_ov), maximum(T_ov)),
    markersize  = 14,
    strokecolor = :grey50,
    strokewidth = 0.5)

lines!(axC, [0.0, 1.05], [0.0, 1.05],
    color = (:black, 0.4), linewidth = 2, linestyle = :dash)

Colorbar(fig[1, 5],
    limits     = (minimum(T_ov), maximum(T_ov)),
    colormap   = cmap_sc,
    label      = "Temperature (°C)",
    labelsize  = 26,
    vertical   = true,
    tellheight = false)

text!(axC, 0.02, 0.98,
    text     = "Pearson r = $(round(r_pearson, digits=3))\nSpearman ρ = $(round(r_spearman, digits=3))",
    color    = :black,
    fontsize = 26,
    align    = (:left, :top))
    
# =============================================================================
# Panel labels
# =============================================================================
Label(fig[1, 2, TopLeft()], "(a)", fontsize = 36, font = :bold, padding = (0, 0, 8, 0))
Label(fig[1, 4, TopLeft()], "(b)", fontsize = 36, font = :bold, padding = (0, 0, 8, 0))

# =============================================================================
# Save
# =============================================================================
# save("../results/diversity_temp_atlas.pdf", fig)
println("Saved →../results/diversity_temp_atlas.pdf")
println("ATLAS Gaussian peak temperature: $(round(T_peak_atlas_gauss, digits=2))°C")
fig