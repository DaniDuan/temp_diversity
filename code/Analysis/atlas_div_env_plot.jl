# =============================================================================
# atlas_div_env_plot.jl
#
# Plots MiCRM and ATLAS richness against temperature.
# Loads:
#   stats_re.jld2         — MiCRM richness mean/se
#   summary_atlas.csv     — ATLAS empirical richness data
# =============================================================================

using CairoMakie, JLD2, Statistics, CSV, DataFrames, StatsBase,
      Interpolations, LinearAlgebra, Random, ProgressMeter

# =============================================================================
# Constants
# =============================================================================
const k_B = 8.617e-5
invT(T_C) = 1.0 / (k_B * (T_C + 273.15))

# =============================================================================
# WLS solver
# =============================================================================
function wls(X, y, w)
    W = Diagonal(w)
    return (X' * W * X) \ (X' * W * y)
end

# =============================================================================
# Weighted R²
# =============================================================================
function wR2(y, ŷ, w)
    ss_res = sum(w .* (y .- ŷ).^2)
    ss_tot = sum(w .* (y .- mean(y)).^2)
    return 1.0 - ss_res / ss_tot
end

# =============================================================================
# Load MiCRM summary statistics
# =============================================================================
@load "../results/stats_re.jld2" rich_mean rich_var Shannon_mean Shannon_var Simpson_mean Simpson_var u_sum_mean u_sum_var m_mean m_var ϵ_mean ϵ_var r_inv_mean r_inv_var Eu_mean Eu_var Em_mean Em_var Tpu_mean Tpu_var Tpm_mean Tpm_var

n_reps    = 999
num_temps = 31
T_axis    = range(0, num_temps - 1, length = num_temps)
rich_se   = sqrt.(rich_var) ./ sqrt(n_reps)

# =============================================================================
# Bootstrap CI on MiCRM peak temperature
# =============================================================================
@load "../results/richness_re.jld2" rich_all

T_vals       = collect(T_axis)
n_bootstrap  = 10000
peak_T_boots = zeros(n_bootstrap)
Random.seed!(666)
for b in 1:n_bootstrap
    boot_means      = [mean(rand(rich_all[j], length(rich_all[j])))
                       for j in 1:num_temps]
    peak_T_boots[b] = T_vals[argmax(boot_means)]
end

T_peak_micrm    = T_vals[argmax(rich_mean)]
T_peak_micrm_lo = quantile(peak_T_boots, 0.025)
T_peak_micrm_hi = quantile(peak_T_boots, 0.975)

println("MiCRM Peak: ", round(T_peak_micrm, digits=2),
        "°C  CI: [", round(T_peak_micrm_lo, digits=2),
        ", ",        round(T_peak_micrm_hi, digits=2), "]")

# =============================================================================
# Load ATLAS data
# =============================================================================
top10_df    = CSV.read("../data/top10_atlas.csv",          DataFrame)
binstats_df = CSV.read("../data/top10_atlas_binstats.csv", DataFrame)
sort!(binstats_df, :Temp_bin)

# Core numeric vectors
T_all_C = Float64.(top10_df.Temperature_C_mean)
R_all   = Float64.(top10_df.Richness)
x_all   = invT.(T_all_C)

# Bin counts → WLS weights
bin_counts_full = combine(groupby(top10_df, :Temp_bin), nrow => :n_full)
top10_df        = leftjoin(top10_df, bin_counts_full, on = :Temp_bin)
w_full          = Float64.(top10_df.n_full)

# =============================================================================
# Diagnostics — pH dropped as confounder
# (r_Temp_pH = 0.08, p_pH_Richness = 0.155 — not significant)
# =============================================================================
println("\n=== Diagnostics ===")
println("Total samples: ", nrow(top10_df))
println("pH not included as confounder: r(T,pH)=0.08, p(pH,Richness)=0.155")

# =============================================================================
# WLS — quadratic in 1/kT on full dataset
# Model: R = β₀ + β₁·x + β₂·x²     x = 1/(k_B·T)
# =============================================================================
println("\n=== WLS: quadratic in 1/kT, full dataset ===")

n_obs = length(x_all)
X     = hcat(ones(n_obs), x_all, x_all.^2)

β    = wls(X, R_all, w_full)
ŷ    = X * β
R2   = wR2(R_all, ŷ, w_full)

println("β₀         = ", round(β[1], digits=4))
println("β₁ (1/kT)  = ", round(β[2], digits=4))
println("β₂ (1/kT)² = ", round(β[3], digits=4))
println("R²         = ", round(R2,   digits=4))

β[3] >= 0 && @warn "β₂ ≥ 0 — parabola concave UP (no peak)"

# Peak: x* = −β₁/2β₂  →  T* = 1/(k_B·x*) − 273.15
invT_peak    = -β[2] / (2.0 * β[3])
T_peak_atlas = 1.0 / (k_B * invT_peak) - 273.15

println("Peak 1/kT  = ", round(invT_peak,    digits=4), " eV⁻¹")
println("Peak T     = ", round(T_peak_atlas, digits=2), " °C")

# =============================================================================
# Prediction curve on fine grid
# =============================================================================
T_min_data = minimum(T_all_C)
T_max_data = maximum(T_all_C)
pred_TC    = collect(range(T_min_data, T_max_data, length = 300))
pred_invT  = invT.(pred_TC)
X_pred     = hcat(ones(300), pred_invT, pred_invT.^2)
wls_pred   = X_pred * β

# =============================================================================
# Bootstrap CI
# =============================================================================
println("\n=== Bootstrap CI on peak ===")

n_boot      = 2000
Random.seed!(666)
boot_peaks  = Float64[]
boot_curves = zeros(300, n_boot)
n_valid     = 0

idx_full = collect(1:nrow(top10_df))
prog     = Progress(n_boot; desc = "Bootstrapping: ", showspeed = true)

for b in 1:n_boot
    next!(prog)

    bi    = sample(idx_full, length(idx_full); replace = true)
    x_b   = x_all[bi]
    R_b   = R_all[bi]
    w_b   = w_full[bi]

    X_b  = hcat(ones(length(x_b)), x_b, x_b.^2)
    β_b  = try
        wls(X_b, R_b, w_b)
    catch
        continue
    end

    β_b[3] >= 0 && continue

    invT_pk_b = -β_b[2] / (2.0 * β_b[3])
    T_pk_b    = 1.0 / (k_B * invT_pk_b) - 273.15

    (T_pk_b < T_min_data || T_pk_b > T_max_data) && continue

    n_valid += 1
    push!(boot_peaks, T_pk_b)
    boot_curves[:, n_valid] = X_pred * β_b
end

boot_curves = boot_curves[:, 1:n_valid]

println("Valid replicates:  ", n_valid, " / ", n_boot)
println("Peak boot std:     ", round(std(boot_peaks), digits=3), " °C")

T_peak_atlas_lo = quantile(boot_peaks, 0.025)
T_peak_atlas_hi = quantile(boot_peaks, 0.975)

println("ATLAS Peak: ", round(T_peak_atlas,    digits=2), "°C  CI: [",
        round(T_peak_atlas_lo, digits=2), ", ",
        round(T_peak_atlas_hi, digits=2), "]")

boot_std_curve = [std(boot_curves[i, :]) for i in 1:300]
wls_se_lo      = wls_pred .- boot_std_curve
wls_se_hi      = wls_pred .+ boot_std_curve

# =============================================================================
# Bin means and SE for scatter points
# =============================================================================
binstats_raw = combine(groupby(top10_df, :Temp_bin)) do sub
    DataFrame(
        Mean_raw = mean(Float64.(sub.Richness)),
        SE_raw   = std(Float64.(sub.Richness)) / sqrt(nrow(sub)),
    )
end
sort!(binstats_raw, :Temp_bin)

atlas_bin_temp = Float64.(binstats_raw.Temp_bin)
atlas_bin_mean = Float64.(binstats_raw.Mean_raw)
atlas_bin_se   = Float64.(binstats_raw.SE_raw)

# =============================================================================
# Rescale ATLAS values onto MiCRM y-axis
# =============================================================================
atlas_min = minimum(atlas_bin_mean)
atlas_max = maximum(atlas_bin_mean)
micrm_min = minimum(rich_mean)
micrm_max = maximum(rich_mean)

atlas_to_micrm(v) = micrm_min .+ (v .- atlas_min) ./ (atlas_max .- atlas_min) .*
                    (micrm_max .- micrm_min)

scale_factor      = (micrm_max - micrm_min) / (atlas_max - atlas_min)
atlas_mean_scaled = atlas_to_micrm(atlas_bin_mean)
atlas_se_scaled   = atlas_bin_se .* scale_factor
wls_pred_scaled   = atlas_to_micrm(wls_pred)
wls_se_lo_scaled  = wls_pred_scaled .- (boot_std_curve .* scale_factor)
wls_se_hi_scaled  = wls_pred_scaled .+ (boot_std_curve .* scale_factor)

# =============================================================================
# Normalise for panel (b)
# =============================================================================
atlas_itp     = linear_interpolation(atlas_bin_temp, atlas_bin_mean,
                                     extrapolation_bc = NaN)
atlas_err_itp = linear_interpolation(atlas_bin_temp, atlas_bin_se,
                                     extrapolation_bc = NaN)

T_ov_all  = collect(T_axis)
a_interp  = atlas_itp.(T_ov_all)
a_err_int = atlas_err_itp.(T_ov_all)
overlap   = .!isnan.(a_interp)

T_ov        = T_ov_all[overlap]
micrm_ov    = rich_mean[overlap]
micrm_se_ov = rich_se[overlap]
atlas_ov    = a_interp[overlap]
atlas_se_ov = a_err_int[overlap]

norm01(x)     = (x .- minimum(x)) ./ (maximum(x) .- minimum(x))
micrm_norm    = norm01(micrm_ov)
atlas_norm    = norm01(atlas_ov)
micrm_se_norm = micrm_se_ov ./ (maximum(micrm_ov) - minimum(micrm_ov))
atlas_se_norm = atlas_se_ov ./ (maximum(atlas_ov)  - minimum(atlas_ov))

r_pearson  = cor(micrm_norm, atlas_norm)
r_spearman = corspearman(micrm_norm, atlas_norm)
println("\nPearson r  = ", round(r_pearson,  digits=3),
        "  Spearman ρ = ", round(r_spearman, digits=3))

# =============================================================================
# Figure
# =============================================================================
fig = Figure(fontsize = 30, size = (2400, 900))

ax1 = Axis(fig[1, 1],
    xlabel          = "Temperature (°C)",
    ylabel          = "Richness (predicted)",
    xlabelsize      = 40,
    ylabelsize      = 40,
    ygridvisible    = true,
    xgridvisible    = true,
    topspinevisible = false,
)

ax0 = Axis(fig[1, 1],
    ylabel             = "Richness (ATLAS top 10%)",
    ylabelsize         = 36,
    yaxisposition      = :right,
    yticklabelalign    = (:left, :center),
    xticklabelsvisible = false,
    xlabelvisible      = false,
    xticksvisible      = false,
    leftspinevisible   = false,
    bottomspinevisible = false,
    topspinevisible    = false,
    rightspinevisible  = true,
    xgridvisible       = false,
    ygridvisible       = false,
)

ylims!(ax1, micrm_min - 0.05*(micrm_max - micrm_min),
            micrm_max + 0.05*(micrm_max - micrm_min))
ylims!(ax0, atlas_min - 0.05*(atlas_max - atlas_min),
            atlas_max + 0.05*(atlas_max - atlas_min))
linkxaxes!(ax1, ax0)

vspan!(ax1, T_peak_micrm_lo, T_peak_micrm_hi, color = ("#6B8EDE", 0.20))
vspan!(ax1, T_peak_atlas_lo, T_peak_atlas_hi, color = ("#E17542", 0.20))

band!(ax1, T_axis,
      rich_mean .- rich_se,
      rich_mean .+ rich_se,
      color = ("#6B8EDE", 0.20))

band!(ax1, pred_TC,
      wls_se_lo_scaled,
      wls_se_hi_scaled,
      color = ("#E17542", 0.20))

lines!(ax1, T_axis, rich_mean,
       color = ("#6B8EDE", 0.9), linewidth = 5)

lines!(ax1, pred_TC, wls_pred_scaled,
       color = ("#E17542", 0.9), linewidth = 3)

vlines!(ax1, [T_peak_micrm],
        color = ("#6B8EDE", 0.9), linewidth = 2.5, linestyle = :dash)
vlines!(ax1, [T_peak_atlas],
        color = ("#E17542", 0.9), linewidth = 2.5, linestyle = :dash)

scatter!(ax1, atlas_bin_temp, atlas_mean_scaled,
         color = ("#E17542", 1.0), markersize = 14)

cap = 0.15
for (x, y, e) in zip(atlas_bin_temp, atlas_mean_scaled, atlas_se_scaled)
    lines!(ax1, [x, x],             [y - e, y + e], color = "#E17542", linewidth = 2)
    lines!(ax1, [x - cap, x + cap], [y - e, y - e], color = "#E17542", linewidth = 2)
    lines!(ax1, [x - cap, x + cap], [y + e, y + e], color = "#E17542", linewidth = 2)
end

text!(ax1, T_peak_micrm + 0.3,
      micrm_min + 0.05 * (micrm_max - micrm_min),
      text  = "$(Int(round(T_peak_micrm)))°C",
      color = "#6B8EDE", fontsize = 24, align = (:left, :bottom))

text!(ax1, T_peak_atlas + 0.3,
      micrm_min + 0.20 * (micrm_max - micrm_min),
      text  = "$(round(T_peak_atlas, digits=1))°C",
      color = "#E17542", fontsize = 24, align = (:left, :bottom))

l1 = LineElement(color = ("#6B8EDE", 0.9), linewidth = 5)
l2 = LineElement(color = ("#E17542", 0.9), linewidth = 3)
s1 = MarkerElement(color = ("#E17542", 1.0), markersize = 14, marker = :circle)
Legend(fig[1, 1], [s1, l2, l1],
    ["Richness (ATLAS top 10%)",
     "WLS fit on 1/kT",
     "Richness (predicted)"],
    tellheight = false, tellwidth = false,
    halign = :right, valign = :top, framevisible = false)

# =============================================================================
# Panel (b)
# =============================================================================
axC = Axis(fig[1, 2],
    xlabel            = "Richness predicted (normalised)",
    ylabel            = "ATLAS richness top 10% (normalised)",
    xlabelsize        = 36,
    ylabelsize        = 36,
    ygridvisible      = false,
    xgridvisible      = false,
    topspinevisible   = false,
    rightspinevisible = false,
    aspect            = 1,
)
xlims!(axC, -0.05, 1.1)
ylims!(axC, -0.05, 1.1)

cmap_sc   = cgrad(["#4575B4", "#D73027"])
pt_colors = [cmap_sc[(t - minimum(T_ov)) / (maximum(T_ov) - minimum(T_ov))]
             for t in T_ov]

cap_b = 0.005
for (x, y, xe, ye, col) in zip(micrm_norm, atlas_norm,
                                micrm_se_norm, atlas_se_norm, pt_colors)
    lines!(axC, [x - xe, x + xe], [y,         y        ], color=(col,0.7), linewidth=1.5)
    lines!(axC, [x - xe, x - xe], [y - cap_b, y + cap_b], color=(col,0.7), linewidth=1.5)
    lines!(axC, [x + xe, x + xe], [y - cap_b, y + cap_b], color=(col,0.7), linewidth=1.5)
    lines!(axC, [x,       x      ], [y - ye,  y + ye   ], color=(col,0.7), linewidth=1.5)
    lines!(axC, [x - cap_b, x + cap_b], [y - ye, y - ye], color=(col,0.7), linewidth=1.5)
    lines!(axC, [x - cap_b, x + cap_b], [y + ye, y + ye], color=(col,0.7), linewidth=1.5)
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

Colorbar(fig[1, 3],
    limits     = (minimum(T_ov), maximum(T_ov)),
    colormap   = cmap_sc,
    label      = "Temperature (°C)",
    labelsize  = 26,
    vertical   = true,
    tellheight = false)

text!(axC, 0.02, 0.98,
    text     = "Pearson r = $(round(r_pearson, digits=3))\n" *
               "Spearman ρ = $(round(r_spearman, digits=3))",
    color    = :black,
    fontsize = 26,
    align    = (:left, :top))

# =============================================================================
# Panel labels + save
# =============================================================================
Label(fig[1, 1, TopLeft()], "(a)", fontsize=36, font=:bold, padding=(0,0,8,0))
Label(fig[1, 2, TopLeft()], "(b)", fontsize=36, font=:bold, padding=(0,0,8,0))

# save("../results/diversity_temp_atlas.pdf", fig)

println("\nSaved → ../results/diversity_temp_atlas.pdf")
println("MiCRM:  ", round(T_peak_micrm,    digits=2), "°C  [",
        round(T_peak_micrm_lo, digits=2), ", ",
        round(T_peak_micrm_hi, digits=2), "]")
println("ATLAS:  ", round(T_peak_atlas,     digits=2), "°C  [",
        round(T_peak_atlas_lo, digits=2), ", ",
        round(T_peak_atlas_hi, digits=2), "]")
println("β = [", round(β[1], digits=3), ", ",
                 round(β[2], digits=3), ", ",
                 round(β[3], digits=3), "]")
println("R² = ", round(R2, digits=3))
fig