# =============================================================================
# atlas_div_env_plot.jl
# =============================================================================

using CairoMakie, JLD2, Statistics, CSV, DataFrames, StatsBase,
      Interpolations, LsqFit, LinearAlgebra, Random, ProgressMeter

# =============================================================================
# Load MiCRM summary statistics
# =============================================================================
@load "../results/stats_re.jld2" rich_mean rich_var Shannon_mean Shannon_var Simpson_mean Simpson_var u_sum_mean u_sum_var m_mean m_var ϵ_mean ϵ_var r_inv_mean r_inv_var Eu_mean Eu_var Em_mean Em_var Tpu_mean Tpu_var Tpm_mean Tpm_var

n_reps    = 999
num_temps = 31
T_axis    = range(0, num_temps - 1, length = num_temps)
rich_se   = sqrt.(rich_var)./ sqrt(n_reps)

# =============================================================================
# Bootstrap CI on MiCRM peak temperature
# =============================================================================
@load "../results/richness_re.jld2" rich_all

T_vals       = collect(T_axis)
n_bootstrap  = 10000
peak_T_boots = zeros(n_bootstrap)
Random.seed!(42)
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

# =============================================================================
# Lorentzian model + AIC
# =============================================================================
lorentz_model(T, p) = p[1]./ (1.0.+ ((T.- p[2])./ p[3]).^2)

function compute_aic(resid, k)
    n   = length(resid)
    rss = sum(resid.^ 2)
    return n * log(rss / n) + 2 * k
end

bin_counts  = combine(groupby(top10_df, :Temp_bin), nrow => :n)
top10_df    = leftjoin(top10_df, bin_counts, on = :Temp_bin)
weights_vec = Float64.(top10_df.n)
x_data      = Float64.(top10_df.Temperature_C_mean)
y_data      = Float64.(top10_df.Richness)
n_params    = 3

# =============================================================================
# Robust Lorentzian fit — full grid, AIC selection
# =============================================================================
A_grid  = collect(500.0:100.0:4000.0)
T0_grid = collect(0.0:1.0:30.0)
γ_grid  = collect(0.0:1.0:30.0)
n_total = length(A_grid) * length(T0_grid) * length(γ_grid)

best_fit = nothing
best_aic = Inf

prog_fit = Progress(n_total; desc = "Fitting Lorentzian: ", showspeed = true)

for A_init in A_grid
    for T0_init in T0_grid
        for γ_init in γ_grid
            next!(prog_fit)
            p0 = [A_init, T0_init, γ_init]
            try
                fit_attempt = curve_fit(lorentz_model, x_data, y_data,
                                        weights_vec, p0;
                                        maxIter = 10000,
                                        x_tol   = 1e-10,
                                        g_tol   = 1e-10)
                T0_val = fit_attempt.param[2]
                γ_val  = fit_attempt.param[3]
                A_val  = fit_attempt.param[1]
                if T0_val < 0.0 || T0_val > 30.0; continue; end
                if γ_val  < 0.5 || γ_val  > 30.0; continue; end
                if A_val  < 0.0;                   continue; end
                aic = compute_aic(fit_attempt.resid, n_params)
                if aic < best_aic
                    best_aic = aic
                    best_fit = fit_attempt
                end
            catch; continue
            end
        end
    end
end

@assert best_fit !== nothing "Lorentzian fit failed"

A_fit    = best_fit.param[1]
T0_fit   = best_fit.param[2]
γ_fit    = best_fit.param[3]
best_rss = sum(best_fit.resid.^ 2)

println("\n=== Lorentzian Fit ===")
println("  A  = ", round(A_fit,    digits=2))
println("  T₀ = ", round(T0_fit,   digits=2), " °C")
println("  γ  = ", round(γ_fit,    digits=2), " °C")
println("  AIC = ", round(best_aic, digits=4))
println("  RSS = ", round(best_rss, digits=2))

pred_x       = collect(range(0.0, 30.0, length = 300))
lorentz_pred = lorentz_model(pred_x, best_fit.param)  # The actual fit curve
T_peak_atlas = T0_fit

# =============================================================================
# Warm-start bootstrap — SE band + CI on peak
# =============================================================================
n_boot    = 2000
Random.seed!(666)
residuals = y_data.- lorentz_model(x_data, best_fit.param)

δA  = A_fit  * 0.3
δT0 = 5.0
δγ  = γ_fit  * 0.3

A_boot_grid  = collect(max(100.0, A_fit  - δA)  : (2*δA/10)  : (A_fit  + δA))
T0_boot_grid = collect(max(0.0,   T0_fit - δT0) : (2*δT0/10) : min(30.0, T0_fit + δT0))
γ_boot_grid  = collect(max(0.5,   γ_fit  - δγ)  : (2*δγ/10)  : (γ_fit  + δγ))

boot_params = Vector{Vector{Float64}}()
boot_peaks  = Float64[]

prog_boot = Progress(n_boot; desc = "Bootstrapping:      ", showspeed = true)

for _ in 1:n_boot
    next!(prog_boot)
    y_boot = lorentz_model(x_data, best_fit.param).+
             sample(residuals, length(residuals), replace=true)

    local_best     = nothing
    local_best_aic = Inf

    for A_init in A_boot_grid
        for T0_init in T0_boot_grid
            for γ_init in γ_boot_grid
                p0 = [A_init, T0_init, γ_init]
                try
                    fit_b = curve_fit(lorentz_model, x_data, y_boot,
                                      weights_vec, p0;
                                      maxIter = 5000, x_tol = 1e-8, g_tol = 1e-8)
                    T0_b = fit_b.param[2]
                    γ_b  = fit_b.param[3]
                    A_b  = fit_b.param[1]
                    if T0_b < 0.0 || T0_b > 30.0; continue; end
                    if γ_b  < 0.5 || γ_b  > 30.0; continue; end
                    if A_b  < 0.0;                 continue; end
                    aic_b = compute_aic(fit_b.resid, n_params)
                    if aic_b < local_best_aic
                        local_best_aic = aic_b
                        local_best     = fit_b
                    end
                catch; continue
                end
            end
        end
    end

    if local_best !== nothing
        push!(boot_params, local_best.param)
        push!(boot_peaks,  local_best.param[2])
    end
end

println("\nBootstrap replicates: ", length(boot_params), " / ", n_boot)

A_vals  = [p[1] for p in boot_params]
T0_vals = [p[2] for p in boot_params]
γ_vals  = [p[3] for p in boot_params]
println("A  std: ", round(std(A_vals),  digits=4))
println("T0 std: ", round(std(T0_vals), digits=4))
println("γ  std: ", round(std(γ_vals),  digits=4))

# =============================================================================
# SE band — KEY FIX:
# Compute std of bootstrap curves at each x point
# Then SE band = lorentz_pred ± std  (centred on ACTUAL FIT, not boot mean)
# This guarantees symmetric band above AND below the orange line
# =============================================================================
pred_matrix    = hcat([lorentz_model(pred_x, p) for p in boot_params]...)
boot_std_curve = [std(pred_matrix[i, :]) for i in 1:length(pred_x)]

# Band is centred on the actual fit curve — symmetric above and below
lorentz_se_lo  = lorentz_pred.- boot_std_curve   # actual fit MINUS std
lorentz_se_hi  = lorentz_pred.+ boot_std_curve   # actual fit PLUS  std

println("Mean SE band width: ", round(mean(lorentz_se_hi.- lorentz_se_lo), digits=2))

# 95% CI on peak from bootstrap T0 values
T_peak_atlas_lo = quantile(boot_peaks, 0.025)
T_peak_atlas_hi = quantile(boot_peaks, 0.975)

println("ATLAS Peak CI: [", round(T_peak_atlas_lo, digits=2),
        ", ",              round(T_peak_atlas_hi, digits=2), "] °C")
println("CI width: ",      round(T_peak_atlas_hi - T_peak_atlas_lo, digits=2), "°C")

# =============================================================================
# Rescale ALL ATLAS values to MiCRM y-axis scale
# =============================================================================
atlas_bin_mean = Float64.(binstats_df.Mean)
atlas_bin_se   = Float64.(binstats_df.SE)
atlas_bin_temp = Float64.(binstats_df.Temp_bin)

atlas_min = minimum(atlas_bin_mean)
atlas_max = maximum(atlas_bin_mean)
micrm_min = minimum(rich_mean)
micrm_max = maximum(rich_mean)

function atlas_to_micrm(v)
    micrm_min.+ (v.- atlas_min)./ (atlas_max.- atlas_min).*
    (micrm_max.- micrm_min)
end

scale_factor         = (micrm_max - micrm_min) / (atlas_max - atlas_min)
atlas_mean_scaled    = atlas_to_micrm(atlas_bin_mean)
atlas_se_scaled      = atlas_bin_se.* scale_factor
lorentz_pred_scaled  = atlas_to_micrm(lorentz_pred)

# KEY FIX: SE band scaled from lorentz_pred (actual fit), not boot mean
lorentz_se_lo_scaled = lorentz_pred_scaled.- (boot_std_curve.* scale_factor)
lorentz_se_hi_scaled = lorentz_pred_scaled.+ (boot_std_curve.* scale_factor)

println("Scaled SE band width: ",
        round(mean(lorentz_se_hi_scaled.- lorentz_se_lo_scaled), digits=4))

# =============================================================================
# Normalise for panel (b)
# =============================================================================
atlas_itp     = linear_interpolation(atlas_bin_temp, atlas_bin_mean,
                                     extrapolation_bc = NaN)
atlas_err_itp = linear_interpolation(atlas_bin_temp, atlas_bin_se,
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

norm01(x)     = (x.- minimum(x))./ (maximum(x).- minimum(x))
micrm_norm    = norm01(micrm_ov)
atlas_norm    = norm01(atlas_ov)
micrm_se_norm = micrm_se_ov./ (maximum(micrm_ov) - minimum(micrm_ov))
atlas_se_norm = atlas_se_ov./ (maximum(atlas_ov)  - minimum(atlas_ov))

r_pearson  = cor(micrm_norm, atlas_norm)
r_spearman = corspearman(micrm_norm, atlas_norm)
println("Pearson r = ", round(r_pearson, digits=3),
        "  Spearman ρ = ", round(r_spearman, digits=3))

# =============================================================================
# Figure
# =============================================================================
fig = Figure(fontsize = 30, size = (2400, 900));

# ── FIX 2: removed yticklabelcolor / ylabelcolor from BOTH axes ───────────
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
    ylabelsize         = 40,
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

# =============================================================================
# PLOTTING — order matters: backgrounds first, then lines, then points on top
# =============================================================================

# ── LAYER 1: vspan backgrounds (drawn FIRST so nothing covers them) ────────

# MiCRM peak 95% CI — blue vertical band
vspan!(ax1, T_peak_micrm_lo, T_peak_micrm_hi,
       color = ("#6B8EDE", 0.20))

# ATLAS peak 95% CI — orange vertical band
# Drawn on ax1 (shares x with ax0) so it is guaranteed to render
vspan!(ax1, T_peak_atlas_lo, T_peak_atlas_hi,
       color = ("#E17542", 0.20))

# ── LAYER 2: SE bands ─────────────────────────────────────────────────────

# MiCRM SE band — blue
band!(ax1, T_axis,
      rich_mean.- rich_se,
      rich_mean.+ rich_se,
      color = ("#6B8EDE", 0.20))

# ATLAS Lorentzian SE band — orange
# lorentz_se_lo_scaled = lorentz_pred_scaled - std  (below the line)
# lorentz_se_hi_scaled = lorentz_pred_scaled + std  (above the line)
# Both are symmetric around the actual orange fit curve
band!(ax1, pred_x,
      lorentz_se_lo_scaled,
      lorentz_se_hi_scaled,
      color = ("#E17542", 0.20))

# ── LAYER 3: Fit lines ────────────────────────────────────────────────────

# MiCRM richness line — blue
lines!(ax1, T_axis, rich_mean,
       color = ("#6B8EDE", 0.9), linewidth = 5)

# ATLAS Lorentzian fit line — orange
lines!(ax1, pred_x, lorentz_pred_scaled,
       color = ("#E17542", 0.9), linewidth = 3)

# ── LAYER 4: Peak dashed vertical lines ───────────────────────────────────

# MiCRM peak dashed line
vlines!(ax1, [T_peak_micrm],
        color = ("#6B8EDE", 0.9), linewidth = 2.5, linestyle = :dash)

# ATLAS peak dashed line
vlines!(ax1, [T_peak_atlas],
        color = ("#E17542", 0.9), linewidth = 2.5, linestyle = :dash)

# ── LAYER 5: Data points + error bars (on top of everything) ──────────────

# ATLAS bin mean points
scatter!(ax1, atlas_bin_temp, atlas_mean_scaled,
         color = ("#E17542", 1.0), markersize = 14)

cap = 0.15
for (x, y, e) in zip(atlas_bin_temp, atlas_mean_scaled, atlas_se_scaled)
    lines!(ax1, [x, x],             [y - e, y + e],     color = "#E17542", linewidth = 2)
    lines!(ax1, [x - cap, x + cap], [y - e, y - e],     color = "#E17542", linewidth = 2)
    lines!(ax1, [x - cap, x + cap], [y + e, y + e],     color = "#E17542", linewidth = 2)
end

# ── LAYER 6: Text labels ──────────────────────────────────────────────────

text!(ax1, T_peak_micrm + 0.3,
      micrm_min + 0.05 * (micrm_max - micrm_min),
      text  = "$(Int(round(T_peak_micrm)))°C",
      color = "#6B8EDE", fontsize = 24, align = (:left, :bottom))

text!(ax1, T_peak_atlas + 0.3,
      micrm_min + 0.20 * (micrm_max - micrm_min),
      text  = "$(round(T_peak_atlas, digits=1))°C",
      color = "#E17542", fontsize = 24, align = (:left, :bottom))

# =============================================================================
# Legend
# =============================================================================
l1 = LineElement(color = ("#6B8EDE", 0.9), linewidth = 5)
l2 = LineElement(color = ("#E17542", 0.9), linewidth = 3)
s1 = MarkerElement(color = ("#E17542", 1.0), markersize = 14, marker = :circle)
Legend(fig[1, 1], [s1, l2, l1],
    ["Richness (ATLAS top 10%)",
     "Lorentzian fit (ATLAS)",
     "Richness (predicted)"],
    tellheight = false, tellwidth = false,
    halign = :right, valign = :top, framevisible = false)

# =============================================================================
# Panel B: Scatter
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

cap = 0.005
for (x, y, xe, ye, col) in zip(micrm_norm, atlas_norm,
                                micrm_se_norm, atlas_se_norm, pt_colors)
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

Colorbar(fig[1, 3],
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
# Panel labels + Save
# =============================================================================
Label(fig[1, 1, TopLeft()], "(a)", fontsize = 36, font = :bold, padding = (0, 0, 8, 0))
Label(fig[1, 2, TopLeft()], "(b)", fontsize = 36, font = :bold, padding = (0, 0, 8, 0))

# save("../results/diversity_temp_atlas.pdf", fig)
println("\nSaved →../results/diversity_temp_atlas.pdf")
println("MiCRM: ", round(T_peak_micrm,       digits=2), "°C  [",
        round(T_peak_micrm_lo,    digits=2), ", ",
        round(T_peak_micrm_hi,    digits=2), "]")
println("ATLAS: ", round(T_peak_atlas,        digits=2), "°C  [",
        round(T_peak_atlas_lo,    digits=2), ", ",
        round(T_peak_atlas_hi,    digits=2), "]")
println("Lorentzian: A=", round(A_fit, digits=2),
        "  T₀=", round(T0_fit, digits=2), "°C",
        "  γ=",  round(γ_fit,  digits=2), "°C",
        "  AIC=", round(best_aic, digits=2))
fig