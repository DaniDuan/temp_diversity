#############################################################################################
# ana_atlas_function.jl
#############################################################################################

using CairoMakie, Statistics, CSV, DataFrames, StatsBase, LinearAlgebra, Random, ProgressMeter

# =============================================================================
# Constants
# =============================================================================
const k_B = 8.617e-5
invT(T_C) = 1.0 / (k_B * (T_C + 273.15))
const col_temp      = "#F5D44B"   # temperate — yellow
const col_temp_dark = "#B8960A"   # darkened for error bars / text
const col_trop      = "#B73508"   # tropical  — dark orange-red

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
    ss_res = sum(w.* (y.- ŷ).^2)
    ss_tot = sum(w.* (y.- mean(y)).^2)
    return 1.0 - ss_res / ss_tot
end

# =============================================================================
# Colours
# =============================================================================
const col_temp = "#F5D44B"   # temperate — yellow
const col_trop = "#B73508"   # tropical  — dark orange-red

# Darkened temperate colour for error bars / text (yellow is invisible on white)
const col_temp_dark = "#B8960A"

# =============================================================================
# Per-zone analysis function
# Returns everything needed for plotting one panel
# =============================================================================
function analyse_zone(df, zone_label; n_boot = 2000, seed = 666)

    println("\n", "="^60)
    println("Zone: ", zone_label)
    println("="^60)

    # ── Core vectors ──────────────────────────────────────────
    T_all_C = Float64.(df.Temperature_C_mean)
    R_all   = Float64.(df.Richness)
    x_all   = invT.(T_all_C)
    w_full  = Float64.(df.n_bin)

    T_min = minimum(T_all_C)
    T_max = maximum(T_all_C)

    # ── Parse pH safely ───────────────────────────────────────
    pH_parsed = map(df.pH) do val
        ismissing(val) && return missing
        v = tryparse(Float64, string(val))
        return isnothing(v) ? missing : v
    end

    has_ph    =.!ismissing.(pH_parsed)
    idx_ph    = findall(has_ph)
    pH_values = Float64.(collect(skipmissing(pH_parsed)))

    println("Samples with valid pH: ", length(idx_ph), " / ", nrow(df))

    # Mean-centre pH; missing → 0.0
    pH_wmean = sum(w_full[idx_ph].* pH_values) / sum(w_full[idx_ph])
    println("Weighted pH mean: ", round(pH_wmean, digits = 4))

    pH_centered = map(enumerate(pH_parsed)) do (i, val)
        ismissing(val) ? 0.0 : (Float64(val) - pH_wmean)
    end

    # ── WLS Model: Richness ~ 1/kT + (1/kT)² + pH ────────────
    n_obs = length(x_all)
    X     = hcat(ones(n_obs), x_all, x_all.^2, pH_centered)
    β     = wls(X, R_all, w_full)
    ŷ     = X * β
    R2    = wR2(R_all, ŷ, w_full)

    println("β₀         = ", round(β[1], digits = 4))
    println("β₁ (1/kT)  = ", round(β[2], digits = 4))
    println("β₂ (1/kT)² = ", round(β[3], digits = 4))
    println("β₃ (pH)    = ", round(β[4], digits = 4))
    println("R²         = ", round(R2,   digits = 4))

    # ── Check concavity — peak only exists if β₂ < 0 ─────────
    has_peak = β[3] < 0

    if !has_peak
        @warn "β₂ ≥ 0 — parabola concave UP — no peak exists in $zone_label"
    end

    # Peak temperature — only meaningful if has_peak
    invT_peak = has_peak ? -β[2] / (2.0 * β[3]) : NaN
    T_peak    = has_peak ? 1.0 / (k_B * invT_peak) - 273.15 : NaN
    println("Peak T = ", has_peak ? "$(round(T_peak, digits=2))°C" : "none (concave up)")

    # ── Prediction curve on fine grid ─────────────────────────
    pred_TC   = collect(range(T_min, T_max, length = 300))
    pred_invT = invT.(pred_TC)
    X_pred    = hcat(ones(300), pred_invT, pred_invT.^2, zeros(300))
    wls_pred  = X_pred * β

    # ── Bootstrap CI — only if point estimate has a valid peak ─
    T_peak_lo  = NaN
    T_peak_hi  = NaN
    boot_sd    = NaN
    ci_width   = NaN
    boot_peaks = Float64[]

    # Initialise boot_std_curve to zero — overwritten if has_peak
    boot_std_curve = zeros(300)

    if has_peak
        println("\nBootstrapping (n = $n_boot)...")
        Random.seed!(seed)

        idx_full    = collect(1:nrow(df))
        boot_curves = Vector{Vector{Float64}}()

        prog = Progress(n_boot; desc = "  Bootstrap [$zone_label]: ",
                        showspeed = true)

        for b in 1:n_boot
            next!(prog)

            bi   = sample(idx_full, length(idx_full); replace = true)
            x_b  = x_all[bi]
            R_b  = R_all[bi]
            w_b  = w_full[bi]
            ph_b = pH_centered[bi]

            X_b = hcat(ones(length(x_b)), x_b, x_b.^2, ph_b)

            β_b = try
                wls(X_b, R_b, w_b)
            catch
                continue
            end

            # Reject concave-up resamples
            β_b[3] >= 0 && continue

            invT_pk_b = -β_b[2] / (2.0 * β_b[3])
            T_pk_b    = 1.0 / (k_B * invT_pk_b) - 273.15

            # Reject peaks outside data range
            (T_pk_b < T_min || T_pk_b > T_max) && continue

            push!(boot_peaks,  T_pk_b)
            push!(boot_curves, X_pred * β_b)
        end

        n_valid = length(boot_peaks)
        println("Valid bootstrap replicates: ", n_valid, " / ", n_boot)

        if n_valid >= 10
            println("Peak boot std: ", round(std(boot_peaks), digits = 3), " °C")

            T_peak_lo = quantile(boot_peaks, 0.025)
            T_peak_hi = quantile(boot_peaks, 0.975)
            boot_sd   = std(boot_peaks)
            ci_width  = T_peak_hi - T_peak_lo

            println("Peak CI: [", round(T_peak_lo, digits = 2),
                    ", ",         round(T_peak_hi, digits = 2), "] °C")

            # Row-wise std across bootstrap curves
            boot_mat       = reduce(hcat, boot_curves)   # 300 × n_valid
            boot_std_curve = [std(boot_mat[i, :]) for i in 1:300]
        else
            @warn "Fewer than 10 valid bootstrap replicates in $zone_label — CI not computed"
        end
    else
        println("Skipping bootstrap — no valid peak in point estimate")
    end

    wls_se_lo = wls_pred.- boot_std_curve
    wls_se_hi = wls_pred.+ boot_std_curve

    # ── Binned scatter (mean ± SE per °C bin) ─────────────────
    # Guard: bins with n = 1 give std() = NaN → SE = 0.0 instead
    binstats = combine(groupby(df, :Temp_bin)) do sub
        n  = nrow(sub)
        μ  = mean(Float64.(sub.Richness))
        se = n > 1 ? std(Float64.(sub.Richness)) / sqrt(n) : 0.0
        DataFrame(bin_mean = μ, bin_se = se)
    end
    sort!(binstats, :Temp_bin)

    bin_temp = Float64.(binstats.Temp_bin)
    bin_mean = Float64.(binstats.bin_mean)
    bin_se   = Float64.(binstats.bin_se)

    return (
        T_min      = T_min,
        T_max      = T_max,
        pred_TC    = pred_TC,
        wls_pred   = wls_pred,
        wls_se_lo  = wls_se_lo,
        wls_se_hi  = wls_se_hi,
        has_peak   = has_peak,
        T_peak     = T_peak,
        T_peak_lo  = T_peak_lo,
        T_peak_hi  = T_peak_hi,
        boot_sd    = boot_sd,
        ci_width   = ci_width,
        bin_temp   = bin_temp,
        bin_mean   = bin_mean,
        bin_se     = bin_se,
        β          = β,
        R2         = R2,
    )
end

# ── Helper: one figure comparing temperate vs tropical for a habitat subset 
function make_scatter_fig(temp_df, trop_df, t_stats, tr_stats, subtitle)
    cap = 0.15

    fig = Figure(fontsize = 30, size = (2400, 900))
    # Temperate panel
    ax1 = Axis(fig[1, 1],
        xlabel          = "Temperature (°C)",
        ylabel          = "Richness (ATLAS top 10% — temperate, $subtitle)",
        xlabelsize      = 36,
        ylabelsize      = 32,
        topspinevisible = false,
    )

    scatter!(ax1, Float64.(temp_df.Temperature_C_mean),
                  Float64.(temp_df.Richness),
             color = ("#F5D44B", 0.5), markersize = 8)

    scatter!(ax1, t_stats.bin_temp, t_stats.bin_mean,
             color = "#B8960A", markersize = 15,
             strokecolor = :black, strokewidth = 1.0)

    for (x, y, e) in zip(t_stats.bin_temp, t_stats.bin_mean, t_stats.bin_se)
        lines!(ax1, [x, x],             [y - e, y + e], color = "#B8960A", linewidth = 2.5)
        lines!(ax1, [x - cap, x + cap], [y - e, y - e], color = "#B8960A", linewidth = 2.5)
        lines!(ax1, [x - cap, x + cap], [y + e, y + e], color = "#B8960A", linewidth = 2.5)
    end

    # ylims!(ax1, nothing, 4000)

    # Tropical panel
    ax2 = Axis(fig[1, 2],
        xlabel          = "Temperature (°C)",
        ylabel          = "Richness (ATLAS top 10% — tropical, $subtitle)",
        xlabelsize      = 36,
        ylabelsize      = 32,
        topspinevisible = false,
    )

    scatter!(ax2, Float64.(trop_df.Temperature_C_mean),
                  Float64.(trop_df.Richness),
             color = ("#B73508", 0.5), markersize = 8)

    scatter!(ax2, tr_stats.bin_temp, tr_stats.bin_mean,
             color = "#7A2005", markersize = 15,
             strokecolor = :black, strokewidth = 1.0)

    for (x, y, e) in zip(tr_stats.bin_temp, tr_stats.bin_mean, tr_stats.bin_se)
        lines!(ax2, [x, x],             [y - e, y + e], color = "#7A2005", linewidth = 2.5)
        lines!(ax2, [x - cap, x + cap], [y - e, y - e], color = "#7A2005", linewidth = 2.5)
        lines!(ax2, [x - cap, x + cap], [y + e, y + e], color = "#7A2005", linewidth = 2.5)
    end

    # ylims!(ax2, nothing, 4000)

    return fig
end

# =============================================================================
# Helper: draw one panel
# =============================================================================
function draw_panel!(fig, col_idx, result, colour, zone_label, panel_label)

    r       = result
    col_err = colour == "#F5D44B" ? "#B8960A" : colour

    # ── Axis ──────────────────────────────────────────────────
    ax = Axis(fig[1, col_idx],
        xlabel          = "Temperature (°C)",
        ylabel          = "Richness (ATLAS top 10% — $zone_label)",
        xlabelsize      = 36,
        ylabelsize      = 32,
        ygridvisible    = true,
        xgridvisible    = true,
        topspinevisible = false,
    )

    # ── Y limits — NaN-safe ───────────────────────────────────
    valid_mean = filter(!isnan, r.bin_mean)
    valid_se   = filter(!isnan, r.bin_se)
    y_min = minimum(valid_mean) - maximum(valid_se)
    y_max = maximum(valid_mean) + maximum(valid_se)
    pad   = 0.08 * (y_max - y_min)
    ylims!(ax, y_min - pad, y_max + pad)

    # ── WLS curve CI band ─────────────────────────────────────
    band!(ax, r.pred_TC, r.wls_se_lo, r.wls_se_hi,
          color = (colour, 0.25))

    # ── WLS curve ─────────────────────────────────────────────
    lines!(ax, r.pred_TC, r.wls_pred,
           color = (colour, 0.95), linewidth = 4)

    # ── Peak elements — only when β₂ < 0 ─────────────────────
    if r.has_peak && !isnan(r.T_peak_lo)
        # CI shaded band
        vspan!(ax, r.T_peak_lo, r.T_peak_hi,
               color = (colour, 0.20))

        # Peak dashed line
        vlines!(ax, [r.T_peak],
                color = (colour, 0.95), linewidth = 2.5, linestyle = :dash)

        # Peak temperature label
        text!(ax, r.T_peak + 0.4,
              y_min + 0.05 * (y_max - y_min),
              text     = "$(round(r.T_peak, digits=1))°C",
              color    = col_err,
              fontsize = 24, align = (:left, :bottom))

    elseif r.has_peak && isnan(r.T_peak_lo)
        # Peak exists but CI could not be computed (too few valid boots)
        vlines!(ax, [r.T_peak],
                color = (colour, 0.95), linewidth = 2.5, linestyle = :dash)

        text!(ax, r.T_peak + 0.4,
              y_min + 0.05 * (y_max - y_min),
              text     = "$(round(r.T_peak, digits=1))°C (no CI)",
              color    = col_err,
              fontsize = 24, align = (:left, :bottom))
    else
        # No peak — annotate clearly
        text!(ax, r.T_min + 0.5,
              y_min + 0.05 * (y_max - y_min),
              text     = "no peak detected", #  (R² = $(round(r.R2, digits=3)))
              color    = :grey50,
              fontsize = 22, align = (:left, :bottom))
    end

    # ── Binned scatter points ─────────────────────────────────
    scatter!(ax, r.bin_temp, r.bin_mean,
             color       = (colour, 1.0),
             markersize  = 14,
             strokecolor = :grey40,
             strokewidth = 0.8)

    # ── Error bars on scatter ─────────────────────────────────
    cap = 0.15
    for (x, y, e) in zip(r.bin_temp, r.bin_mean, r.bin_se)
        lines!(ax, [x, x],             [y - e, y + e], color = col_err, linewidth = 2)
        lines!(ax, [x - cap, x + cap], [y - e, y - e], color = col_err, linewidth = 2)
        lines!(ax, [x - cap, x + cap], [y + e, y + e], color = col_err, linewidth = 2)
    end

    # ── R² annotation ─────────────────────────────────────────
    text!(ax, r.T_min + 0.5,
          y_max - 0.05 * (y_max - y_min),
        #   text     = "R² = $(round(r.R2, digits=3))",
          color    = :grey30,
          fontsize = 24, align = (:left, :top))

    # ── Legend ────────────────────────────────────────────────
    l_curve = LineElement(color = (colour, 0.95), linewidth = 4)
    s_pts   = MarkerElement(color       = (colour, 1.0),
                            markersize  = 14,
                            marker      = :circle,
                            strokecolor = :grey40,
                            strokewidth = 0.8)
    Legend(fig[1, col_idx], [s_pts, l_curve],
           ["Richness (top 10% — $zone_label)", "WLS fit on 1/kT + pH"],
           tellheight   = false,
           tellwidth    = false,
           halign       = :right,
           valign       = :top,
           framevisible = false)

    # ── Panel label ───────────────────────────────────────────
    Label(fig[1, col_idx, TopLeft()], panel_label,
          fontsize = 36, font = :bold, padding = (0, 0, 8, 0))

    return ax
end


