# =============================================================================
# EMP_rich.jl
#
# Loads and visualises Earth Microbiome Project (EMP) richness data across
# a temperature gradient, fits a Gaussian envelope to the 97th-percentile
# richness, and saves a single publication-quality figure as EMP.pdf.
#
# Input:  data/EMP_filtered.csv   — pre-filtered EMP richness-temperature data
#                                   (columns: Temp, Richness)
# Output: result/EMP.pdf          — scatter + Gaussian fit + mean ± SE plot
#
# Usage:
#   include("code/Base/sim_frame.jl")
#   include("code/Analysis/EMP_rich.jl")
# =============================================================================

include("./Base/sim_frame.jl")

# =============================================================================
# Load EMP data
# =============================================================================

EMP_data = CSV.read("../data/EMP_filtered.csv", DataFrame, header = true)

# =============================================================================
# Bin by integer temperature and extract 97th-percentile envelope
# =============================================================================

# Round each sample's temperature to the nearest integer for binning
filtered_data           = copy(EMP_data)
filtered_data.round_Temp = round.(Int, filtered_data.Temp)

bins        = unique(filtered_data.round_Temp)   # one bin per integer °C
filtered_df = DataFrame()                         # rows above 97th percentile
EMP_meanerr = zeros(Float64, length(bins), 3)     # columns: Temp, mean, SE
n = 0

for unique_bin in bins
    n += 1
    bin_data = filter(row -> row.round_Temp == unique_bin, filtered_data)

    # mean and standard error of richness within this bin
    mean_bin = mean(bin_data.Richness)
    err_bin  = std(bin_data.Richness) / sqrt(length(bin_data.Richness))

    # keep only rows above the 97th-percentile richness cutoff
    q97      = quantile(bin_data.Richness, 0.97)
    filtered_rows = filter(row -> row.Richness > q97, bin_data)
    append!(filtered_df, filtered_rows)

    EMP_meanerr[n, :] = [unique_bin, mean_bin, err_bin]
end

# Wrap in a DataFrame and sort by temperature
col_names_EMP = ["Temp", "mean", "err"]
EMP_meanerr   = DataFrame(EMP_meanerr, col_names_EMP)
sort!(EMP_meanerr, :Temp)

# =============================================================================
# Fit Gaussian to 97th-percentile envelope
# =============================================================================

"""
    gaussian(x, params)

Evaluates a Gaussian curve at `x`:
    f(x) = a * exp( -(x - μ)² / (2σ²) )

where `params = [a, μ, σ]`.
"""
gaussian(x, params) = params[1] .* exp.(-(x .- params[2]).^2 ./ (2 .* params[3]^2))

x_data = filtered_df[:, "Temp"]
y_data = filtered_df[:, "Richness"]

# Initial guesses: amplitude = max richness, mean = temperature of max, width = range / 5
p0 = [maximum(y_data),
      x_data[argmax(y_data)],
      (maximum(x_data) - minimum(x_data)) / 5]

fit_result      = curve_fit(gaussian, x_data, y_data, p0)
best_fit_params = coef(fit_result)
std_errors      = stderror(fit_result)

println("Gaussian fit parameters:")
println("  Amplitude  a = ", round(best_fit_params[1], digits = 2),
        " ± ", round(std_errors[1], digits = 2))
println("  Peak temp  μ = ", round(best_fit_params[2], digits = 2),
        " ± ", round(std_errors[2], digits = 2), " °C")
println("  Width      σ = ", round(best_fit_params[3], digits = 2),
        " ± ", round(std_errors[3], digits = 2), " °C")

# Dense x-axis for smooth fitted curve
x_fit = range(minimum(x_data), stop = maximum(x_data), length = 200)
y_fit = gaussian(x_fit, best_fit_params)

# =============================================================================
# Figure: EMP richness scatter + 97th-percentile envelope + Gaussian fit +
#         mean ± SE per integer bin
# =============================================================================

CairoMakie.activate!(type = "pdf")

f  = Figure(fontsize = 35, size = (1400, 900));
ax = Axis(f[1, 1],
    xlabel        = "Temperature (°C)",
    ylabel        = "Richness (EMP)",
    xlabelsize    = 50,
    ylabelsize    = 50,
    ygridvisible  = true,
    xgridvisible  = true)

# All EMP samples — light background scatter
CairoMakie.scatter!(ax, EMP_data.Temp, EMP_data.Richness,
    color      = "#7BA9BE",
    markersize = 12,
    alpha      = 0.2)

# 97th-percentile samples — darker foreground scatter
CairoMakie.scatter!(ax, filtered_df.Temp, filtered_df.Richness,
    color      = "#285C93",
    markersize = 12,
    alpha      = 0.8)

# Per-bin mean ± SE as points with error bars
CairoMakie.scatter!(ax, EMP_meanerr.Temp, EMP_meanerr.mean,
    color      = "#E17542",
    markersize = 15,
    alpha      = 1.0)

cap = 0.1   # half-width of horizontal caps on error bars (°C)
for (x, y, e) in zip(EMP_meanerr.Temp, EMP_meanerr.mean, EMP_meanerr.err)
    lines!(ax, [x, x],         [y - e, y + e],         color = "#E17542", linewidth = 2)
    lines!(ax, [x - cap, x + cap], [y - e, y - e],     color = "#E17542", linewidth = 2)
    lines!(ax, [x - cap, x + cap], [y + e, y + e],     color = "#E17542", linewidth = 2)
end

# Gaussian envelope fit
lines!(ax, x_fit, y_fit,
    color     = ("#285C93", 0.7),
    linewidth = 5)

# Legend
s_all    = [MarkerElement(color = ("#7BA9BE", 0.5), markersize = 12, marker = :circle)]
s_env    = [MarkerElement(color = ("#285C93", 0.8), markersize = 12, marker = :circle)]
l_gauss  = [LineElement(color   = ("#285C93", 0.7), linewidth  = 5)]
s_mean   = [MarkerElement(color = ("#E17542", 1.0), markersize = 15, marker = :circle)]

Legend(f[1, 1],
    [s_all, s_env, l_gauss, s_mean],
    ["All EMP samples", "EMP 97th percentile", "Gaussian fit", "Mean richness ± SE"],
    tellheight = false,
    tellwidth  = false,
    halign     = :right,
    valign     = :top)

# =============================================================================
# Save
# =============================================================================

save("../results/EMP.pdf", f)
println("Saved: results/EMP.pdf")
