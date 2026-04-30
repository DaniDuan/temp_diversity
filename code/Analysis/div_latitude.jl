# =============================================================================
# div_latitude.jl
#
# Plots ATLAS richness vs temperature separately for:
#   - Temperate region (|latitude| >  30°)   colour #F5D44B   panel (a)
#   - Tropical  region (|latitude| <= 30°)   colour #B73508   panel (b)
#
# Loads:
#   top10_tropical.csv   — top 10% richness samples in tropical  zone
#   top10_temperate.csv  — top 10% richness samples in temperate zone
#
# WLS Model (per zone):
#   Richness ~ β₀ + β₁·(1/kT) + β₂·(1/kT)² + β₃·pH
#   Temperature and pH treated as independent predictors
#   Missing pH filled with 0.0 (mean-centred) so those rows
#   contribute to the temperature curve but not the pH coefficient
#   Peak only reported when β₂ < 0 (concave-down parabola)
#
# Output:
#../results/diversity_latitude_atlas.pdf
# =============================================================================

include("./Analysis/ana_atlas_function.jl")
# =============================================================================
# Load zone data
# =============================================================================
tropical_df  = CSV.read("../data/top10_tropical.csv",  DataFrame)
temperate_df = CSV.read("../data/top10_temperate.csv", DataFrame)

# =============================================================================
# Run analysis for both zones
# =============================================================================
temp = analyse_zone(temperate_df, "Temperate")
trop = analyse_zone(tropical_df,  "Tropical")

# =============================================================================
# Figure
# =============================================================================
fig = Figure(fontsize = 30, size = (2400, 900));

ax_temp = draw_panel!(fig, 1, temp, col_temp, "temperate", "(a)")
ax_trop = draw_panel!(fig, 2, trop, col_trop, "tropical",  "(b)")

# =============================================================================
# Save
# =============================================================================
# save("../results/diversity_latitude_atlas.pdf", fig)

println("\nSaved →../results/diversity_latitude_atlas.pdf")

println("\n── Temperate ──")
println("  R²      = ", round(temp.R2, digits = 3))
println("  β       = [", join(round.(temp.β, digits = 3), ", "), "]")
if temp.has_peak
    println("  Peak T  = ", round(temp.T_peak,    digits = 2), "°C")
    println("  Peak CI = [", round(temp.T_peak_lo, digits = 2),
            ", ",             round(temp.T_peak_hi, digits = 2), "]")
else
    println("  Peak T  = none (concave up)")
end

println("\n── Tropical ──")
println("  R²      = ", round(trop.R2, digits = 3))
println("  β       = [", join(round.(trop.β, digits = 3), ", "), "]")
if trop.has_peak
    println("  Peak T  = ", round(trop.T_peak,    digits = 2), "°C")
    println("  Peak CI = [", round(trop.T_peak_lo, digits = 2),
            ", ",             round(trop.T_peak_hi, digits = 2), "]")
else
    println("  Peak T  = none (concave up)")
end

