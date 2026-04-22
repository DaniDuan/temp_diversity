# ============================================================
# atlas_div_env_quantile.R
# Description: Using summary_atlas.csv from atlas_div_env.R,
#              join latitude, assign latitude regions, compute
#              top 10% richness per temp bin, report Temp-pH
#              correlation, and export data files for Julia.
# ============================================================

setwd("/Users/Danica/Documents/temp_diversity/code")

# ===========================================================
# PART 1: Load Pre-computed Summary from atlas_div_env.R
# ===========================================================
cat("Loading summary_atlas.csv...\n")
final_df <- read.csv("../data/summary_atlas.csv", check.names = FALSE)
cat("Samples loaded:", nrow(final_df), "\n")
cat("Columns:", paste(colnames(final_df), collapse = ", "), "\n\n")

# ===========================================================
# PART 2: Assign Latitude Regions
# ===========================================================
# Latitude is already in final_df from atlas_div_env.R
# Uses absolute latitude so Southern Hemisphere is handled correctly:
#   0–30°  → Tropics/Subtropics
#   30–60° → Temperate
#   60–90° → Polar/Subpolar
#   NA     → Unknown

cat("Assigning latitude regions...\n")

final_df$AbsLatitude <- abs(final_df$Latitude)

final_df$LatRegion <- cut(
  final_df$AbsLatitude,
  breaks         = c(0, 30, 60, 90),
  labels         = c("Tropics_Subtropics", "Temperate", "Polar_Subpolar"),
  include.lowest = TRUE,
  right          = FALSE   # intervals: [0,30), [30,60), [60,90]
)
final_df$LatRegion <- as.character(final_df$LatRegion)
final_df$LatRegion[is.na(final_df$Latitude)] <- "Unknown"

cat("Latitude region breakdown:\n")
print(table(final_df$LatRegion, useNA = "ifany"))
cat("\n")

# ===========================================================
# PART 3: pH–Temperature Correlation Report
# ===========================================================
cat("=== pH & Temperature Correlation Report ===\n")

df_ph <- final_df[!is.na(final_df$pH), ]
cat(sprintf("Total samples:        %d\n",   nrow(final_df)))
cat(sprintf("Samples with pH:      %d (%.1f%%)\n",
            nrow(df_ph), 100 * nrow(df_ph) / nrow(final_df)))

cor_pearson  <- cor.test(df_ph$Temperature_C_mean, df_ph$pH,
                         method = "pearson")
cor_spearman <- cor.test(df_ph$Temperature_C_mean, df_ph$pH,
                         method = "spearman")

cat(sprintf("\nTemp-pH Pearson r:    %.4f (p = %.4g)\n",
            cor_pearson$estimate,  cor_pearson$p.value))
cat(sprintf("Temp-pH Spearman rho: %.4f (p = %.4g)\n",
            cor_spearman$estimate, cor_spearman$p.value))

# --- Automatic multicollinearity conclusion -----------------
r_val <- abs(cor_pearson$estimate)
if (r_val < 0.20) {
  cat(sprintf(paste0("\nConclusion: Temp-pH correlation is negligible (|r| = %.4f).\n",
                     "Temperature and pH are effectively independent predictors.\n",
                     "No multicollinearity correction needed in downstream analysis.\n"), r_val))
} else if (r_val < 0.50) {
  cat(sprintf(paste0("\nConclusion: Temp-pH correlation is weak (|r| = %.4f).\n",
                     "Predictors are largely independent — proceed with caution.\n"), r_val))
} else {
  cat(sprintf(paste0("\nWARNING: Temp-pH correlation is moderate-strong (|r| = %.4f).\n",
                     "Consider multicollinearity correction in downstream analysis.\n"), r_val))
}
cat("\n")

# ===========================================================
# PART 4: Top 10% Richness Per Temperature Bin
# ===========================================================
cat("Computing top 10% richness per temperature bin...\n")

top10_df <- do.call(rbind, lapply(split(final_df, final_df$Temp_bin), function(bin_data) {
  threshold <- quantile(bin_data$Richness, 0.90)
  bin_data[bin_data$Richness >= threshold, ]
}))
top10_df <- top10_df[order(top10_df$Temperature_C_mean), ]
row.names(top10_df) <- NULL
cat(sprintf("Total samples in top 10%%: %d\n\n", nrow(top10_df)))

# --- Per-bin summary ----------------------------------------
top10_bin_summary <- do.call(rbind, lapply(sort(unique(top10_df$Temp_bin)), function(t) {
  sub <- top10_df[top10_df$Temp_bin == t, ]
  data.frame(
    Temp_bin = t,
    N        = nrow(sub),
    Mean     = round(mean(sub$Richness), 1),
    SE       = round(sd(sub$Richness) / sqrt(nrow(sub)), 2),
    Min      = min(sub$Richness),
    Max      = max(sub$Richness)
  )
}))
cat("Per-bin summary of top 10%:\n")
print(top10_bin_summary, row.names = FALSE)

cat("\nRaw counts per bin:\n")
print(table(top10_df$Temp_bin))

# --- Latitude region breakdown within top 10% ---------------
cat("\nLatitude region breakdown within top 10%:\n")
print(table(top10_df$LatRegion, useNA = "ifany"))

# ===========================================================
# PART 5: Save All Data Files for Julia
# ===========================================================
cat("\nSaving data files for Julia...\n")

# Full filtered dataset (all samples, with latitude regions)
write.csv(final_df,
          "../data/summary_atlas.csv",
          row.names = FALSE)
cat("  Saved:../data/summary_atlas.csv\n")

# Top 10% richness per bin (with latitude regions)
write.csv(top10_df,
          "../data/top10_atlas.csv",
          row.names = FALSE)
cat("  Saved:../data/top10_atlas.csv\n")

# Per-bin summary statistics of top 10%
write.csv(top10_bin_summary,
          "../data/top10_atlas_binstats.csv",
          row.names = FALSE)
cat("  Saved:../data/top10_atlas_binstats.csv\n")

cat("\nDone. All files ready for Julia.\n")
cat("Columns in summary_atlas.csv:  ", paste(colnames(final_df),  collapse = ", "), "\n")
cat("Columns in top10_atlas.csv:    ", paste(colnames(top10_df),  collapse = ", "), "\n")
