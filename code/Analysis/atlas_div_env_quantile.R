# ============================================================
# atlas_div_env.R
# ============================================================

setwd("/Users/Danica/Documents/temp_diversity/code")

if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
library(rhdf5)

# ===========================================================
# PART 1: Load Pre-computed Richness
# ===========================================================
richness_df <- read.csv("../data/sample_richness.csv", check.names = FALSE)

# ===========================================================
# PART 2: Load Mapping, Temperature, Metadata
# ===========================================================
mapping  <- read.delim("../data/community_cluster_mapping.tsv",       check.names = FALSE)
env_vars <- read.delim("../data/community_cluster_env_variables.tsv", check.names = FALSE)
metadata <- read.delim("../data/community_cluster_metadata.tsv",      check.names = FALSE)

env_with_temp <- env_vars[!is.na(env_vars$Temperature_C_mean), ]

# ===========================================================
# PART 3: Join Everything
# ===========================================================
final_df <- merge(richness_df, mapping, by = "SampleID", all.x = FALSE)
final_df <- merge(final_df,
                  env_with_temp[, c("ComClustID", "Temperature_C_mean", "pH_mean")],
                  by = "ComClustID", all.x = FALSE)
final_df <- merge(final_df, metadata[, c("ComClustID", "Env_cons")],
                  by = "ComClustID", all.x = FALSE)
final_df <- final_df[, c("SampleID", "ComClustID", "Env_cons",
                         "Temperature_C_mean", "pH_mean", "Richness")]
final_df <- final_df[order(final_df$Temperature_C_mean), ]
row.names(final_df) <- NULL

# ===========================================================
# PART 4: Filter Habitats
# ===========================================================
exclude_habitats <- c(
  "animal", "bird", "cattle", "dog", "fish", "fly",
  "horse", "human", "mosquito", "mouse", "pig", "rat", "sheep",
  "plant", "rhizosphere", "marine", "ocean", "sea"
)

contains_excluded <- sapply(final_df$Env_cons, function(x) {
  habitats <- trimws(unlist(strsplit(x, ";")))
  any(habitats %in% exclude_habitats)
})
final_df <- final_df[!contains_excluded, ]
cat("Samples after habitat filtering:", nrow(final_df), "\n")

# ===========================================================
# PART 5: Filter Temperature 0-30°C & Bin
# ===========================================================
final_df <- final_df[final_df$Temperature_C_mean >= 0 &
                       final_df$Temperature_C_mean <= 30, ]
final_df$Temp_bin <- round(final_df$Temperature_C_mean)
cat("Samples after temperature filtering:", nrow(final_df), "\n")

# ===========================================================
# PART 6: pH Correlation Analysis
# ===========================================================
cat("\n=== pH & Temperature Correlation Report ===\n")
df_ph <- final_df[!is.na(final_df$pH_mean), ]
cat(sprintf("Total samples:        %d\n", nrow(final_df)))
cat(sprintf("Samples with pH:      %d (%.1f%%)\n",
            nrow(df_ph), 100 * nrow(df_ph) / nrow(final_df)))

cor_pearson  <- cor.test(df_ph$Temperature_C_mean, df_ph$pH_mean, method = "pearson")
cor_spearman <- cor.test(df_ph$Temperature_C_mean, df_ph$pH_mean, method = "spearman")
cat(sprintf("Temp-pH Pearson r:    %.4f (p = %.4f)\n",
            cor_pearson$estimate, cor_pearson$p.value))
cat(sprintf("Temp-pH Spearman rho: %.4f (p = %.4f)\n",
            cor_spearman$estimate, cor_spearman$p.value))

cor_pr <- cor.test(df_ph$pH_mean, df_ph$Richness, method = "pearson")
cat(sprintf("pH-Richness Pearson r: %.4f (p = %.4f)\n",
            cor_pr$estimate, cor_pr$p.value))

lm_fit  <- lm(Richness ~ Temperature_C_mean + pH_mean, data = df_ph)
cat("\n=== Multiple Regression: Richness ~ T + pH ===\n")
print(summary(lm_fit))

######
df_sens <- final_df[!is.na(final_df$pH_mean), ]
lm_sens <- lm(Richness ~ I(1/(8.617e-5*(Temperature_C_mean+273.15))) + 
                I((1/(8.617e-5*(Temperature_C_mean+273.15)))^2) + 
                pH_mean, data = df_sens)
summary(lm_sens)

cor_pearson  <- cor.test((1/(8.617e-5*(df_sens$Temperature_C_mean+273.15)))^2, df_sens$pH_mean, method = "pearson")
cor_spearman <- cor.test((1/(8.617e-5*(df_sens$Temperature_C_mean+273.15)))^2, df_sens$pH_mean, method = "spearman")
cat(sprintf("Temp-pH Pearson r:    %.4f (p = %.4f)\n",
            cor_pearson$estimate, cor_pearson$p.value))
cat(sprintf("Temp-pH Spearman rho: %.4f (p = %.4f)\n",
            cor_spearman$estimate, cor_spearman$p.value))

lm_final <- lm(Richness ~ I(1/(8.617e-5*(Temperature_C_mean+273.15))) +
                 I((1/(8.617e-05*(Temperature_C_mean+273.15)))^2),
               data = final_df)
summary(lm_final)
# ===========================================================
# PART 7: Top 10% Per Bin
# ===========================================================
top10_df <- do.call(rbind, lapply(split(final_df, final_df$Temp_bin), function(bin_data) {
  threshold <- quantile(bin_data$Richness, 0.90)
  bin_data[bin_data$Richness >= threshold, ]
}))
top10_df <- top10_df[order(top10_df$Temperature_C_mean), ]
row.names(top10_df) <- NULL
cat(sprintf("\nTotal samples in top 10%%: %d\n", nrow(top10_df)))

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
cat("\nPer-bin summary of top 10%:\n")
print(top10_bin_summary, row.names = FALSE)

# a quick table of raw counts
cat("\nRaw counts per bin:\n")
print(table(top10_df$Temp_bin))
# ===========================================================
# PART 8: WLS Fit
# ===========================================================
top10_ph <- top10_df[!is.na(top10_df$pH_mean), ]
bin_counts      <- table(top10_ph$Temp_bin)
top10_ph$weight <- as.numeric(bin_counts[as.character(top10_ph$Temp_bin)])

wls_no_int <- lm(Richness ~ Temperature_C_mean + pH_mean,
                 data = top10_ph, weights = weight)
wls_int    <- lm(Richness ~ Temperature_C_mean * pH_mean,
                 data = top10_ph, weights = weight)

cat("\n=== WLS no interaction ===\n"); print(summary(wls_no_int))
cat("\n=== WLS with interaction ===\n"); print(summary(wls_int))

p_int      <- summary(wls_int)$coefficients["Temperature_C_mean:pH_mean", "Pr(>|t|)"]
best_model <- if (p_int < 0.05) wls_int else wls_no_int
cat(sprintf("\nInteraction p = %.4f — using: %s\n", p_int,
            ifelse(p_int < 0.05, "with interaction", "no interaction")))
print(coef(best_model))

# ===========================================================
# PART 9: Save
# ===========================================================
write.csv(final_df,          "../data/summary_atlas.csv",         row.names = FALSE)
write.csv(top10_df,          "../data/top10_atlas.csv",           row.names = FALSE)
write.csv(top10_ph,          "../data/top10_atlas_ph.csv",        row.names = FALSE)
write.csv(top10_bin_summary, "../data/top10_atlas_binstats.csv",  row.names = FALSE)
write.csv(as.data.frame(t(coef(best_model))), "../data/wls_coefs.csv", row.names = FALSE)
cat("\nAll files saved.\n")

