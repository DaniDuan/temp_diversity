# ============================================================
# atlas_div_env.R
# Description: Load per-sample richness from BIOM file,
#              match with temperature + habitat data, filter,
#              and export summary for plotting in Julia.
# ============================================================

setwd("/Users/Danica/Documents/temp_diversity/code")

# --- Libraries ----------------------------------------------
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("biomformat")
if (!requireNamespace("rhdf5",       quietly = TRUE)) BiocManager::install("rhdf5")
library(rhdf5)
library(biomformat)

# ===========================================================
# PART 1: Extract Richness from BIOM File
# ===========================================================
biom_file <- "../data/samples-otus.97.mapped.metag.minfilter.refilt.biom"
h5ls(biom_file)  # Inspect structure

cat("Reading sample IDs and index pointers...\n")
sample_ids <- h5read(biom_file, "/sample/ids")
indptr     <- h5read(biom_file, "/sample/matrix/indptr")
n_samples  <- length(sample_ids)
cat("Total samples:", n_samples, "\n")

cat("Calculating richness per sample...\n")
CHUNK_SIZE <- 500
richness   <- integer(n_samples)

for (chunk_start in seq(1, n_samples, by = CHUNK_SIZE)) {
  chunk_end <- min(chunk_start + CHUNK_SIZE - 1, n_samples)
  cat(sprintf("  Processing samples %d to %d of %d...\n", chunk_start, chunk_end, n_samples))
  
  for (i in chunk_start:chunk_end) {
    start <- indptr[i] + 1
    end   <- indptr[i + 1]
    if (end < start) { richness[i] <- 0L; next }
    vals        <- h5read(biom_file, "/sample/matrix/data", index = list(start:end))
    richness[i] <- sum(vals > 0)
  }
}
H5close()

richness_df <- data.frame(SampleID = sample_ids, Richness = richness, row.names = NULL)
cat("Richness calculated for", nrow(richness_df), "samples.\n")
write.csv(richness_df, "../data/sample_richness.csv", row.names = FALSE)
cat("Richness saved to../data/sample_richness.csv\n\n")

# # --- Load pre-computed richness -----------------------------
# richness_df <- read.csv("../data/sample_richness.csv", check.names = FALSE)

# ===========================================================
# PART 2: Load Mapping, Temperature & Metadata
# ===========================================================
cat("Loading mapping and environmental data...\n")

mapping  <- read.delim("../data/community_cluster_mapping.tsv",       check.names = FALSE)
env_vars <- read.delim("../data/community_cluster_env_variables.tsv", check.names = FALSE)
metadata <- read.delim("../data/community_cluster_metadata.tsv",      check.names = FALSE)

env_with_temp <- env_vars[!is.na(env_vars$Temperature_C_mean), ]

# ===========================================================
# PART 3: Join Richness + Temperature + Habitat
# ===========================================================
final_df <- merge(richness_df, mapping,  by = "SampleID",   all.x = FALSE)
final_df <- merge(final_df, env_with_temp[, c("ComClustID", "Temperature_C_mean")],
                  by = "ComClustID", all.x = FALSE)
final_df <- merge(final_df, metadata[, c("ComClustID", "Env_cons")],
                  by = "ComClustID", all.x = FALSE)
final_df <- final_df[, c("SampleID", "ComClustID", "Env_cons", "Temperature_C_mean", "Richness")]
final_df <- final_df[order(final_df$Temperature_C_mean), ]
row.names(final_df) <- NULL

# ===========================================================
# PART 4: Filter Habitats
# ===========================================================
exclude_habitats <- c(
  # Host-associated (immune/diet controlled — not free resource competition)
  "animal", "bird", "cattle", "dog", "fish", "fly",
  "horse", "human", "mosquito", "mouse", "pig", "rat", "sheep")

contains_excluded <- sapply(final_df$Env_cons, function(x) {
  habitats <- trimws(unlist(strsplit(x, ";")))
  any(habitats %in% exclude_habitats)
})

final_df <- final_df[!contains_excluded, ]
cat("Samples after habitat filtering:", nrow(final_df), "\n")

# ===========================================================
# PART 5: Filter Temperature 0-30°C
# ===========================================================
final_df <- final_df[final_df$Temperature_C_mean >= 0 &
                       final_df$Temperature_C_mean <= 30, ]
final_df$Temp_bin <- round(final_df$Temperature_C_mean)
cat("Samples after temperature filtering:", nrow(final_df), "\n")

# ===========================================================
# PART 6: Save for Julia
# ===========================================================
write.csv(final_df, "../data/summary_atlas.csv", row.names = FALSE)
cat("Saved to../data/summary_atlas.csv\n")
cat("Columns:", paste(colnames(final_df), collapse = ", "), "\n")
