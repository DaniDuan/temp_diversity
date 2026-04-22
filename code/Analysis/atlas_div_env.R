# ============================================================
# atlas_div_env.R
# Description: Load per-sample richness from BIOM file,
#              match with temperature + habitat + latitude +
#              pH data, filter to AMPLICON samples only,
#              and export summary for plotting in Julia.
#              Also plots richness vs pH.
# ============================================================

setwd("/Users/Danica/Documents/temp_diversity/code")

# --- Libraries ----------------------------------------------
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("biomformat")
if (!requireNamespace("rhdf5",       quietly = TRUE)) BiocManager::install("rhdf5")
library(rhdf5)
library(biomformat)
library(ggplot2)

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
# PART 2: Load Mapping, Temperature, Metadata, Latitude & pH
# ===========================================================
cat("Loading mapping and environmental data...\n")

mapping  <- read.delim("../data/community_cluster_mapping.tsv",       check.names = FALSE)
env_vars <- read.delim("../data/community_cluster_env_variables.tsv", check.names = FALSE)
metadata <- read.delim("../data/community_cluster_metadata.tsv",      check.names = FALSE)

# --- Load latitude/longitude data ---------------------------
# na.strings covers all observed missing-value tokens in this file:
#   "None"    — Python/pandas-style None exported as string
#   "missing" — explicit missing label in LatFieldValue
#   ""        — empty cells
latlon <- read.delim(
  "../data/samples.info.latlon.parsed.tsv",
  check.names = FALSE,
  na.strings  = c("NA", "None", "missing", "")
)

# Retain only the clean numeric latitude column and rename for joining
latlon <- latlon[, c("Sample", "LatitudeParsed")]
colnames(latlon) <- c("BioSampleID", "Latitude")
latlon$Latitude  <- as.numeric(latlon$Latitude)

cat("Latitude coverage:", sum(!is.na(latlon$Latitude)),
    "of", nrow(latlon), "samples have a valid latitude.\n")

# --- Auto-detect pH column name in env_vars -----------------
cat("Columns in env_vars:\n", paste(colnames(env_vars), collapse = ", "), "\n\n")
PH_COL <- grep("pH|ph|PH", colnames(env_vars), value = TRUE)[1]
cat("Using pH column:", PH_COL, "\n")

# --- Pull pH and temperature as separate subsets ------------
# Kept separate so that samples with pH but no temperature are
# not dropped before the pH plot
env_ph   <- env_vars[, c("ComClustID", PH_COL)]
env_temp <- env_vars[!is.na(env_vars$Temperature_C_mean),
                     c("ComClustID", "Temperature_C_mean")]

# ===========================================================
# PART 3: Join Richness + Habitat + SeqTech + Latitude + pH
# ===========================================================
final_df <- merge(richness_df, mapping, by = "SampleID", all.x = FALSE)
final_df <- merge(final_df, env_ph,     by = "ComClustID", all.x = FALSE)

# --- Join Env_cons and SeqTech_cons from metadata -----------
final_df <- merge(final_df,
                  metadata[, c("ComClustID", "Env_cons", "SeqTech_cons")],
                  by = "ComClustID", all.x = FALSE)

# --- Extract BioSample ID from compound SampleID ------------
# SampleIDs are formatted as "SRR10095609.SRS5370006" —
# the BioSample accession (SRS/ERS/DRS) is always to the right of the dot.
# If there is no dot the full SampleID is used as a fallback.
final_df$BioSampleID <- sub(".*\\.", "", final_df$SampleID)

cat("BioSampleID extraction check (first 6):\n")
print(head(final_df[, c("SampleID", "BioSampleID")], 6))
cat("Overlap with latlon after extraction:",
    sum(final_df$BioSampleID %in% latlon$BioSampleID), "samples\n\n")

# --- Join latitude on BioSampleID ---------------------------
final_df <- merge(final_df, latlon, by = "BioSampleID", all.x = TRUE)

# --- Standardise pH column name -----------------------------
colnames(final_df)[colnames(final_df) == PH_COL] <- "pH"

final_df <- final_df[, c("SampleID", "BioSampleID", "ComClustID", "Env_cons",
                         "SeqTech_cons", "Latitude", "pH", "Richness")]
row.names(final_df) <- NULL

cat("Samples after joining all variables:", nrow(final_df), "\n")
cat("  Missing latitude:", sum(is.na(final_df$Latitude)), "\n")
cat("  Missing pH:      ", sum(is.na(final_df$pH)),       "\n")
cat("  SeqTech breakdown:\n")
print(table(final_df$SeqTech_cons, useNA = "ifany"))

# ===========================================================
# PART 4: Filter Sequencing Method — AMPLICON only
# ===========================================================
# Exclude WGS (not comparable to OTU richness) and blanks
# (unknown method — cannot be confidently assigned)
final_df <- final_df[final_df$SeqTech_cons == "AMPLICON", ]
cat("Samples after filtering to AMPLICON only:", nrow(final_df), "\n")

# ===========================================================
# PART 5: Filter Habitats
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
# PART 6: Plot Richness vs pH
# (before temperature filtering — retains full pH range)
# ===========================================================
cat("Plotting Richness vs pH...\n")

ph_df <- final_df[!is.na(final_df$pH), ]
cat("Samples with valid pH for plotting:", nrow(ph_df), "\n")

p_ph <- ggplot(ph_df, aes(x = pH, y = Richness)) +
  
  geom_point(colour = "black", alpha = 0.1, size = 1.8, shape = 16) +
  
  labs(
    # title    = "Richness vs pH",
    # subtitle = paste0("n = ", nrow(ph_df), " samples | Host-associated habitats excluded"),
    x = "pH",
    y = "Richness"
  ) +
  theme_bw(base_size = 20) +
  theme(
    plot.title       = element_text(face = "bold", size = 15),
    plot.subtitle    = element_text(size = 10, colour = "grey40"),
    panel.grid.minor = element_blank()
  )

print(p_ph)

pdf("../results/richness_pH.pdf", width = 9, height = 6, pointsize = 1)
print(p_ph)
dev.off()
cat("Plot saved to../results/richness_pH.pdf\n")

# ===========================================================
# PART 7: Join Temperature & Filter to 0-30°C
# ===========================================================

# --- Join temperature (drops samples with no temperature data) --------------
final_df <- merge(final_df, env_temp, by = "ComClustID", all.x = FALSE)
final_df <- final_df[, c("SampleID", "BioSampleID", "ComClustID", "Env_cons",
                         "SeqTech_cons", "Temperature_C_mean",
                         "Latitude", "pH", "Richness")]
final_df <- final_df[order(final_df$Temperature_C_mean), ]
row.names(final_df) <- NULL
cat("Samples after joining temperature:", nrow(final_df), "\n")

# --- Filter temperature range -----------------------------------------------
final_df <- final_df[final_df$Temperature_C_mean >= 0 &
                       final_df$Temperature_C_mean <= 30, ]
final_df$Temp_bin <- round(final_df$Temperature_C_mean)
cat("Samples after temperature filtering:", nrow(final_df), "\n")

# ===========================================================
# PART 8: Save for Julia
# ===========================================================
write.csv(final_df, "../data/summary_atlas.csv", row.names = FALSE)
cat("Saved to../data/summary_atlas.csv\n")
cat("Columns:", paste(colnames(final_df), collapse = ", "), "\n")
