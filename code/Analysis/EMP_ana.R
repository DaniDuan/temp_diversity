# =============================================================================
# EMP_ana.R
#
# Loads and filters the Earth Microbiome Project (EMP) dataset, computes
# relative species richness, and extracts the 99th percentile of relative
# richness per temperature for use as an empirical richness envelope.
#
# Input:  data/EMP.csv
# Output: data/EMP_filtered.csv
#
# Location: code/Analysis/EMP_ana.R
# Data:     data/ (parallel to code/)
# =============================================================================

# =============================================================================
# Load and clean data
# =============================================================================
EMP_data <- read.csv("../../data/EMP.csv", header = TRUE)
names(EMP_data) <- c("ID", "Sample", "Temp", "pH", "Richness")

# retain samples with valid temperature readings in the 0–30°C range
EMP_data = subset(EMP_data,Temp != "" & Temp <= 30 & Temp > 0)

# quick visual check of raw richness vs temperature
scatter.smooth(EMP_data$Temp, EMP_data$Richness,
               xlab = "Temperature (°C)", ylab = "Richness")

# =============================================================================
# Relative richness
# =============================================================================
# normalise richness by the global maximum
EMP_data$relative_rich <- EMP_data$Richness / max(EMP_data$Richness)

scatter.smooth(EMP_data$Temp, EMP_data$relative_rich,
               xlab = "Temperature (°C)", ylab = "Relative Richness")

# =============================================================================
# Save filtered dataset
# =============================================================================
write.csv(EMP_data, "../../data/EMP_filtered.csv", row.names = TRUE)

# =============================================================================
# 99th percentile of relative richness per temperature step
# Used as an empirical richness envelope for comparison with simulations
# =============================================================================

# global 99th percentile
quantile(EMP_data$relative_rich, probs = 0.99)

# per-temperature 99th percentile
quan = c()
for(i in unique(EMP_data$Temp)){
  sub = subset(EMP_data, Temp == i)
  quan = c(quan, quantile(sub$relative_rich, probs = 0.99))
}
