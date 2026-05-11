# ============================================================
# prep_latitude_data.R
#
# Description: Starting from summary_atlas.csv, this script:
#   1. Drops samples with missing latitude
#   2. Splits into tropical  (|lat| <= 30) and
#                  temperate (|lat| >  30) zones
#   3. Within each zone, keeps the top 10% richness samples
#      across ALL temperatures in that zone
#   4. Exports:
#../data/top10_tropical.csv
#../data/top10_temperate.csv
# ============================================================

setwd("/Users/Danica/Documents/temp_diversity/code")

# --- Libraries ----------------------------------------------
library(dplyr)

# --- Load full ATLAS data -----------------------------------
cat("Loading summary_atlas.csv...\n")
df <- read.csv("../data/summary_atlas.csv", check.names = FALSE)
cat("Total samples loaded:", nrow(df), "\n")

# --- Drop missing latitude ----------------------------------
df <- df[!is.na(df$Latitude), ]
cat("Samples after dropping missing latitude:", nrow(df), "\n")

# --- Assign latitude zone -----------------------------------
df$Zone <- case_when(
  abs(df$Latitude) <= 30              ~ "tropical",
  abs(df$Latitude) <= 60              ~ "temperate",
  abs(df$Latitude) >  60              ~ "polar"
)
cat("Zone breakdown:\n")
print(table(df$Zone))

# # --- Unique habitats in tropical zone with temperature < 20°C ---------------
# tropical_low_temp <- df[df$Zone == "tropical" & df$Temperature_C_mean < 20, ]
# cat("Samples in tropical zone with temp < 20°C:", nrow(tropical_low_temp), "\n")
# 
# unique_habitats <- unique(trimws(unlist(strsplit(tropical_low_temp$Env_cons, ";"))))
# cat("Unique habitats:\n")
# print(sort(unique_habitats))

# ============================================================
# Helper: filter to top 10% richness WITHIN each temp bin
# ============================================================
top10_per_bin <- function(zone_df, zone_name) {
  filtered <- zone_df %>%
    group_by(Temp_bin) %>%
    filter(Richness >= quantile(Richness, 0.90, na.rm = TRUE)) %>%
    ungroup()
  
  cat(sprintf("\n[%s] Samples in top 10%% per bin: %d / %d\n",
              zone_name, nrow(filtered), nrow(zone_df)))
  
  # Add bin counts (used as WLS weights in Julia)
  bin_counts <- filtered %>%
    group_by(Temp_bin) %>%
    summarise(n_bin = n(), .groups = "drop")
  
  filtered <- left_join(filtered, bin_counts, by = "Temp_bin")
  return(filtered)
}

# --- Apply filter per zone and habitat subset ---------------
tropical_df              <- top10_per_bin(df[df$Zone == "tropical",  ],  "tropical")
temperate_df             <- top10_per_bin(df[df$Zone == "temperate", ],  "temperate")

tropical_df_soil         <- top10_per_bin(df[df$Zone == "tropical"  & grepl("soil",    df$Env_cons), ], "tropical_soil")
tropical_df_aquatic      <- top10_per_bin(df[df$Zone == "tropical"  & grepl("aquatic", df$Env_cons), ], "tropical_aquatic")
tropical_df_not_aquatic  <- top10_per_bin(df[df$Zone == "tropical"  & !grepl("aquatic", df$Env_cons), ], "tropical_not_aquatic")

temperate_df_soil        <- top10_per_bin(df[df$Zone == "temperate" & grepl("soil",    df$Env_cons), ], "temperate_soil")
temperate_df_aquatic     <- top10_per_bin(df[df$Zone == "temperate" & grepl("aquatic", df$Env_cons), ], "temperate_aquatic")
temperate_df_not_aquatic <- top10_per_bin(df[df$Zone == "temperate" & !grepl("aquatic", df$Env_cons), ], "temperate_not_aquatic")

# --- Save ---------------------------------------------------
write.csv(tropical_df,              "../data/top10_tropical.csv",              row.names = FALSE)
write.csv(temperate_df,             "../data/top10_temperate.csv",             row.names = FALSE)
write.csv(tropical_df_soil,         "../data/top10_tropical_soil.csv",         row.names = FALSE)
write.csv(tropical_df_aquatic,      "../data/top10_tropical_aquatic.csv",      row.names = FALSE)
write.csv(tropical_df_not_aquatic,  "../data/top10_tropical_not_aquatic.csv",  row.names = FALSE)
write.csv(temperate_df_soil,        "../data/top10_temperate_soil.csv",        row.names = FALSE)
write.csv(temperate_df_aquatic,     "../data/top10_temperate_aquatic.csv",     row.names = FALSE)
write.csv(temperate_df_not_aquatic, "../data/top10_temperate_not_aquatic.csv", row.names = FALSE)
# cat("\nSaved:\n")
# cat("../data/top10_tropical.csv  —", nrow(tropical_df),  "samples\n")
# cat("../data/top10_temperate.csv —", nrow(temperate_df), "samples\n")
# cat("Columns:", paste(colnames(tropical_df), collapse = ", "), "\n")