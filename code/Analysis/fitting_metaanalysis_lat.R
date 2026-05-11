# =============================================================================
# fitting_metaanalysis_lat.R
#
# Fits thermal performance curves (TPCs) to microbial growth and respiration
# rate data from a meta-analysis database using two methods:
#   1. Linear (Arrhenius) model on log-transformed sub-peak data
#   2. Sharpe-Schoolfield (1981) nonlinear model via nls_multstart
#
# Only strains with statistically significant fits (all p < 0.05) are retained.
# Results are standardised (B0 z-scored within source group) and saved.
#
# Input:  data/database.csv
# Output: data/summary.csv
#         data/summary_lat.csv   <- includes latitude + region label
#
# Location: code/Analysis/fitting_metaanalysis_lat.R
# Data:     data/ (parallel to code/)
# =============================================================================
setwd("/Users/Danica/Documents/temp_diversity/code")

library(rTPC)
library(nls.multstart)
library(broom)
library(tidyverse)
library(progress)
library(ggpubr)
library(ggforce)

# =============================================================================
# Load and prepare data
# =============================================================================

# FIX: read the file once, then subset — eliminates the full_join entirely
# and makes Latitude.x / Latitude.y impossible
df_raw <- read.csv("../data/database.csv")

df_meta <- df_raw %>%
  filter(StandardisedTraitName == "Specific Growth Rate") %>%
  select(
    Strain      = OriginalID,
    growth_rate = StandardisedTraitValue,
    Temperature = ConTemp,
    Latitude    = Latitude
  ) %>%
  mutate(source = "growth")

df_res <- df_raw %>%
  filter(StandardisedTraitName != "Specific Growth Rate") %>%
  select(
    Strain      = OriginalID,
    growth_rate = StandardisedTraitValue,
    Temperature = ConTemp,
    Latitude    = Latitude
  ) %>%
  mutate(source = "res")

# FIX: bind_rows() instead of full_join() — the two frames are disjoint
# subsets of the same file so stacking is all that is needed; no join keys,
# no suffix columns
df_full <- bind_rows(df_meta, df_res) %>%
  filter(growth_rate > 0.0)

# quick diagnostic plot
df_full %>%
  ggplot(aes(Temperature, growth_rate)) +
  geom_point() +
  facet_wrap(~source, scales = "free") +
  labs(x = "Temperature (°C)", y = "Rate")

# =============================================================================
# Nest data by strain and source, initialise fit columns
# =============================================================================

# Compute one representative latitude per Strain x source before nesting
# so Latitude is a plain scalar column on df_nested, not buried in the list-col
df_lat <- df_full %>%
  group_by(Strain, source) %>%
  summarise(Latitude = median(Latitude, na.rm = TRUE), .groups = "drop")

# FIX: nest only the columns that vary per measurement; Latitude is excluded
# from the nest and re-attached cleanly via left_join — no suffix risk because
# Latitude no longer exists in df_full at nest time
df_nested <- df_full %>%
  select(-Latitude) %>%                                  # FIX: drop before nesting
  nest(data = c(growth_rate, Temperature)) %>%
  left_join(df_lat, by = c("Strain", "source")) %>%      # re-attach scalar latitude
  mutate(
    E_lm  = 0.0,
    B0_lm = 0.0,
    E_ss  = 0.0,
    B0_ss = 0.0,
    Tpk   = 0.0,
    Ed    = 0.0
  )

# =============================================================================
# Fit loop: linear Arrhenius + Sharpe-Schoolfield per strain
# =============================================================================
pb <- progress::progress_bar$new(
  total  = nrow(df_nested),
  clear  = FALSE,
  format = "[:bar] :percent :elapsedfull"
)

for (i in seq_len(nrow(df_nested))) {
  pb$tick()
  
  x <- df_nested$data[[i]] %>% filter(growth_rate > 0)
  
  # --- Method 1: linear Arrhenius on sub-peak data ---
  i_pk  <- which.max(x$growth_rate)
  x_lin <- x %>%
    filter(Temperature < x$Temperature[i_pk]) %>%
    mutate(T = (1 / 8.617e-5) * ((1 / (Temperature + 273.15)) - (1 / 283.15)))
  
  if (nrow(x_lin) > 3 && length(unique(x_lin$T)) > 3) {
    lm_mod <- lm(log(growth_rate) ~ T, data = x_lin)
    
    if (all(tidy(lm_mod)$p.value < 0.05)) {
      df_nested$B0_lm[i] <- coef(lm_mod)[1]
      df_nested$E_lm[i]  <- coef(lm_mod)[2]
    }
  }
  
  # --- Method 2: Sharpe-Schoolfield nonlinear fit ---
  start_vals <- get_start_vals(x$Temperature, x$growth_rate,
                               model_name = "sharpeschoolhigh_1981")
  
  start_vals[is.na(start_vals)]       <- 0.0
  start_vals[is.infinite(start_vals)] <- 0.0
  
  ss_mod <- tryCatch(
    nls_multstart(
      growth_rate ~ sharpeschoolhigh_1981(
        temp = Temperature, r_tref, e, eh, th, tref = 10
      ),
      data        = x,
      iter        = c(4, 4, 4, 4),
      start_lower = start_vals - 10,
      start_upper = start_vals + 10,
      lower       = get_lower_lims(x$Temperature, x$growth_rate,
                                   model_name = "sharpeschoolhigh_1981"),
      upper       = get_upper_lims(x$Temperature, x$growth_rate,
                                   model_name = "sharpeschoolhigh_1981"),
      supp_errors = "Y"
    ),
    error = function(e) NULL
  )
  
  if (!is.null(ss_mod)) {
    tidy_ss <- tryCatch(tidy(ss_mod), error = function(e) NULL)
    if (!is.null(tidy_ss) &&
        !any(is.nan(tidy_ss$p.value)) &&
        all(tidy_ss$p.value < 0.05)) {
      df_nested$B0_ss[i] <- coef(ss_mod)[1]
      df_nested$E_ss[i]  <- coef(ss_mod)[2]
      df_nested$Ed[i]    <- coef(ss_mod)[3]
      df_nested$Tpk[i]   <- coef(ss_mod)[4]
    }
  }
}

# =============================================================================
# Post-processing: filter, log-transform, and standardise B0 within source
# =============================================================================

df_final <- df_nested %>%
  filter(B0_ss != 0.0, E_ss != 0.0, log(B0_ss) > -15) %>%
  mutate(B0 = log(B0_ss), E = E_ss) %>%
  group_by(source) %>%
  mutate(B0 = (B0 - mean(B0)) / sd(B0))

# =============================================================================
# Diagnostic plots
# =============================================================================

df_final %>%
  ggplot(aes(B0, E, color = source)) +
  geom_point() +
  labs(x = "Standardised log(B0)", y = "E (eV)", color = "Source")

df_final %>%
  pivot_longer(c(B0, E)) %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 30) +
  facet_wrap(name ~ source, scales = "free") +
  labs(x = "Value", y = "Count")

# =============================================================================
# Build summary_lat.csv — add latitude and climate region
# =============================================================================

# Classify absolute latitude into three climate zones:
#   0–30°  -> "Tropical"
#  30–60°  -> "Temperate"
#  60–90°  -> "Polar"
# Strains with missing latitude are labelled NA so they are not silently
# dropped but are still easy to filter out downstream.
df_final_lat <- df_final %>%
  mutate(
    abs_lat = abs(Latitude),               # FIX: plain Latitude, no .x/.y suffix
    Region  = case_when(
      is.na(abs_lat)  ~ NA_character_,
      abs_lat <= 30   ~ "Tropical",
      abs_lat <= 60   ~ "Temperate",
      abs_lat <= 90   ~ "Polar",
      TRUE            ~ NA_character_      # safeguard for out-of-range values
    )
  ) %>%
  select(-abs_lat)                         # drop helper column before saving

# Diagnostic plot — E and B0 distributions broken down by climate region
df_final_lat %>%
  filter(!is.na(Region)) %>%
  pivot_longer(c(B0, E)) %>%
  ggplot(aes(value, fill = Region)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  facet_wrap(name ~ source, scales = "free") +
  scale_fill_manual(values = c(
    "Tropical"  = "#E69F00",
    "Temperate" = "#56B4E9",
    "Polar"     = "#009E73"
  )) +
  labs(x = "Value", y = "Count", fill = "Region",
       title = "B0 and E distributions by climate region and source")

# Scatter plot — E along latitude by source and region
df_final_lat %>%
  filter(!is.na(Latitude), !is.na(Region)) %>%
  ggplot(aes(abs(Latitude), E, color = Region)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "black", linewidth = 0.7) +
  facet_wrap(~source, scales = "free_y") +
  scale_color_manual(values = c(
    "Tropical"  = "#E69F00",
    "Temperate" = "#56B4E9",
    "Polar"     = "#009E73"
  )) +
  labs(x = "Latitude (°)", y = "E (eV)", color = "Region",
       title = "Activation energy (E) along latitude by source")


# compute rho per region first
rho_labels <- df_final_lat %>%
  filter(!is.na(Region), Region %in% c("Tropical", "Temperate"),
         source == "growth") %>%
  group_by(Region) %>%
  summarise(
    rho   = cor(B0, E, method = "pearson", use = "complete.obs"),
    label = paste0("rho == ", round(rho, 2)),   # FIX: plotmath syntax
    .groups = "drop"
  )

cov_plot <- df_final_lat %>%
  filter(!is.na(Region), Region %in% c("Tropical", "Temperate"),
         source == "growth") %>%
  ggplot(aes(B0, E, color = Region, fill = Region)) +
  geom_point(alpha = 0.8, size = 3) +
  stat_ellipse(
    geom      = "polygon",
    level     = 0.95,
    alpha     = 0.15,
    linewidth = 0.7
  ) +
  geom_text(
    data        = rho_labels,
    aes(label   = label),
    x           = Inf, y = Inf,
    hjust       = 1.1, vjust = 1.5,
    color       = "black",
    parse       = TRUE, 
    size        = 5,   
    inherit.aes = FALSE
  ) +
  facet_wrap(~Region, scales = "free") +
  scale_color_manual(values = c(
    "Tropical"  = "#B73508",
    "Temperate" = "#F5D44B"
  )) +
  scale_fill_manual(values = c(
    "Tropical"  = "#B73508",
    "Temperate" = "#F5D44B"
  )) +
  labs(x     = "log(B0)", y = "E",
       color = "Region", fill = "Region")+
  theme(
    legend.position = "none",          # removes legend entirely
    text      = element_text(size = 20),   # changes ALL text (labels, legend, strip)
    axis.text = element_text(size = 15)    # changes axis tick labels specifically
  )
# title    = "B0 vs E covariance by region (growth rate only)",
# subtitle = "Ellipse = 95% covariance region, ρ = Pearson correlation coefficient")

# print to screen
print(cov_plot)

# save to results/
ggsave(
  filename = "../results/covariance_B0_E_by_region.pdf",
  plot     = cov_plot,
  width    = 10,
  height   = 6,
  units    = "in"
)
# df_final_lat %>%
#   filter(!is.na(Region), Region %in% c("Tropical", "Temperate"),
#          source == "growth") %>%
#   group_by(Region) %>%
#   summarise(
#     mean_Tpk = mean(Tpk, na.rm = TRUE),
#     var_Tpk  = var(Tpk,  na.rm = TRUE),
#     sd_Tpk   = sd(Tpk,   na.rm = TRUE),   # included for reference
#     n        = n(),
#     .groups  = "drop"
#   )
df_final_lat %>%
  filter(!is.na(Latitude)) %>%
  group_by(Region) %>%
  summarise(n_species = n_distinct(Strain))

# Save latitude-enriched output
write_csv(df_final_lat, "../data/summary_lat.csv")
