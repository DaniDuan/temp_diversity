# =============================================================================
# fitting_metaanalysis.R
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
#
# Location: code/Analysis/fitting_metaanalysis.R
# Data:     data/ (parallel to code/)
# =============================================================================

library(rTPC)
library(nls.multstart)
library(broom)
library(tidyverse)
library(progress)

# =============================================================================
# Load and prepare data
# =============================================================================

# growth rate records
df_meta <- read.csv("../../data/database.csv") %>%
  filter(StandardisedTraitName == "Specific Growth Rate") %>%
  select(
    Strain      = OriginalID,
    growth_rate = StandardisedTraitValue,
    Temperature = ConTemp
  ) %>%
  mutate(source = "growth")

# respiration rate records (all non-growth traits)
df_res <- read.csv("../../data/database.csv") %>%
  filter(StandardisedTraitName != "Specific Growth Rate") %>%
  select(
    Strain      = OriginalID,
    growth_rate = StandardisedTraitValue,
    Temperature = ConTemp
  ) %>%
  mutate(source = "res")

# combine and remove non-positive rates (cannot log-transform)
df_full <- full_join(df_meta, df_res, by = c("Strain", "growth_rate", "Temperature", "source")) %>%
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
df_nested <- df_full %>%
  nest(data = c(growth_rate, Temperature)) %>%
  mutate(
    E_lm  = 0.0,   # Arrhenius activation energy (linear fit)
    B0_lm = 0.0,   # Arrhenius normalization constant (linear fit)
    E_ss  = 0.0,   # Sharpe-Schoolfield activation energy
    B0_ss = 0.0,   # Sharpe-Schoolfield normalization constant at Tref
    Tpk   = 0.0,   # peak temperature (°C)
    Ed    = 0.0    # high-temperature deactivation energy
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
  # use only data points below the peak temperature
  i_pk  <- which.max(x$growth_rate)
  x_lin <- x %>%
    filter(Temperature < x$Temperature[i_pk]) %>%
    mutate(T = (1 / 8.617e-5) * ((1 / (Temperature + 273.15)) - (1 / 283.15)))
  
  if (nrow(x_lin) > 3 && length(unique(x_lin$T)) > 3) {
    lm_mod <- lm(log(growth_rate) ~ T, data = x_lin)
    
    # only store if both intercept and slope are significant
    if (all(tidy(lm_mod)$p.value < 0.05)) {
      df_nested$B0_lm[i] <- coef(lm_mod)[1]
      df_nested$E_lm[i]  <- coef(lm_mod)[2]
    }
  }
  
  # --- Method 2: Sharpe-Schoolfield nonlinear fit ---
  start_vals <- get_start_vals(x$Temperature, x$growth_rate,
                               model_name = "sharpeschoolhigh_1981")
  
  # replace any NA/Inf starting values with 0 to allow fitting
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
  
  # only store if fit succeeded and all parameters are significant
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

# retain strains with successful Sharpe-Schoolfield fits and plausible B0
df_final <- df_nested %>%
  filter(B0_ss != 0.0, E_ss != 0.0, log(B0_ss) > -15) %>%
  mutate(B0 = log(B0_ss), E = E_ss) %>%
  group_by(source) %>%
  mutate(B0 = (B0 - mean(B0)) / sd(B0))   # z-score B0 within source group

# =============================================================================
# Diagnostic plots
# =============================================================================

# B0 vs E by source
df_final %>%
  ggplot(aes(B0, E, color = source)) +
  geom_point() +
  labs(x = "Standardised log(B0)", y = "E (eV)", color = "Source")

# distributions of B0 and E by source
df_final %>%
  pivot_longer(c(B0, E)) %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 30) +
  facet_wrap(name ~ source, scales = "free") +
  labs(x = "Value", y = "Count")

# =============================================================================
# Save output
# =============================================================================
write_csv(df_final, "../../data/summary.csv")