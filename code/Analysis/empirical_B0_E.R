# =============================================================================
# empirical_B0_E.R
#
# Extracts and summarises empirical thermal performance curve (TPC) parameters
# (B0, E) for microbial growth and respiration rates from two published
# datasets (Smith et al. 2019, 2021). Computes median carbon use efficiency
# (CUE) at the reference temperature (10°C) and derives the uptake
# normalization constant B0_u from the CUE constraint.
# Also plots individual TPCs and B0/E distributions.
#
# Inputs:  data/aerobic_tpc_data.csv   — Smith et al. 2021 (33 strains)
#          data/summary.csv            — Smith et al. 2019 meta-analysis
# Outputs: results/TPC_empirical.pdf
#          results/B_Ea_empirical.pdf
#          results/density_growth_B0.pdf
#          results/density_growth_E.pdf
#
# Location: code/Analysis/empirical_B0_E.R
# Data:     data/ (parallel to code/)
# =============================================================================

graphics.off()
library(minpack.lm)
library(ggplot2)
library(reshape2)
library(grid)
library(dplyr)   # needed for bind_rows

# Boltzmann constant (eV/K)
k <- 8.617e-5

# leakage rate assumed for CUE calculation
L <- 0.3

# =============================================================================
# Modified Sharpe-Schoolfield equation
# Evaluates trait value at a given temperature (K) given TPC parameters.
# Reference temperature Tr = 283.15 K (10°C)
# =============================================================================
Schoolfield <- function(Temp, B0, T_pk, Ea, E_D) {
  B0 * exp(-Ea * (1/Temp - 1/283.15) / k) /
    (1 + (Ea / (E_D - Ea)) * exp(E_D * (1/T_pk - 1/Temp) / k))
}

# =============================================================================
# Carbon use efficiency
# CUE = G / (G + R), where G = growth rate, R = respiration rate
# =============================================================================
CUE <- function(G, R) G / (G + R)

# =============================================================================
# Dataset 1: smith2020systematic — Experimental growth and res data, Tr = 0 (33 strains)
# Filter to strains with sufficient temperature points before peak
# =============================================================================
data <- read.csv("../../data/aerobic_tpc_data.csv")
data <- data[-which(data$temps_before_peak_resp  <= 3 | data$temps_before_peak_growth <= 3), ]

# separate growth and respiration parameter tables
res_data <- data.frame(data$E_resp, data$B0_resp, data$E_D_resp, data$Tpk_resp, data$Ppk_resp, data$r_sq_resp)
names(res_data) <- c('Ea', 'B0', 'E_D', 'T_pk', 'P_pk','r_sq')
grow_data <- data.frame(data$E_growth, data$B0_growth, data$E_D_growth, data$Tpk_growth, data$Ppk_growth, data$r_sq_growth)
names(grow_data) <- c('Ea', 'B0', 'E_D', 'T_pk', 'P_pk','r_sq')

# --- Summary statistics: growth ---
mean_Br <- mean(grow_data$B0)
mean_Er <- mean(grow_data$Ea)
var_Br <- var(log(grow_data$B0))/abs(mean(log(grow_data$B0))) # standardizing with mean value
var_Er <- var(grow_data$Ea)/abs(mean(grow_data$Ea)) # standardizing with mean value
rho_r <- cor(log(grow_data$B0), grow_data$Ea) # B0-E correlation coefficient

# --- Summary statistics: respiration ---
mean_Bm <- mean(res_data$B0)
mean_Em <- mean(res_data$Ea) 
median(res_data$Ea)
var_Bm <- var(log(res_data$B0))/abs(mean(log(res_data$B0))) # standardizing with mean value
var_Em <- var(res_data$Ea)/abs(mean(res_data$Ea)) # standardizing with mean value
rho_m <- cor(log(res_data$B0), res_data$Ea) # B0-E correlation coefficient

# --- Maintenance normalization constant at Tr = 10°C ---
Bm <- Schoolfield(283.15, res_data$B0, res_data$T_pk, res_data$Ea, res_data$E_D)
hist(Bm) 
mean(log(Bm))

# --- Median CUE at 10°C across all strains ---
CUE10 <- numeric(nrow(res_data))
for (i in seq_len(nrow(res_data))) {
  G <- Schoolfield(283.15, grow_data$B0[i], grow_data$T_pk[i], grow_data$Ea[i],  grow_data$E_D[i])
  R <- Schoolfield(283.15, res_data$B0[i],  res_data$T_pk[i], res_data$Ea[i],   res_data$E_D[i])
  CUE10[i] <- CUE(G, R)
}
BCUE <- median(CUE10)
# BCUE <- mean(CUE10)

# --- Uptake normalization constant from CUE constraint ---
# B0_u = B0_m / (1 - L - CUE)
Bu <- Bm / (1 - L - BCUE)
mean(log(Bu))

# =============================================================================
# Dataset 2: smith2019community — meta-analysis summary (growth & respiration)
# =============================================================================
summ <- read.csv("../../data/summary.csv")

s_data <- summ[summ$source == "growth", ] # growth rate TPC parameters
m_data <- summ[summ$source == "res", ] # respiration rate TPC parameters

# --- Summary statistics: growth (meta-analysis) ---
smean_Br <- mean(log(s_data$B0_ss))
median(log(s_data$B0_ss))
smean_Er <- mean(s_data$E)
svar_Br  <- var(log(s_data$B0_ss)) / mean(log(s_data$B0_ss))
svar_Er  <- var(s_data$E)          / mean(s_data$E)
srho_r   <- cor(log(s_data$B0_ss), s_data$E)

median(s_data$Tpk) # median peak temperatures

# --- Summary statistics: respiration (meta-analysis) ---
mean(m_data$E)
svar_Bm  <- var(log(m_data$B0_ss)) / mean(log(m_data$B0_ss))
svar_Em  <- var(m_data$E)          / mean(m_data$E)
srho_m   <- cor(log(m_data$B0_ss), m_data$E)

# uptake normalization from meta-analysis maintenance B0
Bu_meta <- mean(m_data$B0_ss) / (1 - L - BCUE)

# =============================================================================
# Plot: individual TPCs for growth (meta-analysis strains)
# =============================================================================
temps <- 273.15 + seq(0, 30, by = 1)
plot_data <- data.frame()
for (i in seq_len(nrow(s_data))) {
  temp_p  <- log(Schoolfield(
    Temp = temps,
    B0   = s_data$B0_ss[i],
    T_pk = 273.15 + s_data$Tpk[i],
    Ea   = s_data$E[i],
    E_D  = s_data$Ed[i]
  ))
  temp_df   <- data.frame(Temperature = temps,
                          GrowthRate  = temp_p,
                          Species     = as.factor(i))
  plot_data <- bind_rows(plot_data, temp_df)
}

p_tpc <- ggplot(plot_data, aes(x = Temperature, y = GrowthRate, color = Species)) +
  geom_line(alpha = 0.4, linewidth = 1.2) +
  labs(x = "Temperature (°C)", y = "Growth rate (log)", color = "Species") +
  scale_x_continuous(
    breaks = seq(273.15, 303.15, by = 5),
    labels = function(x) sprintf("%.0f", x - 273.15)
  ) +
  scale_y_continuous(limits = c(-20, -5)) +
  theme_bw() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position  = "none",
    text             = element_text(size = 50)
  )

ggsave("../../results/TPC_empirical.pdf", plot = p_tpc, width = 16, height = 12)

# =============================================================================
# Plot: B0 vs E scatter (growth, meta-analysis)
# =============================================================================
p_BEa <- ggplot(s_data, aes(x = log(B0_ss), y = E)) +
  geom_point(size = 4, shape = 16, color = "black", alpha = 0.7) +
  labs(title = "Empirical growth rate",
       x     = expression("log(B"[0]*")"),
       y     = expression("E"[r])) +
  theme_bw() +
  theme(panel.background = element_blank(),
        text = element_text(size = 50))

ggsave("../../results/B_Ea_empirical.pdf", plot = p_BEa, width = 16, height = 12)

# =============================================================================
# Plot: density of B0 (growth, meta-analysis)
# =============================================================================
p_B0 <- ggplot(s_data, aes(x = B0_ss)) +
  geom_density(fill = "#E5ACA5", alpha = 0.5) +
  labs(tag = "(a)", title = "Empirical growth rate",
       x = expression(B[0]), y = "Density") +
  theme_bw() +
  theme(panel.background = element_blank(),
        text = element_text(size = 50))

ggsave("../../results/density_growth_B0.pdf", plot = p_B0, width = 16, height = 12)

# =============================================================================
# Plot: density of E (growth, meta-analysis)
# =============================================================================
p_E <- ggplot(s_data, aes(x = E)) +
  geom_density(fill = "#B7CAB7", alpha = 0.5) +
  labs(tag = "(b)", title = "Empirical growth rate",
       x = "E (eV)", y = "Density") +
  theme_bw() +
  theme(panel.background = element_blank(),
        text = element_text(size = 50))

ggsave("../../results/density_growth_E.pdf", plot = p_E, width = 16, height = 12)