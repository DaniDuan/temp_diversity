# The Role of Metabolic Strategies in Determining Microbial Community Diversity along Temperature Gradients

## Description

This repository contains the **simulation code**, **empirical parameterisation scripts**, **data**, and **results** for the study.

## Installation

```bash
git clone https://github.com/DaniDuan/temp_diversity.git
```

## Languages

- [x] Julia (simulation and analytical code)
- [x] R (empirical parameterisation, data analysis)
- [x] Bash (HPC cluster submission)

## Repository Structure

```
temp_diversity/
├── code/
│   ├── Base/               # Core model source files (Julia)
│   ├── Analysis/           # Empirical data (R) analysis and simulation output (Julia) analysis
│   └── Cluster/            # HPC job submission scripts
├── data/                   # Empirical datasets (CSV)
├── results/                # Simulation outputs
└── sandbox/                # Tests and scratch notes
```

## Dependencies

### Julia packages

```julia
using Pkg
Pkg.add(["Distributions", "Random", "LinearAlgebra", "DifferentialEquations", "Sundials", "Parameters", "CSV", "DataFrames", "CairoMakie", "LsqFit", "Logging", "JLD2"])
```

### R packages

```r
install.packages(c("ggplot2", "reshape2", "dplyr", "tidyverse", "minpack.lm", "rTPC", "nls.multstart", "broom", "progress"))
```

## Main Project Structure and Usage

### `code/Base/` --- Core MiCRM simulation files (Julia)

- **`sim_frame.jl`**: Top-level framework file. Loads all Julia dependencies and includes all source files below. Start here for any simulation run.

- **`micrm_params.jl`**: Parameter generation for the MiCRM. Provides default and modular sampling functions for all model parameters: uptake matrix $\mathbf{U}$, leakage tensor $\mathbf{L}$, maintenance costs $m$, resource supply $\rho$, and dilution $\omega$. The top-level `generate_params()` assembles these into a single NamedTuple. Modular sampling (`modular_uptake`, `modular_leakage`) groups consumers and resources into metabolic modules with within-module specialisation controlled by `N_modules` and `s_ratio`.

- **`micrm_dx.jl`**: ODE system for the MiCRM. Tracks $N$ consumer biomasses and $M$ resource abundances. Uses a modular structure: `growth_MiCRM!`, `supply_MiCRM!`, and `depletion_MiCRM!` can be swapped independently. Supports four resource supply regimes via `input_type`: `"constant"`, `"leaching"`, `"chemostat"`, and `"self-renewing"` (temperature-dependent resource growth).

- **`temp.jl`**: Temperature-dependent trait functions implementing the modified Sharpe-Schoolfield equation for uptake and maintenance rates. `temp_trait()` returns temperature dependent trait values for $N$ species. `randtemp_param()` samples TPC parameters ($B_0$, $E$, $T_{pk}$) from empirically constrained multivariate normal distributions following Smith et al. (2019, 2021), encoding the generalist-specialist trade-off via the $B_0$-$E$ covariance $\rho_t$.

- **`invasion_r.jl`**: Computes the invasion growth rate `r_inv` for each species. For species $i$, the resident community (all other species) is simulated to equilibrium, and species $i$'s per-capita growth rate is evaluated against the resulting resource environment. Positive `r_inv` means the species can invade from rare.

- **`LV_params.jl`**: Derives effective Lotka-Volterra (eLV) parameters from a MiCRM equilibrium via quasi-steady-state approximation of resource dynamics (MacArthur 1970). Returns the effective interaction matrix ($N \times N$) and intrinsic growth rates $r$ for the surviving community.

- **`LV_dx.jl`**: ODE system for the effective Lotka-Volterra model. Used to integrate consumer dynamics under the eLV approximation as an alternative to the full MiCRM.

- **`Jacobian.jl`**: Computes the community Jacobian matrix of the eLV system restricted to the feasible (surviving) submatrix at equilibrium. Eigenvalues are used to assess local asymptotic stability.

- **`analytical_functions.jl`**: Utility functions including cosine similarity, Bray-Curtis dissimilarity, and Euclidean distance for comparing species uptake profiles or community compositions; and AIC for nonlinear model selection.

---

### `code/Analysis/` --- Empirical data analysis (R)

- **`empirical_B0_E.R`**: Extracts and summarises empirical TPC parameters ($B_0$, $E$) for microbial growth and respiration from Smith et al. (2019, 2021). 
    - Input: `data/aerobic_tpc_data.csv`, `data/summary.csv`
    - Output: `results/TPC_empirical.pdf`, `results/B_Ea_empirical.pdf`, `results/density_growth_B0.pdf`, `results/density_growth_E.pdf`

- **`fitting_metaanalysis.R`**: Fits TPCs to microbial growth and respiration data from meta-analysis database using the Sharpe-Schoolfield nonlinear model. 
    - Input: `data/database.csv`
    - Output: `data/summary.csv`

- **`EMP_ana.R`**: Loads and filters the Earth Microbiome Project (EMP) dataset, normalises richness by the global maximum, and extracts the 99th percentile richness envelope per temperature step for comparison with model predictions.
    - Input: `data/EMP.csv`
    - Output: `data/EMP_filtered.csv`

## Running a Single Assembly (Local Test)

```julia
include("code/Base/sim_frame.jl")

p = generate_params(N = 50, M = 50;
    f_u        = modular_uptake,
    f_l        = modular_leakage,
    f_m        = F_m, f_ρ = F_ρ, f_ω = F_ω,
    N_modules  = round(Int, M / 3),
    s_ratio    = 100.0,
    L          = fill(0.3, N), 
    T          = 15.0 + 273.15,
    ρ_t        = [-0.35 -0.35],
    Tr         = 273.15 + 10,
    Ed         = 3.5,
    input_type = "constant",
    ω          = 0.0)

x0   = vcat(fill(0.1, N), fill(1, M))
prob = ODEProblem(dx!, x0, (0.0, 2.5e10), p)
sol  = solve(prob, AutoVern7(Rodas5()), save_everystep = true)

bm  = sol.u[end][1:N]
sur = (1:N)[bm .> 1e-7]
println("Richness: ", length(sur))
```

## References

- Smith, T. P., Thomas, T. J., Garc´ıa-Carreras, B., Sal, S., Yvon-Durocher, G., Bell, T. & Pawar, S. (2019), ‘Community-level respiration of prokaryotic microbes may rise with global warming’, *Nature communications* 10(1), 1–11.
- Smith, T. P., Clegg, T., Bell, T. & Pawar, S. (2021), ‘Systematic variation in the temperature dependence of bacterial carbon use efficiency’, *Ecology Letters* 24(10), 2123–2133.
- Thompson, L. R., Sanders, J. G., McDonald, D., Amir, A., Ladau, J., Locey, K. J., Prill, R. J., Tripathi, A., Gibbons, S. M., Ackermann, G. et al. (2017), ‘A communal catalogue reveals earth’s multiscale microbial diversity’, *Nature* 551(7681), 457–463.
- MacArthur, R. (1970), ‘Species packing and competitive equilibrium for many species’, *Theoretical population biology* 1(1), 1–11.
- Marsland III, R., Cui, W., Goldford, J., Sanchez, A., Korolev, K. & Mehta, P. (2019), ‘Available energy fluxes drive a transition in the diversity, stability, and functional structure of microbial communities’, *PLoS computational biology* 15(2), e1006793.

## Author and Contact

**Quqiming (Danica) Duan** --- Imperial College London, Silwood Park

Email: d.duan20@imperial.ac.uk

**Samraat Pawar** --- Imperial College London, Silwood Park

Email: s.pawar@imperial.ac.uk
