# =============================================================================
# temp.jl
#
# Temperature-dependent trait functions for the MiCRM, implementing the modified Sharpe-Schoolfield equation for uptake and maintenance rates.
# =============================================================================

"""
    temp_trait(N, kw)

Returns temperature-scaled trait values for `N` species at temperature `T`
using the modified Sharpe-Schoolfield equation:

    trait(T) = B * exp(-E/k * (1/T - 1/Tr)) / (1 + E/(Ed-E) * exp(Ed/k * (1/Tp - 1/T)))

Returns a matrix `temp_p` of size `(N × 2)` where column 1 is the temperature-scaled uptake rate and column 2 is the temperature-scaled maintenance rate for each species. 
Also returns the sampled trait parameters `B`, `E`, `Tp` from `randtemp_param`.

# Required keys in `kw`
- `T`  : current temperature (K)
- `Tr` : reference temperature (K)
- `Ed` : deactivation energy (eV)
"""
function temp_trait(N, kw) 
    k = 0.0000862 # Boltzman constant
    @unpack T, Tr, Ed= kw
    B,E,Tp = randtemp_param(N, kw)
    temp_p = B .* exp.((-E./k) * ((1/T)-(1/Tr)))./(1 .+ (E./(Ed .- E)) .* exp.(Ed/k * (1 ./Tp .- 1/T)))
    # Eϵ = (B[:,2] .* (E[:,1] .- E[:,2]))./(B[:,1]*(1-L)-B[:,2])  #m0(Eu − Em)/(u0(1 − l) − m0)
    return temp_p, B, E, Tp
end

"""
    randtemp_param(N, kw)

Samples species-level thermal performance curve (TPC) parameters for `N` species from empirically constrained multivariate normal distributions, following Smith et al. (2019, 2021).

Parameters are sampled jointly for uptake (u) and maintenance (m):
- `B0` (normalization constants) and `E` (activation energies) are drawn from a bivariate normal distribution with covariance determined by `ρ_t`, encoding the generalist-specialist trade-off.
- `Tp` (peak temperatures) are drawn from a normal distribution, with maintenance peaking 3°C above uptake for each species.

Normalization constants are parameterized to produce a median carbon use efficiency (CUE) of 0.1953 at the reference temperature `Tr`, constraining uptake to exceed maintenance.

# Returns
- `B`  : `(N × 2)` matrix of normalization constants [uptake, maintenance]
- `E`  : `(N × 2)` matrix of activation energies (eV) [uptake, maintenance]
- `Tp` : `(N × 2)` matrix of peak temperatures (K) [uptake, maintenance]

# Required keys in `kw`
- `L`   : per-species leakage vector (length N), used to compute median CUE
- `T`   : current temperature (K)
- `ρ_t` : B0-E correlation coefficient encoding generalist-specialist trade-off
- `Tr`  : reference temperature (K)
- `Ed`  : deactivation energy (eV)
"""
function randtemp_param(N, kw)
    @unpack L, T, ρ_t, Tr, Ed = kw
    L_v = mean(L)
    B0_m = -1.4954; # log normalization for maintenance at Tr
    B0_CUE = 0.1953 # median CUE at Tr (Smith et al. 2021)
    B0_u = log(exp(B0_m) / (1 - L_v - B0_CUE)) # uptake normalization from CUE constraint
    
    B0 = [B0_u B0_m] 
    B0_var = 0.17 .* abs.(B0); # variance in log(B0)
    E_mean = [0.8146 0.5741]; # mean activation energies: Eu > Em (eV)
    E_var =  0.1364 .* E_mean # variance in E 

    # covariance between log(B0) and E — encodes generalist-specialist trade-off
    cov_xy = ρ_t .* B0_var.^0.5 .* E_var .^ 0.5
    meanv = [B0 ; E_mean]
    cov_u = [B0_var[1] cov_xy[1]; cov_xy[1] E_var[1]]
    cov_m = [B0_var[2] cov_xy[2]; cov_xy[2] E_var[2]]

    # sample B0 and E jointly for uptake and maintenance
    allu = rand(MvNormal(meanv[:,1], cov_u), N)
    allm = rand(MvNormal(meanv[:,2], cov_m), N)
    B = [exp.(allu[1,:]) exp.(allm[1,:])]
    E = [allu[2,:] allm[2,:]]

    # peak temperatures: maintenance peaks 3°C above uptake
    Tpu = 273.15 .+ rand(Normal(35, 5), N)
    Tpm = Tpu .+ 3
    Tp = [Tpu Tpm]
    return B,E,Tp
end 
