# =============================================================================
# Jacobian.jl
#
# Computes the community Jacobian matrix of the effective Lotka-Volterra
# system, restricted to the feasible (surviving) submatrix at equilibrium.
# Used to assess local asymptotic stability of the assembled community.
#
# Requires p_lv from LV_params.jl and sol from solving dx! in micrm_dx.jl.
# =============================================================================

"""
    Eff_Lv_Jac(; p_lv, sol)

Computes the community Jacobian matrix of the effective Lotka-Volterra system, restricted to the feasible (surviving) species at equilibrium.

A species is considered a survivor if its final biomass exceeds 1e-7.

The Jacobian element for species pair (i, j) in the survivor submatrix is:

    J_ij = ℵ_ij * C_i*

where ℵ_ij is the effective interaction coefficient from `p_lv` and C_i* is the equilibrium biomass of species i. The diagonal elements J_ii = ℵ_ii * C_i* capture effective self-regulation.

The eigenvalues of this matrix can be used to assess local asymptotic stability of the feasible equilibrium: stability requires all eigenvalues to have negative real parts.

# Arguments
- `p_lv`: NamedTuple of effective LV parameters from `Eff_LV_params`,
          must contain fields `N`, `ℵ`, and `r`.
- `sol`:  ODE solution from `solve`, used to extract equilibrium biomasses.

# Returns
- `LV_Jac`: `(N_sur × N_sur)` matrix where `N_sur` is the number of survivors.
"""

function Eff_Lv_Jac(; p_lv, sol)

    bm = sol.u[length(sol.t)][1:p_lv.N]
    sur = (1:p_lv.N)[bm .> 1.0e-7]
    N = length(sur)
    ℵ = [p_lv.ℵ[sur[i],sur[j]] for i in 1:N, j in 1:N]
    r = p_lv.r[sur]
    C = bm[sur]
    LV_Jac = [ℵ[i, j]*C[i] for i in 1:N, j in 1:N]

    return(LV_Jac)
end
