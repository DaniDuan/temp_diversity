# =============================================================================
# LV_params.jl
#
# Computes effective Lotka-Volterra (eLV) parameters from a MiCRM equilibrium solution via a multivariate Taylor expansion of consumer dynamics around the quasi-steady state of resource abundances (MacArthur 1970, Marsland et al. 2020, Eqns. 5-7 in the paper).
#
# The eLV approximation assumes resource dynamics equilibrate faster than consumer dynamics, allowing resources to be analytically eliminated and pairwise consumer interactions to be expressed as effective coefficients.
#
# Called after solving the full MiCRM (dx! in micrm_dx.jl) to obtain eLV parameters for stability analysis (Eff_Lv_Jac in Jacobian.jl) or for integrating the reduced eLV system (LV_dx! in LV_dx.jl).
# =============================================================================

"""
    LV_params(; p, sol)

Computes effective Lotka-Volterra parameters from the MiCRM equilibrium solution using a quasi-steady-state approximation of resource dynamics.

The eLV system takes the form (Eqn. 5):

    dCᵢ/dt ≈ Cᵢ (rᵢ + ∑ⱼ ℵᵢⱼ Cⱼ)

The effective interaction coefficients are (Eqn. 6):

    ℵᵢⱼ = ∑_α uᵢα (1 - lᵢα) ∂R̂α/∂Cⱼ

where the resource sensitivity matrix ∂R̂/∂C is obtained by inverting the resource Jacobian A (the matrix of partial derivatives of resource dynamics at equilibrium):

    Aαβ = -ωα + ∑ᵢ Cᵢ* (lᵢαβ uᵢβ - uᵢβ δαβ)

    ∂R̂α/∂Cⱼ = ∑_{β,γ} A⁻¹αβ uⱼβ Rγ* (δβγ - lⱼβγ)

The effective intrinsic growth rate is (Eqn. 7):

    rᵢ = ∑_α uᵢα (1 - lᵢα) Rα* - ∑ⱼ ℵᵢⱼ Cⱼ* - mᵢ

# Arguments
- `p`  : MiCRM parameter NamedTuple from `generate_params`, requires fields `N`, `M`, `u`, `m`, `l`, `λ`, `ρ`, `ω`
- `sol`: ODE solution from `solve`, used to extract equilibrium consumer biomasses `Ceq` (indices 1:N) and resource abundances `Req` (indices N+1:N+M)

# Returns
A NamedTuple with:
- `ℵ` : `(N × N)` effective interaction matrix; negative = competition, positive = facilitation/cross-feeding
- `r` : length-`N` vector of effective intrinsic growth rates
- `N` : number of consumers (passed through for downstream use)

# Notes
- At a true equilibrium, survivors should satisfy rᵢ + ∑ⱼ ℵᵢⱼ Cⱼ* ≈ 0.
- The accuracy of the eLV approximation degrades in strongly coupled systems; see Mustri et al. (2025) for conditions under which it holds.

# Example
    sol   = solve(ODEProblem(dx!, x0, tspan, p), AutoVern7(Rodas5()))
    p_lv  = LV_params(p=p, sol=sol)
    # p_lv.ℵ  — interaction matrix
    # p_lv.r  — intrinsic growth rates
"""
function LV_params(; p, sol)

    M, N, l, ρ, ω, m, u, λ = p.M, p.N, p.l, p.ρ, p.ω, p.m, p.u, p.λ

    # Kronecker delta: δ(x,y) = 1 if x == y, else 0
    δ(x, y) = ==(x, y)

    # equilibrium consumer biomasses and resource abundances
    Ceq = sol[1:N, end]
    Req = sol[(N+1):(N+M), end]

    # -------------------------------------------------------------------------
    # Resource Jacobian A (M × M)
    # Aαβ = -ωα + ∑ᵢ Cᵢ* (lᵢαβ uᵢβ - uᵢβ δαβ)
    # Encodes how resource α responds to a perturbation in resource β, accounting for dilution (ω), leakage into α, and uptake from α.
    # -------------------------------------------------------------------------
    A = [(-ω[α] + sum(l[i, α, β] * u[i, β] * Ceq[i] - u[i, β] * Ceq[i] * δ(α, β) for i in 1:N)) for α in 1:M, β in 1:M]

    invA = inv(A)
    A_thing = u .* (1 .- λ)   # net uptake per species per resource: uᵢα (1 - lᵢα), (N × M)

    # -------------------------------------------------------------------------
    # Resource sensitivity matrix ∂R̂/∂C (M × N)
    # ∂R̂α/∂Cⱼ = ∑_{β,γ} A⁻¹αβ uⱼβ Rγ* (δβγ - lⱼβγ)
    # Quantifies how equilibrium resource α shifts when consumer j increases.
    # Negative for shared resources (competition), positive for cross-fed metabolites (facilitation).
    # -------------------------------------------------------------------------
    ∂R = [sum(invA[α, β] * u[j, β] * Req[γ] * (δ(β, γ) - l[j, β, γ]) for β in 1:M, γ in 1:M) for α in 1:M, j in 1:N]

    # -------------------------------------------------------------------------
    # Effective interaction matrix ℵ (N × N)
    # ℵᵢⱼ = ∑_α uᵢα (1 - lᵢα) ∂R̂α/∂Cⱼ
    # -------------------------------------------------------------------------
    ℵ = [sum(A_thing[i, α] * ∂R[α, j] for α in 1:M) for i in 1:N, j in 1:N]

    # -------------------------------------------------------------------------
    # Effective intrinsic growth rates r (length N)
    # rᵢ = ∑_α uᵢα (1 - lᵢα) Rα* - ∑ⱼ ℵᵢⱼ Cⱼ* - mᵢ
    # O: gross resource assimilation at equilibrium resources
    # P: interaction contribution from all neighbours at equilibrium
    # -------------------------------------------------------------------------
    O = [sum(A_thing[i, α] * Req[α] for α in 1:M) for i in 1:N]
    P = [dot(ℵ[i, :], Ceq) for i in 1:N]
    r = O .- P .- m

    return (ℵ = ℵ, r = r, N = N)
end