# =============================================================================
# LV_dx.jl
#
# Defines the ODE system for the effective Lotka-Volterra (eLV) model, derived from the MiCRM via a quasi-steady-state approximation of resource dynamics (see LV_params.jl). 
# Used to integrate consumer dynamics under the eLV approximation as an alternative to the full MiCRM (dx! in micrm_dx.jl).
# =============================================================================

"""
    growth_LV!(dx, x, p, t, i)

Updates the derivative of the `i`-th species in the effective
Lotka-Volterra system:

    dCᵢ/dt = rᵢ Cᵢ + Cᵢ ∑ⱼ ℵᵢⱼ Cⱼ

The first term is density-independent growth at effective intrinsic rate `rᵢ`, which accounts for resource availability and pairwise interaction effects at low density (Eqn. 7 in the paper). 
The second term captures resource-mediated pairwise interactions: `ℵᵢⱼ < 0` is competition, `ℵᵢⱼ > 0` is facilitation/cross-feeding.

Both `r` and `ℵ` are computed from the underlying MiCRM parameters and equilibrium abundances by `Eff_LV_params` in LV_params.jl.

# Arguments
- `dx`: derivative vector, modified in-place
- `x` : current state vector of length `N` (consumer biomasses)
- `p` : parameter NamedTuple, requires fields `r` (length N), `ℵ` (N × N matrix), `N`
- `t` : current time (unused, required by DiffEq interface)
- `i` : index of the consumer species to update
"""
function growth_LV!(dx, x, p, t, i)

    dx[i] += p.r[i]*x[i]

    for j =1:p.N
        dx[i] += x[i]*p.ℵ[i, j]*x[j]
    end
end


"""
    LV_dx!(dx, x, p, t; growth! = growth_LV!)

Derivative function for the effective Lotka-Volterra system, compatible with the DifferentialEquations.jl solver interface. 
Loops over all `N` species and calls `growth!` for any species above numerical zero.

Species at or below `eps(x[i])` (floating-point machine epsilon relative to their current value) are skipped to prevent numerical drift of effectively extinct populations — this is a lighter-weight alternative to a hard threshold, since `eps` scales with the magnitude of `x[i]`.

The `growth!` function can be swapped to test alternative interaction formulations without modifying the core loop, consistent with the modular design of `dx!` in micrm_dx.jl.

# Arguments
- `dx`     : derivative vector, modified in-place
- `x`      : current state vector of length `N`
- `p`      : parameter NamedTuple, requires field `N`
- `t`      : current time (unused, required by DiffEq interface)
- `growth!`: growth function, default `growth_LV!`

# Example
    prob = ODEProblem(LV_dx!, x0, tspan, p_lv)
    sol  = solve(prob, AutoVern7(Rodas5()))
"""
function LV_dx!(dx, x, p, t;
    growth!::Function = growth_LV!)

    for i =1:p.N
        dx[i] = 0.0

        if x[i] > eps(x[i])
            growth!(dx, x, p, t, i)
        end
    end
end
