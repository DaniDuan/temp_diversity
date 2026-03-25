# =============================================================================
# invasion_r.jl
#
# Computes the invasion growth rate of each species in a MiCRM community.
# The invasion growth rate r_inv^i is the per-capita growth rate of species i
# when rare, evaluated against the equilibrium resource environment set by
# the resident community (all species except i).
#
# A positive r_inv^i means species i can invade from rare — it is not
# competitively excluded. At a true community equilibrium, survivors should
# have r_inv^i ≈ 0 and extinct species r_inv^i < 0.
#
# Used in sim_div.jl to diagnose coexistence and competitive exclusion
# across the temperature gradient.
# =============================================================================

"""
    invasion_growth_rate(i, p, x0, tspan, cb)

Computes the invasion growth rate of species `i` into the resident
community (all species except `i`) using the following steps:

1. Remove species `i` to form the resident community of size `N-1`
2. Simulate the resident community to equilibrium to obtain R*₋ᵢ
3. Evaluate species `i`'s per-capita growth rate at R*₋ᵢ (Eqn. 1):

        r_inv^i = ∑_α uᵢα (1 - ∑_β lᵢαβ) R*α₋ᵢ - mᵢ

A positive value indicates species `i` can grow when rare (can invade).
A negative value indicates competitive exclusion.
At a true equilibrium, survivors should satisfy r_inv^i ≈ 0.

# Arguments
- `i`     : index of the focal species (1-based)
- `p`     : MiCRM parameter NamedTuple from `generate_params`, requires
            fields `N`, `M`, `u`, `m`, `l`, `λ`, `ρ`, `ω`,
            `input_type`, `B`, `E`, `Tp`
- `x0`    : initial condition vector of length `N + M` used for the
            resident simulation (consumer biomasses then resource abundances)
- `tspan` : time span tuple `(t_start, t_end)` for the resident ODE
- `cb`    : DifferentialEquations.jl callback for early termination at
            steady state

# Returns
- scalar invasion growth rate `r_inv^i`

# Example
    r_inv = zeros(N)
    for i in 1:N
        r_inv[i] = invasion_growth_rate(i, p, x0, tspan, cb)
    end
"""
function invasion_growth_rate(i, p, x0, tspan, cb)

    # indices of all species except the focal species i
    residents = setdiff(1:p.N, i)
    N_res     = length(residents)
    M         = p.M

    # -------------------------------------------------------------------------
    # Build reduced parameter NamedTuple for the resident community
    # Species i is removed from all consumer-level arrays (u, m, l, λ, B, E, Tp)
    # Resource-level arrays (ρ, ω) are unchanged
    # -------------------------------------------------------------------------
    p_res = (
        N          = N_res,
        M          = M,
        u          = p.u[residents, :],
        m          = p.m[residents],
        l          = p.l[residents, :, :],
        λ          = p.λ[residents, :],
        ρ          = p.ρ,
        ω          = p.ω,
        input_type = p.input_type,
        B          = p.B[residents, :],
        E          = p.E[residents, :],
        Tp         = p.Tp[residents, :]
    )

    # initial conditions for resident simulation:
    # use original x0 biomasses for residents, original resource abundances
    x0_res = vcat(x0[residents], x0[p.N+1:p.N+M])

    # -------------------------------------------------------------------------
    # Simulate resident community to equilibrium
    # save_everystep=false keeps memory usage low since we only need the final state
    # -------------------------------------------------------------------------
    prob_res = ODEProblem(dx!, x0_res, tspan, p_res)
    sol_res  = solve(prob_res, AutoVern7(Rodas5()), save_everystep=false, callback=cb)

    # equilibrium resource abundances with species i absent
    R_star = sol_res.u[end][N_res+1:N_res+M]

    # -------------------------------------------------------------------------
    # Evaluate invasion growth rate at R*₋ᵢ (Eqn. 1)
    # η_i[α] = 1 - ∑_β lᵢαβ is the net retention fraction for species i on resource α (fraction of uptake converted to biomass rather than leaked)
    # -------------------------------------------------------------------------
    η_i   = 1.0 .- vec(sum(p.l[i, :, :], dims=2))
    r_inv = dot(p.u[i, :] .* η_i, R_star) - p.m[i]

    return r_inv
end