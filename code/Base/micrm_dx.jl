# =============================================================================
# micrm_dx.jl
#
# Defines the ODE system for the Microbial Consumer-Resource Model (MiCRM).
# The model tracks N consumer biomasses and M resource abundances (Eqns. 1-2 in the paper):
#
#   dCᵢ/dt = Cᵢ (∑_α uᵢα Rα - ∑_αβ uᵢα lᵢαβ Rα - mᵢ)
#   dRα/dt = ρα(R) - ∑ᵢ Cᵢ uᵢα Rα + ∑ᵢ∑_β Cᵢ uᵢβ Rβ lᵢβα
#
# The ODE is split into three modular sub-functions (growth!, supply!, depletion!) that can be swapped independently to test alternative biological assumptions without modifying the core integration loop.
#
# State vector layout: x = [C₁, ..., Cₙ, R₁, ..., Rₘ]
# Consumer indices:    1:N
# Resource indices:    N+1:N+M
# =============================================================================

"""
    growth_MiCRM!(dx, x, p, t, i)

Updates the derivative of the `i`-th consumer biomass:

    dCᵢ/dt = Cᵢ (∑_α uᵢα Rα - ∑_αβ uᵢα lᵢαβ Rα - mᵢ)

The first term is gross resource uptake, the second is leakage loss (fraction of uptake returned to the resource pool), and the third is maintenance respiration.

Requires `p.u` (N×M uptake matrix), `p.l` (N×M×M leakage tensor), `p.m` (length-N maintenance vector), `p.N`, `p.M`.
"""
function growth_MiCRM!(dx, x, p, t, i)
    # maintenance cost
    dx[i] = -x[i] * p.m[i]
    # gross uptake minus leakage loss across all resources
    for α = 1:p.M
        dx[i] += x[α + p.N] * x[i] * p.u[i, α]          # gross uptake of resource α
        for β = 1:p.M
            dx[i] += -x[i] * x[α + p.N] * p.u[i, α] * p.l[i, α, β]  # leakage from α to β
        end
    end
end

"""
    supply_MiCRM!(dx, x, p, t, α)

Updates the abiotic supply term of the `α`-th resource according to `p.input_type`. Four supply regimes are supported:

- `"constant"`: fixed inflow, no outflow.
        dRα/dt += ρα

- `"leaching"`: fixed inflow with first-order dilution at rate ωα.
        dRα/dt += ρα - ωα Rα

- `"chemostat"`: chemostat dilution toward carrying capacity Kc[α].
        dRα/dt += ωα (Kc[α] - Rα)

- `"self-renewing"`: logistic resource growth with temperature-dependent intrinsic growth rate r_self and carrying capacity Kc_self, both scaled by the Sharpe-Schoolfield equation using `p.T`, `p.Tr`, `p.Ed`, `p.Kc` (peak temperature fixed at 25°C).
        dRα/dt += r_self[α] * Rα / Kc_self[α] * (Kc_self[α] - Rα)

Required fields per input type:
- `"constant"`, `"leaching"`: `p.ρ`
- `"leaching"`: also `p.ω`
- `"chemostat"`: `p.ω`, `p.Kc`
- `"self-renewing"`: `p.T`, `p.Tr`, `p.Ed`, `p.Kc`
"""
function supply_MiCRM!(dx, x, p, t, α)
    if p.input_type == "constant"
        dx[α + p.N] = p.ρ[α]

    elseif p.input_type == "leaching"
        dx[α + p.N] = p.ρ[α] - (x[α + p.N] * p.ω[α])

    elseif p.input_type == "chemostat"
        dx[α + p.N] = p.ω[α] * (p.Kc[α] - x[α + p.N])

    elseif p.input_type == "self-renewing"
        k          = 0.0000862        # Boltzmann constant (eV/K)
        E_r_self   = 0.8              # activation energy for resource growth (eV)
        Tpk_r_self = 273.15 + 25.0   # peak temperature for resource growth (K), fixed at 25°C
        # temperature-scaled intrinsic growth rate via Sharpe-Schoolfield
        r_self     = fill(1, p.M) * exp((-E_r_self / k) * ((1 / p.T) - (1 / p.Tr))) / (1 + (E_r_self / (p.Ed - E_r_self)) * exp(p.Ed / k * (1 / Tpk_r_self - 1 / p.T)))
        # temperature-scaled carrying capacity
        Kc_self    = p.Kc * exp((0.8 / k) * ((1 / p.T) - (1 / p.Tr)))
        # logistic growth
        dx[α + p.N] = (r_self[α] * x[α + p.N]) / Kc_self[α] * (Kc_self[α] - x[α + p.N])
    end
end

"""
    depletion_MiCRM!(dx, x, p, t, i, α)

Updates the derivative of the `α`-th resource due to consumption by consumer `i` and cross-feeding leakage back into α from all resources β:

    dRα/dt += -uᵢα Rα Cᵢ + ∑_β uᵢβ lᵢβα Rβ Cᵢ

The first term is resource depletion through direct uptake. The second term is leakage input into α from species `i` consuming resource β and leaking a fraction lᵢβα back as resource α (cross-feeding/byproduct).

Requires `p.u`, `p.l`, `p.M`, `p.N`.
"""
function depletion_MiCRM!(dx, x, p, t, i, α)
    # direct uptake of resource α by consumer i
    dx[α + p.N] += -(x[α + p.N] * p.u[i, α] * x[i])
    # leakage from all resources β back into resource α (cross-feeding)
    for β = 1:p.M
        dx[α + p.N] += x[β + p.N] * x[i] * p.u[i, β] * p.l[i, β, α]
    end
end

"""
    dx!(dx, x, p, t;
        growth!    = growth_MiCRM!,
        supply!    = supply_MiCRM!,
        depletion! = depletion_MiCRM!)

Top-level derivative function for the MiCRM, compatible with the DifferentialEquations.jl solver interface. Loops over all consumers and resources, delegating to the three modular sub-functions.

The modular design allows each biological process to be swapped independently — e.g. replacing `supply_MiCRM!` with a custom chemostat function — without modifying the integration loop.

Requires `p.input_type` for `supply_MiCRM!`. See each sub-function's docstring for full field requirements.

# Example
    prob = ODEProblem(dx!, x0, tspan, p)
    sol  = solve(prob, AutoVern7(Rodas5()), callback=cb)
"""
function dx!(dx, x, p, t;
    growth!::Function    = growth_MiCRM!,
    supply!::Function    = supply_MiCRM!,
    depletion!::Function = depletion_MiCRM!)

    # consumer dynamics
    for i = 1:p.N
        dx[i] = 0.0
        growth!(dx, x, p, t, i)
    end

    # resource dynamics
    for α = 1:p.M
        dx[p.N + α] = 0.0
        supply!(dx, x, p, t, α)       # abiotic supply term
        for i = 1:p.N
            depletion!(dx, x, p, t, i, α)  # biotic depletion and cross-feeding
        end
    end
end