"""
    growth_MiCRM!(dx,x,p,t,i)

default growth function for the MiCRM.
"""
function growth_MiCRM!(dx, x, p, t, i)
    # maintenance
    dx[i] = -x[i]*p.m[i]
    # resoruce uptake and leakage
    for α = 1:p.M
        dx[i] += x[α + p.N] * x[i] * p.u[i, α]
        for β=1:p.M
            dx[i] += -x[i]*x[α + p.N]*p.u[i, α]*p.l[i, α, β]
        end
    end
end

"""
    supply_MiCRM!(dx,x,p,t,α)

default supply function for the MiCRM.
"""
function supply_MiCRM!(dx, x, p, t, α)
    if p.input_type == "constant"
        dx[α + p.N] = p.ρ[α]
 
    elseif p.input_type == "leaching"
        dx[α + p.N] = p.ρ[α] - (x[α + p.N] * p.ω[α])
 
    elseif p.input_type == "chemostat"
        dx[α + p.N] = p.ω[α] * (p.Kc[α] - x[α + p.N])
 
    elseif p.input_type == "self-renewing"
        k          = 0.0000862       # Boltzmann constant (eV/K)
        E_r_self   = 0.8             # activation energy for resource growth
        Tpk_r_self = 273.15 + 25.0  # peak temperature (K)
        r_self     = fill(1, p.M) * exp((-E_r_self / k) * ((1 / p.T) - (1 / p.Tr))) /
                     (1 + (E_r_self / (p.Ed - E_r_self)) * exp(p.Ed / k * (1 / Tpk_r_self - 1 / p.T)))
        Kc_self    = p.Kc * exp((0.8 / k) * ((1 / p.T) - (1 / p.Tr)))
        dx[α + p.N] = (r_self[α] * x[α + p.N]) / Kc_self[α] * (Kc_self[α] - x[α + p.N])
    end
end

"""
    depletion_MiCRM!(dx,x,p,t,i,α)

default depletion function for the MiCRM.
"""
function depletion_MiCRM!(dx, x, p, t, i, α)
    # uptake
    dx[α + p.N] += -(x[α + p.N] * p.u[i, α] * x[i])
    # leakage
    for β = 1:p.M
        dx[α + p.N] += x[β + p.N] * x[i] * p.u[i, β] * p.l[i, β, α]
    end
end

"""
    dx!(dx,x,p,t; 
        growth!::Function = growth_MiCRM!, 
        supply!::Function = supply_MiCRM!,
        depletion!::Function = depletion_MiCRM!)

Derivative function for the general MICRM model. This function is used internally by the `DiffEq` solver which requires the first four arguments (`dx!(dx,x,p,t)`). 

"""
function dx!(dx, x, p, t;
    growth!::Function = growth_MiCRM!,
    supply!::Function = supply_MiCRM!,
    depletion!::Function = depletion_MiCRM!)

    for i = 1:p.N
        # reset derivatives
        dx[i] = 0.0
        growth!(dx, x, p, t, i)# update dx of ith consumer
    end
    
    for α = 1:p.M
        # reset derivatives
        dx[p.N + α] = 0.0
        #supply term
        supply!(dx, x, p, t, α)
        # loop over consumers
        for i = 1:p.N
            depletion!(dx, x, p, t, i, α)
        end
    end
end
