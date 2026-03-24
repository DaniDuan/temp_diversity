"""
    invasion_growth_rate(i, p, sol, x0, tspan, cb)

Calculates the invasion growth rate of species `i` by:
1. Removing species `i` from the community
2. Running the resident community to equilibrium to get R*_{-i}
3. Evaluating species i's per-capita growth rate at R*_{-i} using Eq. 1

Returns the scalar invasion growth rate r_inv^i.
"""
function invasion_growth_rate(i, p, x0, tspan, cb)
    # indices of resident species (all except i)
    residents = setdiff(1:p.N, i)
    N_res = length(residents)
    M = p.M

    # build reduced parameter set for resident community
    p_res = (N = N_res, M = M, u = p.u[residents, :], m = p.m[residents],
        l = p.l[residents, :, :], λ = p.λ[residents, :],
        ρ = p.ρ, ω = p.ω, input_type = p.input_type,
        B = p.B[residents, :], E = p.E[residents, :], Tp = p.Tp[residents, :])

    # initial conditions
    x0_res = vcat(x0[residents], x0[p.N+1:p.N+p.M])# excluding species i from the original species pool

    # run resident community to equilibrium
    prob_res = ODEProblem(dx!, x0_res, tspan, p_res)
    sol_res = solve(prob_res, AutoVern7(Rodas5()), save_everystep=false, callback=cb)

    # extract equilibrium resource abundances R*_{-i}
    R_star = sol_res.u[end][N_res+1:N_res+M]

    # compute invasion growth rate from Eq. 1:
    # r_inv^i = sum_α u_iα (1 - sum_β l^i_αβ) R*_α - m_i
    η_i = 1.0 .- vec(sum(p.l[i, :, :], dims=2))  # net retention per resource, length M
    r_inv = dot(p.u[i, :] .* η_i, R_star) - p.m[i]

    return r_inv
end