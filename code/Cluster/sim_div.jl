# =============================================================================
# sim_div.jl
#
# Temperature-diversity simulation script for cluster execution.
# Runs MiCRM community assembly across 31 temperature steps (0–30°C) for a single random seed, determined by SLURM_ARRAY_TASK_ID.
#
# Output: saves results to results/YYYYMMDD/temp_rich/iters_$(index).jld2
#
# Usage (local):
#   ENV["SLURM_ARRAY_TASK_ID"] = "1"
#   include("sim_div.jl")
#
# Usage (cluster):
#   sbatch sim_div.sh
# =============================================================================

include(joinpath(@__DIR__, "../Base/sim_frame.jl"))
# include("./Base/sim_frame.jl") # for use in Running example

# =============================================================================
# Model parameters
# =============================================================================
N=50
M=50
L = fill(0.3, N);
input_type = ["constant", "leaching", "chemostat", "self-renewing"]

# Temp params
num_temps = 31; # Running for 31 temp steps from 0 to 30 ᵒC 
# ρ_t= [0.0000 0.0000]; # Minimal trade-off
ρ_t= [-0.3500 -0.3500]; # Realistic trade-off
# ρ_t= [-0.9999 -0.9999]; # Maximal trade-off
Tr=273.15+10; # reference temperature
Ed=3.5; # high temp deactivation

# =============================================================================
# ODE solver setup
# =============================================================================
tspan = (0.0, 2.5e10)
x0 = vcat(fill(0.1, N), fill(1, M)) 
# callback: terminate integration when system reaches steady state (derivative norm falls below machine epsilon)
condition(du, t, integrator) = norm(integrator(t, Val{1})) <= eps(); 
affect!(integrator) = terminate!(integrator)
cb = DiscreteCallback(condition, affect!)

# =============================================================================
# Cluster array index — determines random seed for this replicate
# =============================================================================
index = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

# # =============================================================================
# # Running example
# # =============================================================================
# T = 15 + 273.15
# p = generate_params(N, M; f_u = modular_uptake, f_l = modular_leakage, f_m=F_m, f_ρ=F_ρ, f_ω=F_ω, N_modules = round(Int, M / 3), s_ratio = 100.0, L = L, T = T, ρ_t = ρ_t, Tr = Tr, Ed = Ed, input_type = input_type[1])
# prob = ODEProblem(dx!, x0, tspan, p)
# sol =solve(prob, AutoVern7(Rodas5()), save_everystep = true, callback=cb)
# bm = sol.u[length(sol.t)][1:N]
# sur = (1:N)[bm .> 1.0e-7]
# invasion_growth_rate(1, p, x0, tspan, cb)

# =============================================================================
# Pre-allocate output containers
# =============================================================================
all_rich = Float64[]; all_Shannon =  Float64[]; all_Simpson = Float64[]; 
all_sur = Vector{Vector{Float64}}(); all_ext = Vector{Vector{Float64}}();
all_u = Vector{Matrix{Float64}}(); all_m = Vector{Vector{Float64}}(); all_ϵ = Vector{Vector{Float64}}(); 
all_r_inv = Vector{Vector{Float64}}(); 
all_Eu = Vector{Vector{Float64}}(); all_Em =  Vector{Vector{Float64}}(); 
all_Tpu = Vector{Vector{Float64}}(); all_Tpm =  Vector{Vector{Float64}}(); 

# =============================================================================
# Main simulation loop: sweep temperature from 0 to 30°C
# =============================================================================
@time for i in range(0, stop = 30, length = 31)
    Random.seed!(index)     # same seed across temperatures for comparability
    T = 273.15 + i          # convert °C to Kelvin

    # generate MiCRM parameters at temperature T
    p = generate_params(N, M; f_u = modular_uptake, f_l = modular_leakage, f_m=F_m, f_ρ=F_ρ, f_ω=F_ω, N_modules = round(Int, M / 3), s_ratio = 100.0, L = L, T = T, ρ_t = ρ_t, Tr = Tr, Ed = Ed, input_type = input_type[1])

    # intrinsic carbon use efficiency per species
    ϵ = (p.u * x0[N+1:N+M] .* (1 .- p.L) .- p.m) ./ (p.u * x0[N+1:N+M])

    # solve ODE to equilibrium
    prob = ODEProblem(dx!, x0, tspan, p)
    sol =solve(prob, AutoVern7(Rodas5()), save_everystep = false, callback=cb)

    # extract equilibrium biomasses and classify survivors / extinct
    bm = sol.u[length(sol.t)][1:N]
    sur = (1:N)[bm .> 1.0e-7]
    ext = (1:N)[bm .< 1.0e-7]

    # diversity indices on surviving species
    p_v = bm[sur]./sum(bm[sur])
    Shannon = - sum(p_v .* log.(p_v))
    Simpson = 1/ sum(p_v .^2)

    # thermal trait values for all species
    Eu = p.E[:,1]; Em = p.E[:,2]
    Tpu = p.Tp[:,1]; Tpm = p.Tp[:,2]

    # effective LV parameters at equilibrium
    p_lv = LV_params(p=p, sol=sol);

    # invasion growth rates for all N species
    r_inv = zeros(N)
    for i in 1:N
        r_inv[i] = invasion_growth_rate(i, p, x0, tspan, cb)
    end

    # store results
    push!(all_rich, length(sur)); push!(all_Shannon, Shannon); push!(all_Simpson, Simpson);
    push!(all_sur, sur); push!(all_ext, ext);
    push!(all_r_inv, r_inv); push!(all_ϵ, ϵ); 
    push!(all_u, p.u); push!(all_m, p.m); 
    push!(all_Eu, Eu); push!(all_Em, Em); 
    push!(all_Tpu, Tpu); push!(all_Tpm, Tpm); 
    # push!(u_sur, sum(p.u, dims = 2)[sur]); push!(u_ext, sum(p.u, dims = 2)[ext]);    
end 

# =============================================================================
# Save results
# =============================================================================
results_dir = joinpath(@__DIR__, "../../results/20260325/temp_rich")
mkpath(results_dir)
@save joinpath(results_dir, "iters_$(index).jld2") all_rich all_Shannon all_Simpson all_sur all_ext all_r_inv all_ϵ all_u all_m all_Eu all_Em all_Tpu all_Tpm