include("./Base/sim_frame.jl")

### Model params
N=7
M=5
L = fill(0.3, N);
input_type = ["constant", "leaching", "chemostat", "self-renewing"]

### Temp params 
num_temps = 31; # Running for 31 temp steps from 0 to 30 ᵒC 
ρ_t= [0.0000 0.0000]; # Minimal trade-off
# ρ_t= [-0.3500 -0.3500]; # Realistic trade-off
# ρ_t= [-0.9999 -0.9999]; # Maximal trade-off
Tr=273.15+10; # reference temperature
Ed=3.5; # high temp deactivation

# Generate MiCRM parameters
tspan = (0.0, 2.5e10)
x0 = vcat(fill(0.1, N), fill(1, M)) 
# here we define a callback that terminates integration as soon as system reaches steady state
condition(du, t, integrator) = norm(integrator(t, Val{1})) <= eps(); 
affect!(integrator) = terminate!(integrator)
cb = DiscreteCallback(condition, affect!)

p = generate_params(N, M; f_u = modular_uptake, f_l = modular_leakage, f_m=F_m, f_ρ=F_ρ, f_ω=F_ω, N_modules = round(Int, M / 3), s_ratio = 100.0, L = L, T = T, ρ_t = ρ_t, Tr = Tr, Ed = Ed, input_type = input_type[1])
prob = ODEProblem(dx!, x0, tspan, p)
sol =solve(prob, AutoVern7(Rodas5()), save_everystep = true, callback=cb)
bm = sol.u[length(sol.t)][1:N]
sur = (1:N)[bm .> 1.0e-7]

