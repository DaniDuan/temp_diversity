include("./Base/sim_frame.jl")

### Model params
N=100
M=50
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

# Retrieve the environment variable as a string
index_str = ENV["SLURM_ARRAY_TASK_ID"]
# Convert the string to a numeric value (e.g., Integer)
index = parse(Int, index_str)

# T = 0 + 273.15
# p = generate_params(N, M; f_u = modular_uptake, f_l = modular_leakage, f_m=F_m, f_ρ=F_ρ, f_ω=F_ω, N_modules = round(Int, M / 3), s_ratio = 100.0, L = L, T = T, ρ_t = ρ_t, Tr = Tr, Ed = Ed, input_type = input_type[1])
# prob = ODEProblem(dx!, x0, tspan, p)
# sol =solve(prob, AutoVern7(Rodas5()), save_everystep = true, callback=cb)
# bm = sol.u[length(sol.t)][1:N]
# sur = (1:N)[bm .> 1.0e-7]
# invasion_growth_rate(5, p, x0, tspan, cb)

all_rich = Float64[]; all_Shannon =  Float64[]; all_Simpson = Float64[]; 
ϵ_sur = Vector{Vector{Float64}}(); ϵ_ext = Vector{Vector{Float64}}(); all_ϵ = Vector{Vector{Float64}}(); 
u_sur =  Vector{Vector{Float64}}(); u_ext =  Vector{Vector{Float64}}(); m_sur =  Vector{Vector{Float64}}(); m_ext =  Vector{Vector{Float64}}();
all_r_inv = Vector{Vector{Float64}}(); r_inv_sur =  Vector{Vector{Float64}}(); r_inv_ext = Vector{Vector{Float64}}();
all_Eu = Vector{Vector{Float64}}(); all_Em =  Vector{Vector{Float64}}(); all_Eu_sur = Vector{Vector{Float64}}(); all_Em_sur = Vector{Vector{Float64}}(); 
all_Tpu = Vector{Vector{Float64}}(); all_Tpm =  Vector{Vector{Float64}}(); all_Tpu_sur =  Vector{Vector{Float64}}(); all_Tpm_sur =  Vector{Vector{Float64}}();
@time for i in range(0, stop = 30, length = 31)
    T = 273.15 + i 
    p = generate_params(N, M; f_u = modular_uptake, f_l = modular_leakage, f_m=F_m, f_ρ=F_ρ, f_ω=F_ω, N_modules = round(Int, M / 3), s_ratio = 100.0, L = L, T = T, ρ_t = ρ_t, Tr = Tr, Ed = Ed, input_type = input_type[1])
    ## Calc CUE
    ϵ = (p.u * x0[N+1:N+M] .* (1 .- p.L) .- p.m) ./ (p.u * x0[N+1:N+M])
    ## run simulation
    prob = ODEProblem(dx!, x0, tspan, p)
    sol =solve(prob, AutoVern7(Rodas5()), save_everystep = false, callback=cb)
    bm = sol.u[length(sol.t)][1:N]
    sur = (1:N)[bm .> 1.0e-7]
    ext = (1:N)[bm .< 1.0e-7]
    ## Shannon & Simpson
    p_v = bm[sur]./sum(bm[sur])
    Shannon = - sum(p_v .* log.(p_v))
    Simpson = 1/ sum(p_v .^2)
    ## collecting E and Tp
    Eu = p.E[:,1]; Em = p.E[:,2]
    Eu_sur = Eu[sur]; Em_sur = Em[sur]
    Tpu = p.Tp[:,1]; Tpm = p.Tp[:,2]
    Tpu_sur = Tpu[sur]; Tpm_sur = Tpm[sur]
    ###
    p_lv = LV_params(p=p, sol=sol);
    r_inv = zeros(N)
    for i in 1:N
        r_inv[i] = invasion_growth_rate(i, p, x0, tspan, cb)
    end

    push!(all_rich, length(sur)); push!(all_Shannon, Shannon); push!(all_Simpson, Simpson);
    push!(ϵ_sur, ϵ[sur]); push!(ϵ_ext, ϵ[ext]); push!(all_ϵ, ϵ); 
    push!(all_r_inv, r_inv); push!(r_inv_sur, r_inv[sur]); push!(r_inv_ext, r_inv[ext]);
    push!(all_Eu, Eu); push!(all_Em, Em); push!(all_Eu_sur, Eu_sur); push!(all_Em_sur, Em_sur); #Eu Em
    push!(all_Tpu, Tpu); push!(all_Tpm, Tpm); push!(all_Tpu_sur, Tpu_sur); push!(all_Tpm_sur, Tpm_sur); #Tp
    push!(u_sur, sum(p.u, dims = 2)[sur]); push!(u_ext, sum(p.u, dims = 2)[ext]);
    push!(m_sur, p.m[sur]); push!(m_ext, p.m[ext]);
end 

@save "../data/20260324/temp_rich_0/iters0_$(index).jld2" all_rich all_Shannon all_Simpson ϵ_sur ϵ_ext all_ϵ all_r_inv r_inv_sur r_inv_ext all_Eu all_Em all_Eu_sur all_Em_sur all_Tpu all_Tpm all_Tpu_sur all_Tpm_sur u_sur u_ext m_sur m_ext