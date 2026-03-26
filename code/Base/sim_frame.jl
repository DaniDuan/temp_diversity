# =========================================================================================================================
# sim_frame.jl
#
# Main simulation framework for the temperature-dependent MiCRM.
# Sets up all dependencies by including the required source files.
#
# Required files (must be in the same directory):
#   micrm_params.jl         — parameter generation: generate_params, modular_uptake, modular_leakage, def_*
#   micrm_dx.jl             — ODE system: dx!, growth_MiCRM!, supply_MiCRM!, depletion_MiCRM!
#   LV_dx.jl                — effective LV ODE system: LV_dx!, growth_LV!
#   LV_params.jl            — effective LV parameter calculation: Eff_LV_params
#   temp.jl                 — temperature trait functions: temp_trait, randtemp_param
#   invasion_r.jl           — invasion growth rate calculation
#   analytical_functions.jl — community similarity and model selection metrics (analytical functions that might be useful)
#   Jacobian.jl             — community Jacobian for survivor submatrix
#
# Typical usage example:
#   cd("WORKING_DIRECTORY")
#   include("sim_frame.jl")
#   p = generate_params(N, M; f_u=modular_uptake, f_l=modular_leakage,
#           f_m=F_m, f_ρ=F_ρ, f_ω=F_ω, N_modules=round(Int, M/3),
#           s_ratio=100.0, L=L, T=T, ρ_t=ρ_t, Tr=Tr, Ed=Ed,
#           input_type="leaching")
# =========================================================================================================================

# Load libraries
using Distributions, Random
using LinearAlgebra
using DifferentialEquations
# using Plots, StatsPlots
using Sundials
using Parameters
using CSV, DataFrames
using CairoMakie
using LsqFit 
using Logging
using JLD2
using StatsBase

# Include simulation code files
include("./micrm_params.jl")
include("./micrm_dx.jl")
include("./LV_dx.jl")
include("./LV_params.jl")
include("./temp.jl")
include("./invasion_r.jl")
include("./analytical_functions.jl")
include("Jacobian.jl")