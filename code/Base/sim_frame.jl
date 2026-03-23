# cd("WORKING_DIRECTORY")

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

# Include simulation code files
include("./micrm_params.jl")   # generate_params, modular_uptake, modular_leakage, def_*
include("./micrm_dx.jl")       # dx!, growth_MiCRM!, supply_MiCRM!, depletion_MiCRM!
include("./LV_dx.jl")          # LV_dx!, growth_LV!
include("./LV_params.jl")      # Eff_LV_params
include("./temp.jl")           # temp_trait, randtemp_param
include("./invasion_r.jl")     # invasion growth rate
# include("Jacobian.jl")

####################################################################################################################################################################################

function cosine_similarity(vec1, vec2)
    dot_product = dot(vec1, vec2)
    norm1 = norm(vec1)
    norm2 = norm(vec2)
    return dot_product / (norm1 * norm2)
end

function bray_curtis_dissimilarity(A, B)
    return sum(abs.(A .- B)) / sum(A .+ B)
end

function euclidean_distance(A, B)
    return sqrt(sum((A .- B) .^ 2))
end

function AIC(model, n)
    RSS = sum(residuals(model).^2)
    k = length(coef(model))
    aic_value = n * log(RSS / n) + 2 * k
    return aic_value
end

