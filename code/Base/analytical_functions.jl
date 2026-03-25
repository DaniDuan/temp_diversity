"""
    cosine_similarity(vec1, vec2)

Returns the cosine similarity between two vectors, measuring the angle between them regardless of magnitude. 

A value of 1 indicates identical directions (maximum overlap). 

0 indicates orthogonality (no overlap), and -1 indicates opposite directions.

Useful for comparing species' resource uptake profiles in niche space.
"""
function cosine_similarity(vec1, vec2)
    dot_product = dot(vec1, vec2)
    norm1 = norm(vec1)
    norm2 = norm(vec2)
    return dot_product / (norm1 * norm2)
end

"""
    bray_curtis_dissimilarity(A, B)

Returns the Bray-Curtis dissimilarity between two non-negative vectors `A` and `B`, ranging from 0 (identical) to 1 (completely different).

Commonly used to compare species abundance or biomass compositions between two communities.
"""
function bray_curtis_dissimilarity(A, B)
    return sum(abs.(A .- B)) / sum(A .+ B)
end

"""
    euclidean_distance(A, B)

Returns the Euclidean distance between vectors `A` and `B` in trait or abundance space.
"""
function euclidean_distance(A, B)
    return sqrt(sum((A .- B) .^ 2))
end

"""
    AIC(model, n)

Returns the Akaike Information Criterion (AIC) for a fitted model with `n` observations, using residual sum of squares (RSS) and the number of parameters `k`:

    AIC = n * log(RSS / n) + 2k

Lower values indicate a better trade-off between model fit and complexity.

Requires a fitted model object with `residuals` and `coef` methods, e.g. from `LsqFit.jl`.
"""
function AIC(model, n)
    RSS = sum(residuals(model).^2)
    k = length(coef(model))
    aic_value = n * log(RSS / n) + 2 * k
    return aic_value
end

