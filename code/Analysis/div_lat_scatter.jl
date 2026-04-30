#############################################################################################
# div_lat_scatter.jl
#############################################################################################

include("./Analysis/ana_atlas_function.jl")

# ── Load subset data ──────────────────────────────────────────────────────
temperate_df             = CSV.read("../data/top10_temperate.csv",             DataFrame)
tropical_df              = CSV.read("../data/top10_tropical.csv",              DataFrame)
temperate_df_soil        = CSV.read("../data/top10_temperate_soil.csv",        DataFrame)
tropical_df_soil         = CSV.read("../data/top10_tropical_soil.csv",         DataFrame)
temperate_df_aquatic     = CSV.read("../data/top10_temperate_aquatic.csv",     DataFrame)
tropical_df_aquatic      = CSV.read("../data/top10_tropical_aquatic.csv",      DataFrame)
temperate_df_not_aquatic = CSV.read("../data/top10_temperate_not_aquatic.csv", DataFrame)
tropical_df_not_aquatic  = CSV.read("../data/top10_tropical_not_aquatic.csv",  DataFrame)

# ── Run analyse_zone for each subset ─────────────────────────────────────
temp             = analyse_zone(temperate_df,             "Temperate — all")
trop             = analyse_zone(tropical_df,              "Tropical — all")
temp_soil        = analyse_zone(temperate_df_soil,        "Temperate — soil")
trop_soil        = analyse_zone(tropical_df_soil,         "Tropical — soil")
temp_aquatic     = analyse_zone(temperate_df_aquatic,     "Temperate — aquatic")
trop_aquatic     = analyse_zone(tropical_df_aquatic,      "Tropical — aquatic")
temp_not_aquatic = analyse_zone(temperate_df_not_aquatic, "Temperate — non-aquatic")
trop_not_aquatic = analyse_zone(tropical_df_not_aquatic,  "Tropical — non-aquatic")

# ── Generate four figures ─────────────────────────────────────────────────
fig_scatter_all         = make_scatter_fig(temperate_df,             tropical_df,
                                           temp,             trop,             "all")
fig_scatter_soil        = make_scatter_fig(temperate_df_soil,        tropical_df_soil,
                                           temp_soil,        trop_soil,        "soil")
fig_scatter_aquatic     = make_scatter_fig(temperate_df_aquatic,     tropical_df_aquatic,
                                           temp_aquatic,     trop_aquatic,     "aquatic")
fig_scatter_not_aquatic = make_scatter_fig(temperate_df_not_aquatic, tropical_df_not_aquatic,
                                           temp_not_aquatic, trop_not_aquatic, "non-aquatic")

# ── Save ──────────────────────────────────────────────────────────────────
save("../results/scatter_all.pdf",         fig_scatter_all)
save("../results/scatter_soil.pdf",        fig_scatter_soil)
save("../results/scatter_aquatic.pdf",     fig_scatter_aquatic)
save("../results/scatter_not_aquatic.pdf", fig_scatter_not_aquatic)