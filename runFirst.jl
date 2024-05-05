using LinearAlgebra
using DifferentialEquations
using GLMakie
using JLD2
using FileIO
using Distributions
using StatsBase
using Random

include("Structs.jl")
include("Utils.jl")
include("Dynamics.jl")
include("Measurements.jl")
include("Munkres.jl")
include("Unscented.jl")
include("Plotting.jl")
include("GNN.jl")
include("JPDA.jl")
include("RBMCDA.jl")

temp = load("src/Ephem.jld2")
astra1nStationary = temp["astra1nStationary"]
astra1krStationary = temp["astra1krStationary"]
astra1mStationary = temp["astra1mStationary"]
astra1lStationary = temp["astra1lStationary"]
astra1nManeuver = temp["astra1nManeuver"]
astra1krManeuver = temp["astra1krManeuver"]
astra1mManeuver = temp["astra1mManeuver"]
astra1lManeuver = temp["astra1lManeuver"]

temp = load("src/EphemShort.jld2")
astra1nStationary2 = temp["astra1nStationary2"]
astra1krStationary2 = temp["astra1krStationary2"]
astra1mStationary2 = temp["astra1mStationary2"]
astra1lStationary2 = temp["astra1lStationary2"]

temp = load("src/Meas.jld2")
allMeasStationary = temp["allMeasStationary"]
allMeasManeuver = temp["allMeasManeuver"]

# temp = load("src/MeasNoisy.jld2")
# allMeasStationaryNoisy = temp["allMeasStationaryNoisy"]
# allMeasManeuverNoisy = temp["allMeasManeuverNoisy"]

temp = load("src/MeasLambda.jld2")
allMeasStationary5 = temp["allMeasStationary5"]
allMeasStationary10 = temp["allMeasStationary10"]
allMeasStationary25 = temp["allMeasStationary25"]
allMeasStationary50 = temp["allMeasStationary50"]
allMeasStationary100 = temp["allMeasStationary100"]