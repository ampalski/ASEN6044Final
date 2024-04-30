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

temp = load("src/Meas.jld2")
allMeasStationary = temp["allMeasStationary"]
allMeasManeuver = temp["allMeasManeuver"]

temp = load("src/MeasNoisy.jld2")
allMeasStationaryNoisy = temp["allMeasStationaryNoisy"]
allMeasManeuverNoisy = temp["allMeasManeuverNoisy"]
