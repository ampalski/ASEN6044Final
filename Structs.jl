mutable struct Ephemeris
    time::Vector{Float64}
    ephem::Matrix{Float64}
    ObjectID::Int64
    PD::Float64
    Q::Matrix{Float64}
end

mutable struct Measurement
    ObsID::Int64
    ObsPos::Vector{Float64}
    Time::Float64
    TrueMeas::Vector{Float64}
    ReportedMeas::Vector{Float64}
    R::Matrix{Float64}
    TrueTgtID::Int64
    AssocVec::Vector{Float64}
end

mutable struct Estimate
    ObjectID::Int64
    Mean::Vector{Float64}
    Cov::Matrix{Float64}
    CurrentTime::Float64
    MeasurementApplied::Bool
    Q::Matrix{Float64}
    MeasWithoutUpdate::Int64
end

mutable struct Particle
    weight::Float64
    estimates::Vector{Estimate}
    assocHistory::Vector{Vector{Int64}}
end