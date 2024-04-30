function TrueAssociations(
    ephems::Vector{Ephemeris},
    measurements::Vector{Vector{Measurement}};
    Q::Matrix{Float64}=zeros(2, 2),
)

    nt = length(ephems[1].time) - 1
    # nt = 10

    residualHist = zeros(2, nt, length(ephems))
    errorHist = zeros(6, nt, length(ephems))
    stdHist = zeros(6, nt, length(ephems))

    ests = Vector{Estimate}()
    priorCov = diagm([100.0, 100.0, 100.0, 1e-6, 1e-6, 1e-6])

    for ephem in ephems
        push!(ests, getInitialEstimate(ephem, priorCov, Q=Q))
    end

    for i in 1:nt
        t = i + 1
        display(ephems[1].time[t])
        for j in eachindex(ests)
            ukfPredict!(ests[j], stepEstimate)
            # display(ests[j].Mean - ephems[j].ephem[:, t])

            curMeas = measurements[i]
            ind = 0
            for ind2 in eachindex(curMeas)
                if curMeas[ind2].TrueTgtID == ests[j].ObjectID
                    ind = ind2
                    break
                end
            end
            if ind > 0
                predMeas, S, K = ukfGain(ests[j], curMeas[ind], getMeas)
                newState, newCov, residual = applyUpdate(curMeas[ind], predMeas, [true, false], 2, ests[j], S, K)
                ests[j].Mean = newState
                ests[j].Cov = newCov
                ests[j].MeasurementApplied = true

                residualHist[:, i, j] = residual
            end
            # display(ests[j].Mean - ephems[j].ephem[:, t])
            errorHist[:, i, j] = ests[j].Mean - ephems[j].ephem[:, t]
            stdHist[:, i, j] = sqrt.(diag(ests[j].Cov))
        end
    end
    return residualHist, errorHist, stdHist
end

# stationaryEphem = [astra1nStationary, astra1krStationary, astra1mStationary, astra1lStationary]
# maneuverEphem = [astra1nManeuver, astra1krManeuver, astra1mManeuver, astra1lManeuver]
# baseTime = time()
# residualHistSta, errorHistSta, stdHistSta = TrueAssociations(stationaryEphem, allMeasStationary)
# time1 = time()-baseTime
# baseTime = time()
# residualHistStaNoisy, errorHistStaNoisy, stdHistStaNoisy = TrueAssociations(stationaryEphem, allMeasStationaryNoisy)
# time2 = time() - baseTime
# plotErrHist(stationaryEphem[1].time, errorHistStaNoisy)

# Q = copy(astra1nManeuver.Q)
# Q[4:6, 4:6] .*= 50 #True Assoc Maneuver case recovers in 1/2 day
# baseTime = time()
# residualHistMnv, errorHistMnv, stdHistMnv = TrueAssociations(maneuverEphem, allMeasManeuver; Q=Q)
# time3 = time() - baseTime
# baseTime = time()
# residualHistMnvNoisy, errorHistMnvNoisy, stdHistMnvNoisy = TrueAssociations(maneuverEphem, allMeasManeuverNoisy; Q=Q)
# time4 = time() - baseTime
# plotErrHist(stationaryEphem[1].time, errorHistMnvNoisy)

# jldsave("src/TrueAssocResults.jld2"; residualHistSta, errorHistSta, stdHistSta, residualHistMnv, errorHistMnv, stdHistMnv,residualHistStaNoisy, errorHistStaNoisy, stdHistStaNoisy, residualHistMnvNoisy, errorHistMnvNoisy, stdHistMnvNoisy)


function GNNAssociations!(
    ephems::Vector{Ephemeris},
    measurements::Vector{Vector{Measurement}};
    Q::Matrix{Float64}=zeros(2, 2),
    PG::Float64=0.95,
)

    nt = length(ephems[1].time) - 1
    # nt = 10

    residualHist = zeros(2, nt, length(ephems))
    errorHist = zeros(6, nt, length(ephems))
    stdHist = zeros(6, nt, length(ephems))

    ests = Vector{Estimate}()
    priorCov = diagm([100.0, 100.0, 100.0, 1e-6, 1e-6, 1e-6])

    for ephem in ephems
        push!(ests, getInitialEstimate(ephem, priorCov, Q=Q))
    end

    for i in 1:nt
        t = i + 1
        display(ephems[1].time[t])
        for j in eachindex(ests)
            ukfPredict!(ests[j], stepEstimate)
            # display(ests[j].Mean - ephems[j].ephem[:, t])
        end
        residuals = Vector{Vector{Float64}}()

        GNNCorrect!(ests, measurements[i], [true, false], residuals; PG=PG)

        for j in eachindex(ests)
            residualHist[:, i, j] = residuals[j]
            # display(ests[j].Mean - ephems[j].ephem[:, t])
            errorHist[:, i, j] = ests[j].Mean - ephems[j].ephem[:, t]
            stdHist[:, i, j] = sqrt.(diag(ests[j].Cov))
        end
    end
    return residualHist, errorHist, stdHist
end

stationaryEphem = [astra1nStationary, astra1krStationary, astra1mStationary, astra1lStationary]
maneuverEphem = [astra1nManeuver, astra1krManeuver, astra1mManeuver, astra1lManeuver]

allMeasStationaryGNN = deepcopy(allMeasStationary)
allMeasStationaryGNNNoisy = deepcopy(allMeasStationaryNoisy)
baseTime = time()
residualHistSta, errorHistSta, stdHistSta = GNNAssociations!(stationaryEphem, allMeasStationaryGNN)
time1 = time() - baseTime
baseTime = time()
residualHistStaNoisy, errorHistStaNoisy, stdHistStaNoisy = GNNAssociations!(stationaryEphem, allMeasStationaryGNNNoisy)
time2 = time() - baseTime
plotErrHist(stationaryEphem[1].time, errorHistSta)
AssocErrs = getAssocErrs(4, allMeasStationaryGNN)
AssocErrsNoisy = getAssocErrs(4, allMeasStationaryGNNNoisy)
plotAssocErrs(stationaryEphem[1].time, AssocErrs)
nt = length(allMeasStationaryGNN)
display("No Measurement Cases: $(count(AssocErrs[1, :].==0)/nt), $(count(AssocErrs[2,:].==0)/nt), $(count(AssocErrs[3,:].==0)/nt), $(count(AssocErrs[4,:].==0)/nt),")
display("Measurement Labeled UCT: $(count(AssocErrs[1, :].==1)/nt), $(count(AssocErrs[2,:].==1)/nt), $(count(AssocErrs[3,:].==1)/nt), $(count(AssocErrs[4,:].==1)/nt),")
display("Cross-Tagged: $(count(AssocErrs[1, :].==2)/nt), $(count(AssocErrs[2,:].==2)/nt), $(count(AssocErrs[3,:].==2)/nt), $(count(AssocErrs[4,:].==2)/nt),")
display("Clutter Correlated: $(count(AssocErrs[1, :].==3)/nt), $(count(AssocErrs[2,:].==3)/nt), $(count(AssocErrs[3,:].==3)/nt), $(count(AssocErrs[4,:].==3)/nt),")
display("Correct Association: $(count(AssocErrs[1, :].==4)/nt), $(count(AssocErrs[2,:].==4)/nt), $(count(AssocErrs[3,:].==4)/nt), $(count(AssocErrs[4,:].==4)/nt),")
display("Correct Association: $(count(AssocErrsNoisy[1, :].==4)/nt), $(count(AssocErrsNoisy[2,:].==4)/nt), $(count(AssocErrsNoisy[3,:].==4)/nt), $(count(AssocErrsNoisy[4,:].==4)/nt),")

Q = copy(astra1nManeuver.Q)
Q[4:6, 4:6] .*= 50 #True Assoc Maneuver case recovers in 1/2 day
allMeasManeuverGNN = deepcopy(allMeasManeuver)
allMeasManeuverGNNNoisy = deepcopy(allMeasManeuverNoisy)
baseTime = time()
residualHistMnv, errorHistMnv, stdHistMnv = GNNAssociations!(maneuverEphem, allMeasManeuverGNN; Q=Q, PG=0.99999)
time3 = time() - baseTime
baseTime = time()
residualHistMnvNoisy, errorHistMnvNoisy, stdHistMnvNoisy = GNNAssociations!(maneuverEphem, allMeasManeuverGNNNoisy; Q=Q, PG=0.99999)
time4 = time() - baseTime
plotErrHist(stationaryEphem[1].time, errorHistMnvNoisy)
AssocErrs = getAssocErrs(4, allMeasManeuverGNN)
AssocErrsNoisy = getAssocErrs(4, allMeasManeuverGNNNoisy)
plotAssocErrs(stationaryEphem[1].time, AssocErrs)
nt = length(allMeasManeuverGNN)
display("No Measurement Cases: $(count(AssocErrs[1, :].==0)/nt), $(count(AssocErrs[2,:].==0)/nt), $(count(AssocErrs[3,:].==0)/nt), $(count(AssocErrs[4,:].==0)/nt),")
display("Measurement Labeled UCT: $(count(AssocErrs[1, :].==1)/nt), $(count(AssocErrs[2,:].==1)/nt), $(count(AssocErrs[3,:].==1)/nt), $(count(AssocErrs[4,:].==1)/nt),")
display("Cross-Tagged: $(count(AssocErrs[1, :].==2)/nt), $(count(AssocErrs[2,:].==2)/nt), $(count(AssocErrs[3,:].==2)/nt), $(count(AssocErrs[4,:].==2)/nt),")
display("Clutter Correlated: $(count(AssocErrs[1, :].==3)/nt), $(count(AssocErrs[2,:].==3)/nt), $(count(AssocErrs[3,:].==3)/nt), $(count(AssocErrs[4,:].==3)/nt),")
display("Correct Association: $(count(AssocErrs[1, :].==4)/nt), $(count(AssocErrs[2,:].==4)/nt), $(count(AssocErrs[3,:].==4)/nt), $(count(AssocErrs[4,:].==4)/nt),")
display("Correct Association: $(count(AssocErrsNoisy[1, :].==4)/nt), $(count(AssocErrsNoisy[2,:].==4)/nt), $(count(AssocErrsNoisy[3,:].==4)/nt), $(count(AssocErrsNoisy[4,:].==4)/nt),")
jldsave("src/GNNAssocResults.jld2"; residualHistSta, errorHistSta, stdHistSta, residualHistMnv, errorHistMnv, stdHistMnv, residualHistStaNoisy, errorHistStaNoisy, stdHistStaNoisy, residualHistMnvNoisy, errorHistMnvNoisy, stdHistMnvNoisy, allMeasStationaryGNN, allMeasStationaryGNNNoisy, allMeasManeuverGNN, allMeasManeuverGNNNoisy)


function JPDAAssociations!(
    ephems::Vector{Ephemeris},
    measurements::Vector{Vector{Measurement}};
    Q::Matrix{Float64}=zeros(2, 2),
    PG::Float64=0.95,
)

    nt = length(ephems[1].time) - 1
    # nt = 10

    residualHist = zeros(2, nt, length(ephems))
    errorHist = zeros(6, nt, length(ephems))
    stdHist = zeros(6, nt, length(ephems))

    ests = Vector{Estimate}()
    priorCov = diagm([100.0, 100.0, 100.0, 1e-6, 1e-6, 1e-6])

    for ephem in ephems
        push!(ests, getInitialEstimate(ephem, priorCov, Q=Q))
    end

    # for i in 1:11
    for i in 1:nt
        t = i + 1
        display(ephems[1].time[t])
        for j in eachindex(ests)
            ukfPredict!(ests[j], stepEstimate)
            # display(ests[j].Mean - ephems[j].ephem[:, t])
        end
        residuals = Vector{Vector{Float64}}()
        if i in [1, 10, 100, 1000, 10000]
            JPDACorrect!(ests, measurements[i], [true, false], residuals; PG=PG, plotMeasSpace=false)
        else
            JPDACorrect!(ests, measurements[i], [true, false], residuals; PG=PG)
        end

        for j in eachindex(ests)
            residualHist[:, i, j] = residuals[j]
            # display(ests[j].Mean - ephems[j].ephem[:, t])
            errorHist[:, i, j] = ests[j].Mean - ephems[j].ephem[:, t]
            stdHist[:, i, j] = sqrt.(diag(ests[j].Cov))
        end
    end
    return residualHist, errorHist, stdHist
end
stationaryEphem = [astra1nStationary, astra1krStationary, astra1mStationary, astra1lStationary]
maneuverEphem = [astra1nManeuver, astra1krManeuver, astra1mManeuver, astra1lManeuver]

allMeasStationaryJPDA = deepcopy(allMeasStationary)
allMeasStationaryJPDANoisy = deepcopy(allMeasStationaryNoisy)
baseTime = time()
residualHistSta, errorHistSta, stdHistSta = JPDAAssociations!(stationaryEphem, allMeasStationaryJPDA)
time1 = time() - baseTime
baseTime = time()
residualHistStaNoisy, errorHistStaNoisy, stdHistStaNoisy = JPDAAssociations!(stationaryEphem, allMeasStationaryJPDANoisy)
time2 = time() - baseTime
plotErrHist(stationaryEphem[1].time, errorHistStaNoisy)
# AssocErrs = getAssocErrs(4, allMeasStationaryJPDA)
# plotAssocErrs(stationaryEphem[1].time, AssocErrs)
# need to build new plot for the betas over time
# + residual plot. DOn't plot 0 values, color code by sensor site
# plotResiduals(stationaryEphem[1].time, allMeasStationaryJPDA, residualHistSta)
nt = length(allMeasStationaryJPDA)

Q = copy(astra1nManeuver.Q)
Q[4:6, 4:6] .*= 50 #True Assoc Maneuver case recovers in 1/2 day
allMeasManeuverJPDA = deepcopy(allMeasManeuver)
allMeasManeuverJPDANoisy = deepcopy(allMeasManeuverNoisy)
baseTime = time()
residualHistMnv, errorHistMnv, stdHistMnv = JPDAAssociations!(maneuverEphem, allMeasManeuverJPDA; Q=Q, PG=0.99999)
time3 = time() - baseTime
baseTime = time()
residualHistMnvNoisy, errorHistMnvNoisy, stdHistMnvNoisy = JPDAAssociations!(maneuverEphem, allMeasManeuverJPDANoisy; Q=Q, PG=0.99999)
time4 = time() - baseTime
plotErrHist(stationaryEphem[1].time, errorHistMnvNoisy)
# # AssocErrs = getAssocErrs(4, allMeasManeuverJPDA)
# plotResiduals(stationaryEphem[1].time, allMeasManeuverJPDA, residualHistMnv)
# # plotAssocErrs(stationaryEphem[1].time, AssocErrs)
# nt = length(allMeasManeuverJPDA)
jldsave("src/JPDAAssocResults.jld2"; residualHistSta, errorHistSta, stdHistSta, residualHistMnv, errorHistMnv, stdHistMnv, residualHistStaNoisy, errorHistStaNoisy, stdHistStaNoisy, residualHistMnvNoisy, errorHistMnvNoisy, stdHistMnvNoisy, allMeasStationaryJPDA, allMeasStationaryJPDANoisy, allMeasManeuverJPDA, allMeasManeuverJPDANoisy)

#############################################################
function RBMCDAAssociations!(
    ephems::Vector{Ephemeris},
    measurements::Vector{Vector{Measurement}};
    Q::Matrix{Float64}=zeros(2, 2),
    PG::Float64=0.95,
    numParticles::Int64=100,
)

    nt = length(ephems[1].time) - 1
    # nt = 10

    # residualHist = zeros(2, nt, length(ephems))
    errorHistMMSE = zeros(6, nt, length(ephems))
    errorHistMAP = zeros(6, nt, length(ephems))
    stdHistMMSE = zeros(6, nt, length(ephems))
    stdHistMAP = zeros(6, nt, length(ephems))

    particles = Vector{Particle}()
    for _ in 1:numParticles
        ests = Vector{Estimate}()
        priorCov = diagm([100.0, 100.0, 100.0, 1e-6, 1e-6, 1e-6])

        for ephem in ephems
            push!(ests, getInitialEstimate(ephem, priorCov, Q=Q))
        end
        push!(particles, Particle(1 / numParticles, copy(ests), Vector{Vector{Int64}}()))
    end
    # for i in 1:200
    for i in 1:nt
        t = i + 1
        display(ephems[1].time[t])
        # Resample if needed
        weights = [particles[i].weight for i in 1:numParticles]
        Neff = 1 / sum(weights .^ 2)
        if Neff < (0.25 * numParticles) || count(==(0), weights) > (0.1 * numParticles)
            display(Neff)
            particles = resampleParticles(particles)
        end

        for p in eachindex(particles)
            for j in eachindex(particles[p].estimates)
                ukfPredict!(particles[p].estimates[j], stepEstimate)
                # display(ests[j].Mean - ephems[j].ephem[:, t])
            end
        end
        # residuals = Vector{Vector{Float64}}()
        if i in [1, 10, 100, 1000, 10000]
            RBMCDACorrect!(particles, measurements[i], [true, false]; PG=PG, plotMeasSpace=true)
        else
            RBMCDACorrect!(particles, measurements[i], [true, false]; PG=PG)
        end
        # return particles
        mmse = getMmseEstimate(particles)
        map = getMapEstimate(particles)
        for j in eachindex(particles[1].estimates)
            # residualHist[:, i, j] = residuals[j]
            # display(ests[j].Mean - ephems[j].ephem[:, t])
            errorHistMMSE[:, i, j] = mmse[j].Mean - ephems[j].ephem[:, t]
            errorHistMAP[:, i, j] = map[j].Mean - ephems[j].ephem[:, t]
            stdHistMMSE[:, i, j] = sqrt.(diag(mmse[j].Cov))
            stdHistMAP[:, i, j] = sqrt.(diag(map[j].Cov))
        end
    end
    return errorHistMMSE, errorHistMAP, stdHistMMSE, stdHistMAP, particles
end

stationaryEphem = [astra1nStationary, astra1krStationary, astra1mStationary, astra1lStationary]
maneuverEphem = [astra1nManeuver, astra1krManeuver, astra1mManeuver, astra1lManeuver]

allMeasStationaryRBMCDA = deepcopy(allMeasStationary)
allMeasStationaryRBMCDANoisy = deepcopy(allMeasStationaryNoisy)
baseTime = time()
errorHistMMSEs, errorHistMAPs, stdHistMMSEs, stdHistMAPs, particless = RBMCDAAssociations!(stationaryEphem, allMeasStationaryRBMCDA)
time1 = time() - baseTime
plotErrHist(stationaryEphem[1].time, errorHistMMSEs)
plotAssocErrs(allMeasStationaryRBMCDA)
# need to build new plot for the RBMCDA version
# plotResiduals(stationaryEphem[1].time, allMeasStationaryJPDA, residualHistSta)
# nt = length(allMeasStationaryJPDA)

Q = copy(astra1nManeuver.Q)
Q[4:6, 4:6] .*= 50 #True Assoc Maneuver case recovers in 1/2 day
allMeasManeuverRBMCDA = deepcopy(allMeasManeuver)
errorHistMMSEm, errorHistMAPm, stdHistMMSEm, stdHistMAPm, particlesm = RBMCDAAssociations!(maneuverEphem, allMeasManeuverRBMCDA; numParticles=100)
plotErrHist(stationaryEphem[1].time, errorHistMMSEm)
# # AssocErrs = getAssocErrs(4, allMeasManeuverJPDA)
# plotResiduals(stationaryEphem[1].time, allMeasManeuverJPDA, residualHistMnv)
# # plotAssocErrs(stationaryEphem[1].time, AssocErrs)
# nt = length(allMeasManeuverJPDA)

jldsave("src/RBMCDAAssocResults.jld2"; errorHistMMSEs, errorHistMAPs, stdHistMMSEs, stdHistMAPs, particless, errorHistMMSEm, errorHistMAPm, stdHistMMSEm, stdHistMAPm)