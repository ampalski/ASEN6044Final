function getMeas(state, obsState)
    los = state[1:3] - obsState
    meas = cart2sph(los)
    meas[1] = wrapTo2pi(meas[1])
    return cart2sph(los)
end

function getNoisyMeas(state, R, obsState)
    meas = getMeas(state, obsState)
    v = MvNormal(meas, R)
    noisyMeas = rand(v)
    noisyMeas[1] = wrapTo2pi(noisyMeas[1])
    return (meas, noisyMeas)
end

function generateMeas(ephems::Vector{Ephemeris}, lambda::Int)
    # Randomly select an observation site, rotate to ECI, 
    # get measurements for each object. 1-Pd chance of no ob.
    # Add in Poisson distr false detects randomly dispersed in the frame with λ=~2
    site1 = [4119.001, 549.315, 4822.577]
    site2 = [5056.57, 2689.844, -2797.323]
    R = zeros(2, 2)
    R[1, 1] = 2e-10
    R[2, 2] = 2e-10 #14 urad accuracy, ~500 m projected at GEO 
    R2 = 100 .* R

    allMeas = Vector{Vector{Measurement}}()
    p = Poisson(lambda)
    numObjs = length(ephems)
    ind = 2
    for t in ephems[1].time
        t == 0.0 && (continue)

        # site = rand() < 0.5 ? ECF2ECI(site1, t) : ECF2ECI(site2, t)
        if rand() < 0.5
            site = ECF2ECI(site1, t)
            useR = copy(R)
        else
            site = ECF2ECI(site2, t)
            useR = copy(R2)
        end

        measurements = Vector{Measurement}()
        obsID = 1
        numFalse = rand(p)
        maxRA = -999.0
        minRA = 999.0
        minDec = 999.0
        maxDec = -999.0

        for ephem in ephems
            if rand() > ephem.PD
                continue
            end
            state = ephem.ephem[:, ind]
            truMeas, noisyMeas = getNoisyMeas(state, useR, site)
            if noisyMeas[1] < minRA
                minRA = noisyMeas[1]
            end
            if noisyMeas[1] > maxRA
                maxRA = noisyMeas[1]
            end
            if noisyMeas[2] < minDec
                minDec = noisyMeas[2]
            end
            if noisyMeas[2] > maxDec
                maxDec = noisyMeas[2]
            end

            push!(measurements, Measurement(
                obsID,
                site,
                t,
                truMeas,
                noisyMeas,
                R,
                ephem.ObjectID,
                zeros(numObjs + numFalse)
            ))
            obsID += 1
        end

        u1, u2 = getClutterDistr(minRA, maxRA, minDec, maxDec)

        for i in 1:numFalse
            meas = [rand(u1), rand(u2)]

            push!(measurements, Measurement(
                obsID,
                site,
                t,
                meas,
                meas,
                useR,
                100 + i,
                zeros(numObjs + numFalse)
            ))
            obsID += 1
        end

        push!(allMeas, measurements)
        ind += 1
    end
    return allMeas
end

#t is the number of seconds after the scenario start epoch
#using the USNO approximation of GMST
function ECF2ECI(pos, t)
    baseEpoch = 2460243.077199
    epoch = baseEpoch + t / 86400.0

    JD0 = floor(epoch) + 0.5
    JD0 > epoch && (JD0 -= 1)
    H = 24.0 * (epoch - JD0)

    D = epoch - 2451545.0
    T = D / 36525
    GMST = mod(6.697375 + 0.065709824279 * D + 1.0027379 * H + 0.0854103 * T + 0.0000258 * T^2, 24.0)
    GMST = GMST * pi / 12 #to degrees, to radians

    ecf2eci = zeros(3, 3)
    ecf2eci[3, 3] = 1.0
    c = cos(GMST)
    s = sin(GMST)
    ecf2eci[1, 1] = c
    ecf2eci[2, 2] = c
    ecf2eci[1, 2] = -s
    ecf2eci[2, 1] = s

    return ecf2eci * pos

end

function getClutterDistr(minRA, maxRA, minDec, maxDec)
    buffer = 0.05 * pi / 180
    u1 = Uniform(minRA - buffer, maxRA + buffer)
    u2 = Uniform(minDec - buffer, maxDec + buffer)

    return (u1, u2)
end

function getMinMaxAngles(measurements::Vector{Measurement})
    maxRA = -999.0
    minRA = 999.0
    minDec = 999.0
    maxDec = -999.0

    for meas in measurements
        noisyMeas = meas.ReportedMeas
        if noisyMeas[1] < minRA
            minRA = noisyMeas[1]
        end
        if noisyMeas[1] > maxRA
            maxRA = noisyMeas[1]
        end
        if noisyMeas[2] < minDec
            minDec = noisyMeas[2]
        end
        if noisyMeas[2] > maxDec
            maxDec = noisyMeas[2]
        end
    end
    return (minRA, maxRA, minDec, maxDec)
end
# stationaryEphem = [astra1nStationary, astra1krStationary, astra1mStationary, astra1lStationary]
# stationaryEphem2 = [astra1nStationary2, astra1krStationary2, astra1mStationary2, astra1lStationary2]
# maneuverEphem = [astra1nManeuver, astra1krManeuver, astra1mManeuver, astra1lManeuver]

# allMeasStationary = generateMeas(stationaryEphem, 2)
# allMeasManeuver = generateMeas(maneuverEphem, 2)
# allMeasStationaryNoisy = generateMeas(stationaryEphem, 2)
# allMeasManeuverNoisy = generateMeas(maneuverEphem, 2)

# allMeasStationary5 = generateMeas(stationaryEphem2, 5)
# allMeasStationary10 = generateMeas(stationaryEphem2, 10)
# allMeasStationary25 = generateMeas(stationaryEphem2, 25)
# allMeasStationary50 = generateMeas(stationaryEphem2, 50)
# allMeasStationary100 = generateMeas(stationaryEphem2, 100)
# jldsave("src/MeasLambda.jld2"; allMeasStationary5, allMeasStationary10, allMeasStationary25, allMeasStationary50, allMeasStationary100)