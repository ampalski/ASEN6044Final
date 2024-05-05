function RBMCDACorrect!(
    particles::Vector{Particle},
    measurements::Vector{Measurement},
    wrapVec::Vector{Bool};
    PG::Float64=0.95,
    plotMeasSpace::Bool=false,
)

    nParticles = length(particles)
    nEsts = length(particles[1].estimates)
    nMeas = length(measurements)
    ckPrior = 0.2 # assume no prior knowledge of clutter probabilities

    minRA, maxRA, minDec, maxDec = getMinMaxAngles(measurements)
    u1, u2 = getClutterDistr(minRA, maxRA, minDec, maxDec)
    V = 1 / (u1.b - u1.a) / (u2.b - u2.a)

    maxCost = quantile(Chisq(2), PG)

    # if plotMeasSpace
    # plotMeasurementSpace(measurements, allPredMeas[:, 1], Smatrix[:, 1]; PG=PG)
    # end
    particleHist = zeros(nMeas)
    for i in eachindex(particles)
        push!(particles[i].assocHistory, particleHist)
        for t in eachindex(particles[i].estimates)
            particles[i].estimates[t].MeasWithoutUpdate += 1
        end
    end
    for m in shuffle(1:nMeas)
        allPredMeas = fill(zeros(2), nParticles, nEsts)
        Kmatrix = fill(zeros(2, 2), nParticles, nEsts)
        Smatrix = fill(zeros(2, 2), nParticles, nEsts)
        for i in 1:nParticles
            for j in 1:nEsts
                allPredMeas[i, j], Smatrix[i, j], Kmatrix[i, j] = ukfGain(particles[i].estimates[j], measurements[m], getMeas)
                if !ishermitian(Smatrix[i, j])
                    Smatrix[i, j] = Hermitian(Smatrix[i, j])
                end
            end
        end
        # plotMeasurementSpace(measurements, allPredMeas[1, :], Smatrix[1, :]; PG=PG)
        assocVec = zeros(5)
        # For all particles
        for i in 1:nParticles
            measLikelihood = zeros(nEsts + 1)
            # Build the meas likelihood term
            for j in 1:nEsts
                distr = MvNormal(allPredMeas[i, j], Smatrix[i, j])
                measLikelihood[j] = pdf(distr, measurements[m].ReportedMeas)
            end
            measLikelihood[end] = V
            #Sample a c_k values
            πhat = ckPrior .* measLikelihood
            πhat ./= sum(πhat)
            ck = sample(collect(1:5), Weights(πhat), 1)[1]
            particles[i].assocHistory[end][m] = ck
            assocVec[ck] += 1 / nParticles

            #update the associated internal filter
            if ck <= nEsts
                newState, newCov, _ = applyUpdate(
                    measurements[m],
                    allPredMeas[i, ck],
                    wrapVec,
                    2,
                    particles[i].estimates[ck],
                    Smatrix[i, ck],
                    Kmatrix[i, ck],
                )

                particles[i].estimates[ck].Mean = newState
                particles[i].estimates[ck].Cov = newCov
                particles[i].estimates[ck].MeasurementApplied = true
                particles[i].estimates[ck].MeasWithoutUpdate = 0
            end
            # Calculate new weights
            # display("$(particles[i].weight) * $(ckPrior) * $(measLikelihood[ck]) / $(πhat[ck])")
            particles[i].weight = particles[i].weight * ckPrior * measLikelihood[ck] / πhat[ck]
        end
        # Normalize the weights
        normalizeWeights!(particles)
        # add assocVec to the measurement
        measurements[m].AssocVec = assocVec
    end
end


function normalizeWeights!(particles::Vector{Particle})
    # sum up total weight
    total = sum([particles[i].weight for i in eachindex(particles)])
    # divide each weight by sum
    for i in eachindex(particles)
        particles[i].weight /= total
    end
end

# These need to return either a mean and covariance or an Estimate object
function getMmseEstimate(particles::Vector{Particle})
    mmse = Vector{Estimate}()
    for i in eachindex(particles[1].estimates)
        mean = zeros(6)
        cov = zeros(6, 6)
        for particle in particles
            weight = particle.weight
            mean += weight .* particle.estimates[i].Mean
            cov += weight .* particle.estimates[i].Cov
        end
        push!(mmse, Estimate(i, mean, cov, particles[1].estimates[i].CurrentTime, true, particles[1].estimates[i].Q, 0))
    end
    return mmse
end

function getMapEstimate(particles::Vector{Particle})
    weights = [particles[i].weight for i in eachindex(particles)]
    maxWeightInd = argmax(weights)
    return particles[maxWeightInd].estimates
end

function resampleParticles(particles::Vector{Particle})
    weights = [particles[i].weight for i in eachindex(particles)]
    numParticles = length(particles)
    newParticles = Vector{Particle}()
    for i in 1:numParticles
        ind = sample(collect(1:numParticles), Weights(weights), 1)[1]
        push!(newParticles, deepcopy(particles[ind]))
        newParticles[i].weight = 1 / numParticles
    end
    return newParticles
end

function getRBMCDAAssocErrors(measurements::Vector{Vector{Measurement}})
    # 1 means there was no measurement of the target, and reports the % of UCTs associated with target
    # 2 means there was a measurement of the target, reports the % of UCTS associated with target
    # 3 means there was a measurement, reports the % of time it was tagged as UCT
    # 4 means cross-tag, reports the % of time measurement associated to wrong target
    # 5 is % of time we get correct assoc.
    associations = zeros(5, length(measurements), 4)
    for t in eachindex(measurements)
        estFound = falses(4)
        for measurement in measurements[t]
            est = measurement.TrueTgtID
            if est >= 100
                for i in 1:4
                    if estFound[i]
                        associations[2, t, i] += measurement.AssocVec[i]
                    else
                        associations[1, t, i] += measurement.AssocVec[i]
                    end
                end
                continue
            end
            estFound[est] = true
            for assoc in 1:5
                if assoc == est
                    associations[5, t, est] += measurement.AssocVec[assoc]
                elseif assoc == 5
                    associations[3, t, est] += measurement.AssocVec[assoc]
                else
                    associations[4, t, est] += measurement.AssocVec[assoc]
                end
            end
        end
    end
    return associations
end

function getRBMCDAStats(measurements::Vector{Vector{Measurement}})
    assoc = getRBMCDAAssocErrors(measurements)
    display("SC1 Correct Association: $(mean(assoc[5,:,1]))")
    display("SC2 Correct Association: $(mean(assoc[5,:,2]))")
    display("SC3 Correct Association: $(mean(assoc[5,:,3]))")
    display("SC4 Correct Association: $(mean(assoc[5,:,4]))")
end