function JPDACorrect!(
    ests::Vector{Estimate},
    measurements::Vector{Measurement},
    wrapVec::Vector{Bool},
    residuals::Vector{Vector{Float64}};
    PG::Float64=0.95,
    plotMeasSpace::Bool=false,
)

    nEsts = length(ests)
    nMeas = length(measurements)
    p = length(wrapVec)
    empty!(residuals)
    PD = 0.95

    validation = fill(false, nEsts + 1, nMeas)

    allResids = fill(zeros(2), nEsts, nMeas)
    allPredMeas = fill(zeros(2), nEsts, nMeas)
    Kmatrix = fill(zeros(2, 2), nEsts, nMeas)
    Smatrix = fill(zeros(2, 2), nEsts, nMeas)

    maxCost = quantile(Chisq(2), PG)

    validation[end, :] .= true
    for i in 1:nEsts
        for j in 1:nMeas
            allPredMeas[i, j], Smatrix[i, j], Kmatrix[i, j] = ukfGain(ests[i], measurements[j], getMeas)
            allResids[i, j] = createResidual(measurements[j].ReportedMeas, allPredMeas[i, j], wrapVec, p)
            cost = allResids[i, j]' / Smatrix[i, j] * allResids[i, j]
            if cost < maxCost
                validation[i, j] = true
            end
        end
        push!(residuals, zeros(p))
    end

    if plotMeasSpace
        plotMeasurementSpace(measurements, allPredMeas[:, 1], Smatrix[:, 1]; PG=PG)
    end
    # Evaluate Joint probabilities
    feasibleEvents = getEvents(validation)
    eventProbabilities = fill(2.0^nMeas, length(feasibleEvents))
    for i in eachindex(eventProbabilities)
        measTerm = 1

        for j in 1:nMeas
            # multiply 1/lambda * normal(z; zhat, S)
            estInd = findfirst(feasibleEvents[i][:, j])
            if estInd > nEsts
                continue
            end
            if !ishermitian(Smatrix[estInd, j])
                Smatrix[estInd, j] = Hermitian(Smatrix[estInd, j])
            end
            distr = MvNormal(allPredMeas[estInd, j], Smatrix[estInd, j])
            measTerm *= (pdf(distr, measurements[j].ReportedMeas) / 2) # λ=2
        end
        eventProbabilities[i] *= measTerm
        estTerm = 1
        for t in 1:nEsts
            if count(feasibleEvents[i][t, :]) > 0
                estTerm *= PD
            else
                estTerm *= (1 - PD)
            end
        end
        eventProbabilities[i] *= estTerm
    end
    eventProbabilities ./= sum(eventProbabilities)

    # Create marginal probabilities
    betas = zeros(nEsts, nMeas + 1)
    for t in 1:nEsts
        for j in 1:nMeas
            for k in eachindex(eventProbabilities)
                if feasibleEvents[k][t, j]
                    betas[t, j] += eventProbabilities[k]
                end
            end
        end
        betas[t, end] = 1 - PG * PD
        # display(betas[t, :])
        betas[t, :] ./= sum(betas[t, :])
        # display(betas[t, :])
    end

    # Update estimates
    for t in 1:nEsts
        ests[t].MeasWithoutUpdate += 1
        # Build the residual
        for j in 1:nMeas
            residuals[t] += betas[t, j] * allResids[t, j]
        end

        # Build Ptilde
        innerSection = zeros(p, p)
        for j in 1:nMeas
            innerSection += betas[t, j] * allResids[t, j] * allResids[t, j]'
        end
        innerSection -= (residuals[t] * residuals[t]')
        Ptilde = Kmatrix[t, 1] * innerSection * Kmatrix[t, 1]'

        # Update estimate
        ests[t].Cov = ests[t].Cov + (betas[t, end] - 1) * Kmatrix[t, 1] * Smatrix[t, 1] * Kmatrix[t, 1]' + Ptilde
        ests[t].Mean += Kmatrix[t, 1] * residuals[t]
        ests[t].MeasurementApplied = true
        if residuals[t] != zeros(p)
            ests[t].MeasWithoutUpdate = 0
        end
    end

    # Update measurements
    for j in eachindex(measurements)
        measurements[j].AssocVec = betas[:, j]
    end

end

function getEvents(validation::Matrix{Bool})
    allEvents = Vector{Matrix{Bool}}()
    curValid = copy(validation)
    curInd = CartesianIndex(1, 1)
    createEvent!(allEvents, curValid, curInd)
    return allEvents
end

function createEvent!(allEvents::Vector{Matrix{Bool}}, valid::Matrix{Bool}, curInd)
    n, m = size(valid)
    holdValid = copy(valid)
    newValid = copy(holdValid)
    i = curInd[1]
    j = curInd[2]
    while i < n
        if newValid[i, j]
            for k in (j+1):m
                newValid[i, k] = false
            end
            for k in (i+1):n
                newValid[k, j] = false
            end
            newInd = j == m ? CartesianIndex(i + 1, 1) : CartesianIndex(i, j + 1)
            createEvent!(allEvents, newValid, newInd)
            holdValid[i, j] = false
            newValid = copy(holdValid)
            newValid[i, j] = false
        end

        if j == m
            j = 1
            i += 1
        else
            j += 1
        end
    end
    push!(allEvents, newValid)
end