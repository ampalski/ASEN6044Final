function GNNCorrect!(
    ests::Vector{Estimate},
    measurements::Vector{Measurement},
    wrapVec::Vector{Bool},
    residuals::Vector{Vector{Float64}};
    PG::Float64=0.95,
)

    nEsts = length(ests)
    nMeas = length(measurements)
    p = length(wrapVec)
    empty!(residuals)

    costs = fill(100.0, nEsts, nMeas)
    allResids = fill(zeros(2), nEsts, nMeas)
    allPredMeas = fill(zeros(2), nEsts, nMeas)
    Kmatrix = fill(zeros(2, 2), nEsts, nMeas)
    Smatrix = fill(zeros(2, 2), nEsts, nMeas)

    maxCost = quantile(Chisq(2), PG)

    for i in 1:nEsts
        for j in 1:nMeas
            allPredMeas[i, j], Smatrix[i, j], Kmatrix[i, j] = ukfGain(ests[i], measurements[j], getMeas)
            allResids[i, j] = createResidual(measurements[j].ReportedMeas, allPredMeas[i, j], wrapVec, p)
            cost = allResids[i, j]' / Smatrix[i, j] * allResids[i, j]
            if cost < maxCost
                costs[i, j] = cost
            end
        end
        push!(residuals, zeros(p))
    end

    assignments = Munkres(costs)
    baseAssocVec = zeros(nEsts + 1, nMeas)

    for assignment in assignments
        if costs[assignment] < 100
            baseAssocVec[assignment] = 1
        else
            baseAssocVec[end, assignment[2]] = 1
        end
    end
    for i in 1:nEsts
        measInd = findall(baseAssocVec[i, :] .== 1)
        if !isempty(measInd)
            j = measInd[1]
            newState, newCov, _ = applyUpdate(measurements[j], allPredMeas[i, j], wrapVec, p, ests[i], Smatrix[i, j], Kmatrix[i, j])
            ests[i].Mean = newState
            ests[i].Cov = newCov
            ests[i].MeasurementApplied = true
            residuals[i] = allResids[i, j]
        end
    end

    for j in 1:nMeas
        measurements[j].AssocVec = baseAssocVec[:, j]
    end

end

function getAssocErrs(nEsts, measurements)
    # 0 means there was no measurement of the target, and no measurements were applied 
    # 1 means there was a measurement, but called out as a UCT 
    # 2 means cross-tag 
    # 3 means UCT correlated as the target 
    # 4 is correct assoc.
    errors = zeros(nEsts, length(measurements))
    for j in eachindex(measurements)
        for measurement in measurements[j]
            assocID = findfirst(measurement.AssocVec .== 1)
            isnothing(assocID) && (assocID = 0)
            trueID = measurement.TrueTgtID
            if trueID < 100
                if assocID == (nEsts + 1) || assocID == 0
                    errors[trueID, j] = 1
                elseif trueID == assocID
                    errors[trueID, j] = 4
                else
                    errors[trueID, j] = 2
                end
            else
                if assocID > 0 && assocID <= nEsts
                    errors[assocID, j] = 3
                end
            end
        end
    end
    return errors
end
