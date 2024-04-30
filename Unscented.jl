function ukfTuningParams(n::Int)
    alpha = 0.001
    beta = 2
    kappa = 0
    lambda = 3 - n

    return alpha, beta, kappa, lambda
end

function computeUnscentedWeights(n::Int)
    alpha, beta, kappa, lambda = ukfTuningParams(n)

    nEntries = 2 * n + 1

    Wm = zeros(nEntries)
    Wc = copy(Wm)

    Wm[1] = lambda / (n + lambda)
    Wc[1] = lambda / (n + lambda) + (1 - alpha * alpha + beta)

    weight = 0.5 / (n + lambda)
    Wm[2:nEntries] .= weight
    Wc[2:nEntries] .= weight

    return Wm, Wc
end

function ukfPredict!(est::Estimate, propFun)
    # Set up Variables
    n = length(est.Mean)
    twoNp1 = 2 * n + 1

    (alpha, beta, kappa, lambda) = ukfTuningParams(n)
    (Wm, Wc) = computeUnscentedWeights(n)

    Chi = zeros(n, twoNp1)
    ChiP = copy(Chi) #propagated sigma points

    newState = zeros(n)
    newCov = zeros(n, n)

    # Compute Sigma Points
    Chi[:, 1] = est.Mean
    sqrtP = (n + lambda) * est.Cov
    !isposdef(sqrtP) && (sqrtP = Hermitian(sqrtP))
    C = cholesky(sqrtP)
    sqrtP = C.L

    for i in 1:n
        Chi[:, i+1] = est.Mean + sqrtP[:, i]
        Chi[:, i+1+n] = est.Mean - sqrtP[:, i]
    end

    # Predict State
    for i in 1:twoNp1
        ChiP[:, i] = propFun(Chi[:, i])

        newState += Wm[i] * ChiP[:, i]
    end

    # Predict Covariance
    for i in 1:twoNp1
        newCov += Wc[i] * (ChiP[:, i] - newState) * (ChiP[:, i] - newState)'
    end

    newCov += est.Q

    est.Cov = newCov
    est.Mean = newState
    est.CurrentTime += 10.0
    est.MeasurementApplied = false
end

# From here down, may need to rework to incorporate the Estimate struct and the wrap stuff
# May also need to break apart to work for JPDA, RBPF, etc.
# Keep everything through the creation of the Kalman gain in one function,
# returning the predicted measurement, S, and K. Then the actual update can happen separately.
function ukfGain(
    #function ukfCorrect(
    est::Estimate,
    meas::Measurement,
    measFun::Function,
)

    # Set up Variables
    n = length(est.Mean)
    p = length(meas.ReportedMeas)
    twoNp1 = 2 * n + 1

    # Force wrap to a vector
    # if typeof(wrapAngMeas) <: Bool
    #     wrapMeas = fill(wrapAngMeas, p)
    # else
    #     wrapMeas = wrapAngMeas
    # end
    wrapMeas = [true, false]

    (alpha, beta, kappa, lambda) = ukfTuningParams(n)
    (Wm, Wc) = computeUnscentedWeights(n)

    Chi = zeros(n, twoNp1)
    gamma = zeros(p, twoNp1) #transformed sigma points
    measPred = zeros(p) #predicted measurement

    # Compute Sigma Points
    Chi[:, 1] = est.Mean
    sqrtP = (n + lambda) * est.Cov
    !isposdef(sqrtP) && (sqrtP = Hermitian(sqrtP))
    if !isposdef(sqrtP)
        display(sqrtP)
    end
    C = cholesky(sqrtP)
    sqrtP = C.L

    for i in 1:n
        Chi[:, i+1] = est.Mean + sqrtP[:, i]
        Chi[:, i+1+n] = est.Mean - sqrtP[:, i]
    end

    # Transform Sigma Points to Measurement space
    for i in 1:twoNp1
        gamma[:, i] = measFun(Chi[:, i], meas.ObsPos)
    end

    # Check if gamma values span the 0, 2pi divide and rescale if allowed
    low = 10 * pi / 180
    high = 2 * pi - low
    for i in 1:p
        if wrapMeas[i]
            if minimum(gamma[i, :]) < low && maximum(gamma[i, :]) > high
                gamma[i, gamma[i, :].>pi] .-= (2 * pi)
            end
        end
    end

    # Create predicted measurement
    for i in 1:twoNp1
        measPred += Wm[i] * gamma[:, i]
    end

    # Create Measurement Covariance
    S = zeros(p, p)
    C = zeros(n, p)
    for i in 1:twoNp1
        S += Wc[i] * (gamma[:, i] - measPred) * (gamma[:, i] - measPred)'
        C += Wc[i] * (Chi[:, i] - est.Mean) * (gamma[:, i] - measPred)'
    end
    S = S + meas.R

    # Create Kalman Gain and Update
    K = C / S
    return (measPred, S, K)

    # residual = createResidual(meas, measPred, wrapMeas, p)
    # newState = state + K * residual
    # newCov = cov - K * S * K'
    # return newState, newCov, residual
end

function applyUpdate(meas, measPred, wrapMeas, p, est, S, K)
    residual = createResidual(meas.ReportedMeas, measPred, wrapMeas, p)
    newState = est.Mean + K * residual
    newCov = est.Cov - K * S * K'
    return newState, newCov, residual
end

function createResidual(
    meas::AbstractVector,
    measPred::AbstractVector,
    wrapMeas::AbstractVector,
    p::Int)

    residual = zeros(p)
    low = 10 * pi / 180
    high = 2 * pi - low

    for i in 1:p
        if wrapMeas[i]
            if meas[i] < low && measPred[i] > high
                measPred[i] -= (2 * pi)
            elseif meas[i] > high && measPred[i] < low
                meas[i] -= (2 * pi)
            end
        end
        residual[i] = meas[i] - measPred[i]
    end

    return residual
end

function getInitialEstimate(ephem::Ephemeris, priorCov::Matrix{Float64}; Q::Matrix{Float64}=zeros(2, 2))
    v = MvNormal(ephem.ephem[:, 1], priorCov)
    mean = rand(v)

    useQ = maximum(Q) > 0 ? copy(Q) : copy(ephem.Q)

    return Estimate(ephem.ObjectID, mean, priorCov, 0.0, false, useQ)
end