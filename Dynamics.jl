function InertialOrbit(x, p, t)
    mu = 3.986e5
    r = norm(x[1:3])
    g = -mu / r^3 .* x[1:3]
    dx = zeros(6)

    dx[1] = x[4]
    dx[2] = x[5]
    dx[3] = x[6]
    dx[4] = g[1] + p[1]
    dx[5] = g[2] + p[2]
    dx[6] = g[3] + p[3]

    return dx
end

function stepEstimate(state::Vector{Float64})
    prob = ODEProblem(InertialOrbit, state, (0.0, 10.0), zeros(3))
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)

    return sol.u[end]
end

function buildEphem(initState, dt, T, Q, mnvs)
    N = length(0:dt:T)
    t = zeros(N)
    ephem = zeros(6, N)
    wDist = MvNormal(zeros(6), Q)
    nBurns = mnvs[2, 1] == 0 ? 0 : size(mnvs, 2)

    curState = copy(initState)
    curT = 0.0
    curMnv = 1

    t[1] = curT
    ephem[:, 1] = curState

    for i in 2:N
        prob = ODEProblem(InertialOrbit, curState, (0, dt), zeros(3))
        sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12)
        curState = sol.u[end] + rand(wDist)
        curT += dt

        if curT == mnvs[1, curMnv]
            curState[4:6] += mnvs[2:4, curMnv]
            curMnv = min(nBurns, curMnv + 1)
        end

        t[i] = curT
        ephem[:, i] = curState

    end

    return (t, ephem)
end

function buildAllEphem()
    astra1n = [-6802.42001282, -41613.740202976, -16.635482581, 3.03431508625, -0.496188946, -0.002621748732]
    astra1kr = [-6831.6984448575, -41591.05550192, 11.30691110203, 3.0350724847, -0.4995285472, 0.00181828073]
    astra1m = [-6770.100593874, -41615.97196304, 16.3729861001, 3.034714047137, -0.4951056425, -0.0020695362]
    astra1l = [-6817.442654717, -41593.8697042, -18.525171834, 3.035210337, -0.498058414, 0.000985242]
    # 10/25/23 at 13:51:10

    T = 86400 * 2
    dt = 10
    Q = zeros(6, 6)
    #J2 accel is ~1e-8, velocity is 1e-7 for 10s, pos is 5e-7 for 10 s, squared
    Q[1, 1] = 2.5e-13
    Q[2, 2] = 2.5e-13
    Q[3, 3] = 2.5e-13
    Q[4, 4] = 1e-14
    Q[5, 5] = 1e-14
    Q[6, 6] = 1e-14

    mnvs = zeros(2)

    t, ephem = buildEphem(astra1n, dt, T, Q, mnvs)
    astra1nStationary = Ephemeris(t, ephem, 1, 0.95, Q)
    t, ephem = buildEphem(astra1kr, dt, T, Q, mnvs)
    astra1krStationary = Ephemeris(t, ephem, 2, 0.95, Q)
    t, ephem = buildEphem(astra1m, dt, T, Q, mnvs)
    astra1mStationary = Ephemeris(t, ephem, 3, 0.95, Q)
    t, ephem = buildEphem(astra1l, dt, T, Q, mnvs)
    astra1lStationary = Ephemeris(t, ephem, 4, 0.95, Q)

    mnv1n = [86400.0, -0.9887, 0.3532, 2.8102]
    mnv1n[2:4] *= 0.001
    mnv1kr = [86400.0, -2.9868, 0.1712, -0.2234]
    mnv1kr[2:4] *= 0.001
    mnv1m = [86400.0, 0.8404, 2.8707, -0.2295]
    mnv1m[2:4] *= 0.001
    mnv1l = [86400.0, -1.16536, -2.2012, 1.6723]
    mnv1l[2:4] *= 0.001

    t, ephem = buildEphem(astra1n, dt, T, Q, mnv1n)
    astra1nManeuver = Ephemeris(t, ephem, 1, 0.95, Q)
    t, ephem = buildEphem(astra1kr, dt, T, Q, mnv1kr)
    astra1krManeuver = Ephemeris(t, ephem, 2, 0.95, Q)
    t, ephem = buildEphem(astra1m, dt, T, Q, mnv1m)
    astra1mManeuver = Ephemeris(t, ephem, 3, 0.95, Q)
    t, ephem = buildEphem(astra1l, dt, T, Q, mnv1l)
    astra1lManeuver = Ephemeris(t, ephem, 4, 0.95, Q)

    return (astra1nStationary, astra1krStationary, astra1mStationary, astra1lStationary, astra1nManeuver, astra1krManeuver, astra1mManeuver, astra1lManeuver)
end

## Verify how far away AWGN pushes the orbit state over two days
# astra1n = [-6802.42001282, -41613.740202976, -16.635482581, 3.03431508625, -0.496188946, -0.002621748732]
# astra1kr = [-6831.6984448575, -41591.05550192, 11.30691110203, 3.0350724847, -0.4995285472, 0.00181828073]
# astra1m = [-6770.100593874, -41615.97196304, 16.3729861001, 3.034714047137, -0.4951056425, -0.0020695362]
# astra1l = [-6817.442654717, -41593.8697042, -18.525171834, 3.035210337, -0.498058414, 0.000985242]
# dt = 10.0
# T = 86400 * 2.0

# astra1nStationary, astra1krStationary, astra1mStationary, astra1lStationary, astra1nManeuver, astra1krManeuver, astra1mManeuver, astra1lManeuver = buildAllEphem()

# prob = ODEProblem(InertialOrbit, astra1n, (0, T), zeros(3))
# sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12, tstops=0:dt:T)
# diff = sol.u[end] - astra1nStationary.ephem[:, end]
# display(norm(diff[1:3]))

# prob = ODEProblem(InertialOrbit, astra1kr, (0, T), zeros(3))
# sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12, tstops=0:dt:T)
# diff = sol.u[end] - astra1krStationary.ephem[:, end]
# display(norm(diff[1:3]))

# prob = ODEProblem(InertialOrbit, astra1m, (0, T), zeros(3))
# sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12, tstops=0:dt:T)
# diff = sol.u[end] - astra1mStationary.ephem[:, end]
# display(norm(diff[1:3]))

# prob = ODEProblem(InertialOrbit, astra1l, (0, T), zeros(3))
# sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12, tstops=0:dt:T)
# diff = sol.u[end] - astra1lStationary.ephem[:, end]
# display(norm(diff[1:3]))