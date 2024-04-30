function unit(x)
    return (1 / norm(x) * x)
end

function unit!(x)
    len = norm(x)
    x ./= len
end

# Wrap an unbounded angle to the interval [0, 2pi]
function wrapTo2pi(input::Float64)
    p2 = 2 * pi
    while input < 0
        input += p2
    end

    return input % p2
end


function cart2sph(vec)
    ra = wrapTo2pi(atan(vec[2], vec[1]))
    dec = atan(vec[3], sqrt(vec[1]^2 + vec[2]^2))
    return [ra, dec]
end

function sph2cart(radec)
    x = cos(radec[2]) * cos(radec[1])
    y = cos(radec[2]) * sin(radec[1])
    z = sin(radec[2])
    return [x, y, z]
end

function getRMSE(errorHist)
    rmse = zeros(4)
    nt = size(errorHist, 2)
    for i in 1:4
        for j in 1:nt
            rmse[i] += errorHist[1:3, j, i]' * errorHist[1:3, j, i]
        end
    end
    rmse ./= nt
    return sqrt.(rmse)
end