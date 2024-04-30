function Munkres(inMatrix)
    matrix = copy(inMatrix)
    n, m = size(matrix)
    k = min(n, m)
    isSquare = n == m ? true : false

    coveredCol = falses(m)
    coveredRow = falses(n)

    # Preliminaries
    if isSquare || n < m
        for i in 1:n
            minRow = minimum(matrix[i, :])
            matrix[i, :] .-= minRow
        end
    end

    if isSquare || n > m
        for i in 1:m
            minCol = minimum(matrix[:, i])
            matrix[:, i] .-= minCol
        end
    end

    # Step 1
    specialZeros = step1(matrix)
    nextStep = 2
    uncoveredInd = CartesianIndex(1, 1)
    while nextStep > 0
        # Step 2
        if nextStep == 2
            for i in 1:m
                count(specialZeros[:, i] .== 1) > 0 && (coveredCol[i] = true)
            end
            if count(coveredCol) == k
                nextStep = 0
            else
                nextStep = 3
            end
            # Step 3
        elseif nextStep == 3
            found, uncoveredInd = findUncoveredZero(matrix, coveredCol, coveredRow)
            if found
                specialZeros[uncoveredInd] = 2
                starredInRow = findall(specialZeros[uncoveredInd[1], :] .== 1)
                if isempty(starredInRow)
                    nextStep = 4
                else
                    coveredRow[uncoveredInd[1]] = true
                    coveredCol[starredInRow[1]] = false
                end
            else
                nextStep = 5
            end
            #Step 4
        elseif nextStep == 4
            step4!(specialZeros, coveredCol, coveredRow, uncoveredInd)
            nextStep = 2
        elseif nextStep == 5
            step5!(matrix, coveredCol, coveredRow)
            nextStep = 3
        end
    end
    return findall(specialZeros .== 1)
end

function step1(matrix)
    n, m = size(matrix)
    specialZeros = zeros(Int, n, m) # 1 for star, 2 for prime
    inds = findall(matrix .== 0.0)

    for ind in inds
        count(specialZeros[ind[1], :] .== 1) > 0 && continue
        count(specialZeros[:, ind[2]] .== 1) > 0 && continue

        specialZeros[ind] = 1
    end
    return specialZeros
end

function findUncoveredZero(matrix, coveredCol, coveredRow)
    for ind in CartesianIndices(matrix)
        if coveredCol[ind[2]] || coveredRow[ind[1]] || matrix[ind] != 0
            continue
        end
        return (true, ind)
    end
    return (false, CartesianIndex(1, 1))
end

function step4!(specialZeros, coveredCol, coveredRow, initInd)
    primed = [initInd]
    starred = Vector{CartesianIndex}()

    lastInd = initInd
    temp = findall(specialZeros[:, lastInd[2]] .== 1)
    while length(temp) > 0
        lastInd = CartesianIndex(temp[1], lastInd[2])
        push!(starred, lastInd)

        temp = findall(specialZeros[lastInd[1], :] .== 2)
        lastInd = CartesianIndex(lastInd[1], temp[1])
        push!(primed, lastInd)

        temp = findall(specialZeros[:, lastInd[2]] .== 1)
    end
    for ind in starred
        specialZeros[ind] = 0
    end
    for ind in primed
        specialZeros[ind] = 1
    end
    specialZeros[specialZeros.==2] .= 0
    coveredCol = falses(length(coveredCol))
    coveredRow = falses(length(coveredRow))
end

function step5!(matrix, coveredCol, coveredRow)
    h = maximum(matrix)
    for ind in CartesianIndices(matrix)
        if coveredCol[ind[2]] || coveredRow[ind[1]]
            continue
        end
        if matrix[ind] < h
            h = matrix[ind]
        end
    end
    for row in eachindex(coveredRow)
        if coveredRow[row]
            matrix[row, :] .+= h
        end
    end

    for col in eachindex(coveredCol)
        if !coveredCol[col]
            matrix[:, col] .-= h
        end
    end
end
