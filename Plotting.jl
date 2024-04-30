function plotErrHist(time, errorHist)
    set_theme!(theme_black())
    fig = Figure(size=(800, 1000))
    ax1 = Axis(fig[1, 1])
    ax2 = Axis(fig[2, 1])
    ax3 = Axis(fig[3, 1])
    ax4 = Axis(fig[4, 1])
    ax4.xlabel = "Time (days)"
    ax1.ylabel = "SC1 Error (km)"
    ax2.ylabel = "SC2 Error (km)"
    ax3.ylabel = "SC3 Error (km)"
    ax4.ylabel = "SC4 Error (km)"

    xplot = (1 / 86400) .* time[2:end]
    lines!(ax1, xplot, errorHist[1, :, 1], label="X")
    lines!(ax1, xplot, errorHist[2, :, 1], label="Y")
    lines!(ax1, xplot, errorHist[3, :, 1], label="Z")
    lines!(ax2, xplot, errorHist[1, :, 2], label="X")
    lines!(ax2, xplot, errorHist[2, :, 2], label="Y")
    lines!(ax2, xplot, errorHist[3, :, 2], label="Z")
    lines!(ax3, xplot, errorHist[1, :, 3], label="X")
    lines!(ax3, xplot, errorHist[2, :, 3], label="Y")
    lines!(ax3, xplot, errorHist[3, :, 3], label="Z")
    lines!(ax4, xplot, errorHist[1, :, 4], label="X")
    lines!(ax4, xplot, errorHist[2, :, 4], label="Y")
    lines!(ax4, xplot, errorHist[3, :, 4], label="Z")
    Label(fig[0, :], "Error History", fontsize=20, tellwidth=false)
    axislegend(ax4; position=:rb)
    display(GLMakie.Screen(), fig)

end
function plotAssocErrs(time, AssocErrs)
    set_theme!(theme_black())
    fig = Figure(size=(800, 1000))
    ax1 = Axis(fig[1, 1])
    ax2 = Axis(fig[2, 1])
    ax3 = Axis(fig[3, 1])
    ax4 = Axis(fig[4, 1])
    ax4.xlabel = "Time (days)"
    ax1.ylabel = "SC1 Association Status"
    ax2.ylabel = "SC2 Association Status"
    ax3.ylabel = "SC3 Association Status"
    ax4.ylabel = "SC4 Association Status"

    xplot = (1 / 86400) .* time[2:end]
    scatter!(ax1, xplot, AssocErrs[1, :])
    scatter!(ax2, xplot, AssocErrs[2, :])
    scatter!(ax3, xplot, AssocErrs[3, :])
    scatter!(ax4, xplot, AssocErrs[4, :])
    display(GLMakie.Screen(), fig)

end

function plotResiduals(tvec::Vector{Float64}, allMeasurements::Vector{Vector{Measurement}}, residuals::Array{Float64,3})
    # Get a bool vector of when each sensor was being used 
    # Then for each vehicle build a plotting vector, only include non-zero resids

    set_theme!(theme_black())
    fig = Figure(size=(800, 1000))
    ax1 = Axis(fig[1, 1])
    ax2 = Axis(fig[2, 1])
    ax3 = Axis(fig[3, 1])
    ax4 = Axis(fig[4, 1])
    ax4.xlabel = "Time (days)"
    ax1.ylabel = "SC1 Residuals"
    ax2.ylabel = "SC2 Residuals"
    ax3.ylabel = "SC3 Residuals"
    ax4.ylabel = "SC4 Residuals"

    xplot = (1 / 86400) .* tvec[2:end]
    sc1RA = residuals[1, :, 1]
    sc1DEC = residuals[2, :, 1]
    sc2RA = residuals[1, :, 2]
    sc2DEC = residuals[2, :, 2]
    sc3RA = residuals[1, :, 3]
    sc3DEC = residuals[2, :, 3]
    sc4RA = residuals[1, :, 4]
    sc4DEC = residuals[2, :, 4]

    site1 = fill(true, length(tvec) - 1)
    site2 = fill(true, length(tvec) - 1)
    for i in eachindex(allMeasurements)
        if allMeasurements[i][1].ObsPos[3] > 0
            site2[i] = false
        else
            site1[i] = false
        end
    end
    sc1mask = sc1RA .!= 0.0
    sc2mask = sc2RA .!= 0.0
    sc3mask = sc3RA .!= 0.0
    sc4mask = sc4RA .!= 0.0

    scatter!(ax1, xplot[site1.&sc1mask], sc1RA[site1.&sc1mask], color=:cyan, label="Site 1 Right Ascension")
    scatter!(ax1, xplot[site2.&sc1mask], sc1RA[site2.&sc1mask], color=:green, label="Site 2 Right Ascension")
    scatter!(ax1, xplot[site1.&sc1mask], sc1DEC[site1.&sc1mask], color=:yellow, label="Site 1 Declination")
    scatter!(ax1, xplot[site2.&sc1mask], sc1DEC[site2.&sc1mask], color=:red, label="Site 2 Declination")
    scatter!(ax2, xplot[site1.&sc2mask], sc2RA[site1.&sc2mask], color=:cyan)
    scatter!(ax2, xplot[site2.&sc2mask], sc2RA[site2.&sc2mask], color=:green)
    scatter!(ax2, xplot[site1.&sc2mask], sc2DEC[site1.&sc2mask], color=:yellow)
    scatter!(ax2, xplot[site2.&sc2mask], sc2DEC[site2.&sc2mask], color=:red)
    scatter!(ax3, xplot[site1.&sc3mask], sc3RA[site1.&sc3mask], color=:cyan)
    scatter!(ax3, xplot[site2.&sc3mask], sc3RA[site2.&sc3mask], color=:green)
    scatter!(ax3, xplot[site1.&sc3mask], sc3DEC[site1.&sc3mask], color=:yellow)
    scatter!(ax3, xplot[site2.&sc3mask], sc3DEC[site2.&sc3mask], color=:red)
    scatter!(ax4, xplot[site1.&sc4mask], sc4RA[site1.&sc4mask], color=:cyan)
    scatter!(ax4, xplot[site2.&sc4mask], sc4RA[site2.&sc4mask], color=:green)
    scatter!(ax4, xplot[site1.&sc4mask], sc4DEC[site1.&sc4mask], color=:yellow)
    scatter!(ax4, xplot[site2.&sc4mask], sc4DEC[site2.&sc4mask], color=:red)
    display(GLMakie.Screen(), fig)
end

function plotMeasurementSpace(
    measurements::Vector{Measurement},
    allPredMeas::Vector{Vector{Float64}},
    Smatrix::Vector{Matrix{Float64}};
    PG=0.95,
)
    set_theme!(theme_black())
    fig = Figure(size=(800, 800))
    ax1 = Axis(fig[1, 1])
    ax1.xlabel = "Relative Right Ascension Difference (rad)"
    ax1.ylabel = "Relative Declination Difference (rad)"

    ref = measurements[1].ReportedMeas
    # Plot Measurements
    for meas in measurements
        plotval = meas.ReportedMeas - ref
        clr = meas.TrueTgtID < 100 ? :green : :red
        scatter!(ax1, plotval[1], plotval[2], color=clr, marker=:circle)
    end
    # Plot Predicted measurements and innovation covariance
    for i in eachindex(allPredMeas)
        pred = allPredMeas[i]
        S = Smatrix[i]
        plotval = pred - ref
        scatter!(ax1, plotval[1], plotval[2], color=:cyan, marker=:utriangle)
        xvals, yvals = getellipsepoints(plotval, S; confidence=PG)
        lines!(ax1, xvals, yvals, color=:yellow)
    end
    display(GLMakie.Screen(), fig)

end

# From: https://discourse.julialang.org/t/plot-ellipse-in-makie/82814/4

function getellipsepoints(cx, cy, rx, ry, θ)
    t = range(0, 2 * pi, length=100)
    ellipse_x_r = @. rx * cos(t)
    ellipse_y_r = @. ry * sin(t)
    R = [cos(θ) sin(θ); -sin(θ) cos(θ)]
    r_ellipse = [ellipse_x_r ellipse_y_r] * R
    x = @. cx + r_ellipse[:, 1]
    y = @. cy + r_ellipse[:, 2]
    return (x, y)
end

function getellipsepoints(μ, Σ; confidence=0.95)
    quant = sqrt(quantile(Chisq(2), confidence))
    cx = μ[1]
    cy = μ[2]

    egvs = eigvals(Σ)
    if egvs[1] > egvs[2]
        idxmax = 1
        largestegv = egvs[1]
        smallesttegv = egvs[2]
    else
        idxmax = 2
        largestegv = egvs[2]
        smallesttegv = egvs[1]
    end

    rx = quant * sqrt(largestegv)
    ry = quant * sqrt(smallesttegv)

    eigvecmax = eigvecs(Σ)[:, idxmax]
    θ = atan(eigvecmax[2] / eigvecmax[1])
    if θ < 0
        θ += 2 * π
    end

    return getellipsepoints(cx, cy, rx, ry, θ)
end

function plotAssocErrs(measurements::Vector{Vector{Measurement}})
    set_theme!(theme_black())
    fig = Figure(size=(800, 1000))
    ax1 = Axis(fig[1, 1])
    ax2 = Axis(fig[2, 1])
    ax3 = Axis(fig[3, 1])
    ax4 = Axis(fig[4, 1])
    ax4.xlabel = "Time (days)"
    ax1.ylabel = "SC1 Association Status"
    ax2.ylabel = "SC2 Association Status"
    ax3.ylabel = "SC3 Association Status"
    ax4.ylabel = "SC4 Association Status"
    time = [measurements[i][1].Time for i in eachindex(measurements)]
    AssocErrs = getRBMCDAAssocErrors(measurements)

    xplot = (1 / 86400) .* time[1:200]
    scatter!(ax1, xplot, AssocErrs[1, 1:200, 1], label="No Meas, UCT Associated")
    scatter!(ax1, xplot, AssocErrs[2, 1:200, 1], label="UCT Associated")
    scatter!(ax1, xplot, AssocErrs[3, 1:200, 1], label="Meas Tagged as UCT")
    scatter!(ax1, xplot, AssocErrs[4, 1:200, 1], label="Cross Tag")
    scatter!(ax1, xplot, AssocErrs[5, 1:200, 1], label="Measurement Correctly Associated")
    scatter!(ax2, xplot, AssocErrs[1, 1:200, 2])
    scatter!(ax2, xplot, AssocErrs[2, 1:200, 2])
    scatter!(ax2, xplot, AssocErrs[3, 1:200, 2])
    scatter!(ax2, xplot, AssocErrs[4, 1:200, 2])
    scatter!(ax2, xplot, AssocErrs[5, 1:200, 2])
    scatter!(ax3, xplot, AssocErrs[1, 1:200, 3])
    scatter!(ax3, xplot, AssocErrs[2, 1:200, 3])
    scatter!(ax3, xplot, AssocErrs[3, 1:200, 3])
    scatter!(ax3, xplot, AssocErrs[4, 1:200, 3])
    scatter!(ax3, xplot, AssocErrs[5, 1:200, 3])
    scatter!(ax4, xplot, AssocErrs[1, 1:200, 4])
    scatter!(ax4, xplot, AssocErrs[2, 1:200, 4])
    scatter!(ax4, xplot, AssocErrs[3, 1:200, 4])
    scatter!(ax4, xplot, AssocErrs[4, 1:200, 4])
    scatter!(ax4, xplot, AssocErrs[5, 1:200, 4])
    axislegend(ax1; position=:rb)
    display(GLMakie.Screen(), fig)

end