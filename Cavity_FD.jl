#=
Solve the Navier-Stokes in a cavity flow problem using finite differences.
=#
using GLMakie
using Printf

function main()

println("Initializing cavity flow simulation...")

# define cavity gemetry and B.C.
cavity_h = 1 # height
cavity_w = 1 # width
c = 1 # lid velocity

# Define fluid properties
ρ = 1
ν = 0.01

# Define discretization properties
nx = 41
ny = 41
nit = 500
Δt = 1e-2

# Initializing mesh scheme
Δx = cavity_w / (nx - 1)
Δy = cavity_h / (ny - 1)

# calculate the pressure field
function calc_p(p, u, v)
    for it in 1:nit
        local pn = copy(p)  # Save the previous pressure field
        
        for i in 2:nx-1
            for j in 2:ny-1
                p[j, i] = ((pn[j, i+1] + pn[j, i-1]) * Δy^2 +
                           (pn[j+1, i] + pn[j-1, i]) * Δx^2) /
                          (2 * (Δx^2 + Δy^2)) -
                          ρ * Δx^2 * Δy^2 / (2 * (Δx^2 + Δy^2)) *
                          (1/Δt * ((u[j, i+1] - u[j, i-1]) / (2 * Δx) +
                                   (v[j+1, i] - v[j-1, i]) / (2 * Δy)) -
                          ((u[j, i+1] - u[j, i-1]) / (2 * Δx))^2 -
                          2 * ((u[j+1, i] - u[j-1, i]) / (2 * Δy) * 
                               (v[j, i+1] - v[j, i-1]) / (2 * Δx)) -
                          ((v[j+1, i] - v[j-1, i]) / (2 * Δy))^2)
            end
        end
        
        # Enforce correct boundary conditions for pressure
        p[:, end] = p[:, end-1]   # dp/dx = 0 at x = 2
        p[1, :] = p[2, :]         # dp/dy = 0 at y = 0
        p[:, 1] = p[:, 2]         # dp/dx = 0 at x = 0
        p[end, :] .= 0            # p = 0 at y = 2
    end
    return p
end

# calculate the u velocity field
function calc_u(p, un, vn)

    u_new = zeros(size(un))

    for i in 2:nx-1
        for j in 2:ny-1
            u_new[j, i] = un[j, i] - 
                           un[j, i] * Δt / Δx * (un[j, i] - un[j, i-1]) -
                           vn[j, i] * Δt / Δy * (un[j, i] - un[j-1, i]) - 
                           Δt / (ρ * 2 * Δx) * (p[j, i+1] - p[j, i-1]) + 
                           ν * (Δt / Δx^2 * (un[j, i+1] - 2 * un[j, i] + un[j, i-1]) +
                                Δt / Δy^2 * (un[j+1, i] - 2 * un[j, i] + un[j-1, i]))
        end
    end
    return u_new
end

# calculate the v velocity field
function calc_v(p, un, vn)

    v_new = zeros(size(vn))

    for i in 2:nx-1
        for j in 2:ny-1
            v_new[j, i] = vn[j, i] - 
                           un[j, i] * Δt / Δx * (vn[j, i] - vn[j, i-1]) -
                           vn[j, i] * Δt / Δy * (vn[j, i] - vn[j-1, i]) - 
                           Δt / (2 * Δy * ρ) * (p[j+1, i] - p[j-1, i]) + 
                           ν * (Δt / Δx^2 * (vn[j, i+1] - 2 * vn[j, i] + vn[j, i-1]) +
                                Δt / Δy^2 * (vn[j+1, i] - 2 * vn[j, i] + vn[j-1, i]))
        end
    end
    return v_new
end

# Solve cavity flow
function solve_cavity(e_conv=1e-6, max_iter=100, min_iter=10)

    i_time = time()

    # Initializing properties
    u = zeros((ny, nx))
    v = zeros((ny, nx))
    p = zeros((ny, nx)) 

    # initializing residuals
    δu = ones(max_iter)*Inf
    δv = ones(max_iter)*Inf
    δp = ones(max_iter)*Inf

    iter = 1
    # Solve for u, v, and p 
    while (iter < min_iter || (δu[iter] > e_conv || δv[iter] > e_conv || δp[iter] > e_conv)) && iter < max_iter
        
        pn = copy(p)
        un = copy(u)
        vn = copy(v)

        p = calc_p(p, u, v)
        u = calc_u(p, un, vn)
        v = calc_v(p, un, vn)
        
        # Enforce boundary conditions
        u[1, :] .= 0
        u[:, 1] .= 0
        u[:, end] .= 0
        u[end, :] .= c  # Set velocity on cavity lid equal to c

        v[1, :]  .= 0
        v[end, :] .= 0
        v[:, 1]  .= 0
        v[:, end] .= 0
        
        # Calculate residuals
        δp[iter] = abs((sum(p) - sum(pn)) / sum(p))
        δu[iter] = abs((sum(u) - sum(un)) / sum(u))
        δv[iter] = abs((sum(v) - sum(vn)) / sum(v))

        # print residuals
        if (iter % 5 == 0)
            # Calculate elapsed time
            elapsed_time = time() - i_time
            @printf "Iteration: %d - δu: %.3e - δv: %.3e - δp: %.3e - t: %.1f\n" iter δu[iter] δv[iter] δp[iter] elapsed_time
        end

        iter += 1

    end

    residuals = (δu, δv, δp)

    return p, u, v, residuals
end

# Plotting results
function plot_results(p, u, v, plot_residuals=true)
    println("\nPlotting results...")
    
    x = LinRange(0, cavity_w, nx)
    y = LinRange(0, cavity_h, ny)

    # Define the velocity field function
    function velocity_field(x, y)
        i = round(Int, (x / cavity_w) * (nx - 1)) + 1
        j = round(Int, (y / cavity_h) * (ny - 1)) + 1
        return Point2f(u[j, i], v[j, i])
    end

    fig = Figure(size = (800, 800))
    ax = Axis(fig[1, 1], title = "Cavity Flow", xlabel = "x", ylabel = "y")
    ax.aspect = cavity_w / cavity_h
    
    cf = contourf!(ax, x, y, p', colormap = :viridis)
    streamplot!(ax, velocity_field, 0..cavity_w, 0..cavity_h, alpha=0.5, density=2.5)
    Colorbar(fig[1, 2], cf, label = "Pressure")
    
    display(fig)

    # Plot residuals
    if plot_residuals
        fig = Figure()
        ax = Axis(fig[1, 1], title = "Residuals", xlabel = "Iterations", ylabel = "Residuals", yscale = log10)
        plot!(ax, residuals[1][1:end-1], label = "δu")
        plot!(ax, residuals[2][1:end-1], label = "δv")
        plot!(ax, residuals[3][1:end-1], label = "δp")
        Legend(fig[1, 2], ax, "Residuals", orientation = :vertical)
        display(GLMakie.Screen(), fig)
    end
end

# Calling functions
p, u, v, residuals = solve_cavity(1e-4, 1000, 10)

plot_results(p, u, v, true)

println("\nSimulation Finished.")

# Simulation infos
Re = cavity_h * c / ν
println("Re: ", Re)
println("n° of elements: ", nx*ny)

end

main()