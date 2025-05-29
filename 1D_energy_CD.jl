"""
This program solves the 1D convection-diffusion equation using the FVM with cell centered grid using central differencing for face values.

Note: Because it is used central differencing, the solution may be unstable for high Peclet numbers.
"""

using Plots

# Domain definitions
L = 5 # m (bar length)
Ai = 0.1 # m² (inlet cross-sectional area)
Ao = 0.1 # m² (outlet cross-sectional area)

# Discretization
Nx = 5 # Number of cells
dx = L / Nx
Xp = LinRange(dx/2, L-dx/2, Nx) # Cell center positions
Ax = LinRange(Ai, Ao, Nx) # Cell cross-sectional areas

# Boundary Conditions
u_inlet = 0 # m/s
T_inlet = 100 # °C
u_outlet = u_inlet # °C
T_outlet = 200 # °C
S̄ = 1000 # W/m³

# Define fluid properties
ρ = 1.0 # kg/m³
cp = 1000 # J/kg.K
k = 100 # W/mK

# Define discretization properties
D = k / dx # Difusive Flux

# Initialize variables
u = ones(Float32, Nx) * u_inlet # Velocity at cell faces
T = zeros(Float32, Nx) # Temperature at cell centers
A = zeros(Float32, (Nx, Nx)) # Area at cell centers
b = zeros(Float32, Nx) # Source term

function main()

    # Assemble Matrix
    for i in 1:Nx

        # computing cell values
        V = dx * Ax[i] # Cell volume

        if i != 1 && i != Nx
            # computing faces values
            Aₗ = (Ax[i-1] + Ax[i]) / 2 # Left face area
            uₗ = (u[i-1] + u[i]) / 2 # Left face velocity
            Aᵣ = (Ax[i] + Ax[i+1]) / 2 # Right face area
            uᵣ = (u[i] + u[i+1]) / 2 # Right face velocity

            Fₗ = ρ * cp * uₗ * Aₗ # Left face convective flux 
            Fᵣ = ρ * cp * uᵣ * Aᵣ # Right face convective flux

            # Matrix Parameters
            aₗ = D * Aₗ + Fₗ / 2
            aᵣ = D * Aᵣ - Fᵣ / 2 
            Sₚ = 0
            aₚ = aₗ + aᵣ + (Fᵣ - Fₗ) - Sₚ
            Sᵤ = S̄ * V
        
            # Assemble Matrix
            A[i, i] = aₚ
            A[i, i-1] = -aₗ
            A[i, i+1] = -aᵣ

            b[i] = Sᵤ

        elseif i == 1
            # computing faces values
            Aₗ = Ax[i] # Left face area
            uₗ = u_inlet # Left face velocity
            Aᵣ = (Ax[i] + Ax[i+1]) / 2 # Right face area
            uᵣ = (u[i] + u[i+1]) / 2 # Right face velocity

            Fₗ = ρ * cp * uₗ * Aₗ # Left face convective flux 
            Fᵣ = ρ * cp * uᵣ * Aᵣ # Right face convective flux

            # Matrix Parameters
            aₗ = 0
            aᵣ = D * Aᵣ - Fᵣ / 2 
            Sₚ = -(2 * D * Aₗ + Fₗ) 
            aₚ = aₗ + aᵣ + (Fᵣ - Fₗ) - Sₚ
            Sᵤ = T_inlet * (2 * D * Aₗ + Fₗ) + S̄ * V

            # Assemble Matrix
            A[i, i] = aₚ
            A[i, i+1] = -aᵣ

            b[i] = Sᵤ

        else i == Nx
            # computing faces values
            Aₗ = (Ax[i-1] + Ax[i]) / 2 # Left face area
            uₗ = (u[i-1] + u[i]) / 2 # Left face velocity
            Aᵣ = Ax[i] # Right face area
            uᵣ = u_outlet # Right face velocity

            Fₗ = ρ * cp * uₗ * Aₗ # Left face convective flux 
            Fᵣ = ρ * cp * uᵣ * Aᵣ # Right face convective flux

            # Matrix Parameters
            aₗ = D * Aₗ + Fₗ / 2
            aᵣ = 0
            Sₚ = -(2 * D * Aᵣ - Fᵣ)
            aₚ = aₗ + aᵣ + (Fᵣ - Fₗ) - Sₚ
            Sᵤ = T_outlet * (2 * D * Aᵣ - Fᵣ) + S̄ * V

            # Assemble Matrix
            A[i, i] = aₚ
            A[i, i-1] = -aₗ

            b[i] = Sᵤ

        end

    end

    # Solve system
    T = A \ b

    display(A)
    display(b)
    display(T)

    plot(Xp, T, label="Temperature", xlabel="Position [m]", ylabel="Temperature [°C]", 
        legend=:topleft , marker=:circle)

end

@time main()