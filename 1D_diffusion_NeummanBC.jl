"""
This program solves the 1D diffusion equation using the FVM with cell centered grid using central differencing for face values.

Note: This implementation involves the use of Neumman and Dirichlet boundary conditions and also computes errors.
"""

using Revise
using Plots
using Statistics

# Domain definitions
L = 5 # m (bar length)
Ax = 0.1 # m² (cross-sectional area)

# Discretization
Nx = 5 # Number of cells
dx = L / Nx
Xp = LinRange(dx/2, L-dx/2, Nx) # Cell center positions

# Defining B.C.
S̄ = 1000 # W/m³
T_A = 100 # °C
q_A = 100 # W/m²
T_B = 200 # °C

# Define fluid properties
ρ = 1.0 # kg/m³
cp = 1000 # J/kg.K
k = 100 # W/mK

# Define discretization properties
D = k / dx # Difusive Flux
V = dx * Ax # Cell volume

# Initialize variables
T = zeros(Float32, Nx) # Temperature at cell centers
A = zeros(Float32, (Nx, Nx)) # Area at cell centers
b = zeros(Float32, Nx) # Source term

# Defining cell struct
mutable struct Cell
    type::String # BC or Internal
    BC_type::String # L_BC_D, L_BC_N, R_BC_D, R_BC_N
    BC_value::Float32
    source_value::Float32
end

# Initialize cells
cells = [Cell("Internal", "", 0.0, 0.0) for _ in 1:Nx]

cells[1].type = "L_BC" # B.C. type at A
cells[1].BC_type = "Neumann" # B.C. type at A
cells[1].BC_value = q_A # B.C. value at A

cells[Nx].type = "R_BC" # B.C. type at B
cells[Nx].BC_type = "Dirichlet" # B.C. type at B
cells[Nx].BC_value = T_B # B.C. value at A

cells .= map(c -> (c.source_value = S̄; c), cells) # Assing source in all cells

function main()

    # Assemble Matrix
    for i in 1:Nx
        if cells[i].type == "Internal"
            # Matrix Parameters
            aₗ = D * Ax  
            aᵣ = D * Ax 
            Sₚ = 0
            aₚ = aₗ + aᵣ - Sₚ
            Sᵤ = S̄ * V
        
            # Assemble Matrix
            A[i, i] = aₚ
            A[i, i-1] = -aₗ
            A[i, i+1] = -aᵣ

            b[i] = Sᵤ

        elseif cells[i].type == "L_BC"
            # Matrix Parameters
            aₗ = 0
            aᵣ = D * Ax 
            if cells[i].BC_type == "Dirichlet"
                Sₚ = -(2 * D * Ax) 
                Sᵤ = cells[i].BC_value * (2 * D * Ax) + S̄ * V

            elseif cells[i].BC_type == "Neumann"
                Sₚ = 0
                Sᵤ = -cells[i].BC_value * Ax + S̄ * V
            end
            
            # Assemble Matrix
            aₚ = aₗ + aᵣ - Sₚ
            A[i, i] = aₚ
            A[i, i+1] = -aᵣ

            b[i] = Sᵤ

        elseif cells[i].type == "R_BC"
            # Matrix Parameters
            aₗ = D * Ax
            aᵣ = 0
            if cells[i].BC_type == "Dirichlet"
                Sₚ = -(2 * D * Ax) 
                Sᵤ = cells[i].BC_value * (2 * D * Ax) + S̄ * V

            elseif cells[i].BC_type == "Neumann"
                Sₚ = 0
                Sᵤ = -cells[i].BC_value * Ax + S̄ * V
            end
            
            # Assemble Matrix
            aₚ = aₗ + aᵣ - Sₚ
            A[i, i] = aₚ
            A[i, i-1] = -aₗ

            b[i] = Sᵤ

        end

    end

    # Solve system
    T = A \ b

    # Compute neumann boundary values
    T_A = T[1] + q_A * Ax / (2 * D * Ax * (-1))

    # Compute error
    error = zeros(Float32, Nx)
    for i in 1:Nx
        if cells[i].type == "Internal"
            Qₗ = -k * Ax * (T[i] - T[i-1]) / dx * (-1)
            Qᵣ = -k * Ax * (T[i+1] - T[i]) / dx * (1)
        elseif cells[i].type == "L_BC"
            Qₗ = -k * Ax * (T[i] - T_A) / (dx/2) * (-1)
            Qᵣ = -k * Ax * (T[i+1] - T[i]) / dx * (1)
        elseif cells[i].type == "R_BC"
            Qₗ = -k * Ax * (T[i] - T[i-1]) / dx * (-1)
            Qᵣ = -k * Ax * (T_B - T[i]) / (dx/2) * (1)
        end
        error[i] = cells[i].source_value * V - Qᵣ - Qₗ
    end

    rms = sqrt(mean(error .^ 2))

    println("RMS Error: ", rms)

    println("Error")
    display(error)
    println("Matrix A")
    display(A)
    println("Vector b")
    display(b)
    println("Vector T")
    display(T)

    plot(Xp, T, label="Temperature", xlabel="Position [m]", ylabel="Temperature [°C]", 
        legend=:topleft , marker=:circle)

end

@time main()
