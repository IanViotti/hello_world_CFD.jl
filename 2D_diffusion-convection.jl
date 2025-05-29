"""
This program solves the 1D diffusion equation using the FVM with cell centered grid using central differencing for face values.

Note: Neumann B.C. still not supported for convection computations
"""

using Revise
using Plots
using Statistics


# Domain definitions
Lx::Float32 = 4 # m - length in x
Ly::Float32 = 4 # m - length in y
t::Float32 = 0.1 # m - thickness

# Discretization
Nx::Int16 = 50 # Number of cells
Ny::Int16 = 50 # Number of cells
dx = Lx / Nx
dy = Ly / Ny
Xp = LinRange(dx/2, Lx-dx/2, Nx) # Cell center positions
Yp = LinRange(dy/2, Ly-dy/2, Ny) # Cell center positions

# Defining B.C.
S̄::Float32 = 1000 # W/m³
T_N::Float32 = 0 # °C at north
T_S::Float32 = 0 # °C at south
T_E::Float32 = 100 # °C at east
T_W::Float32 = 0 # °C at west

V_N::Float32 = 0 # m/s at north
V_S::Float32 = 0 # m/s at south
U_E::Float32 = 0 # m/s at east
U_W::Float32 = 0 # m/s at west

# Field velocities
V::Float32 =  # m/s
U::Float32 = 0 # m/s

# Define fluid properties
ρ::Float32 = 1.0 # kg/m³
cp::Float32 = 1000 # J/kg.K
k::Float32 = 100 # W/mK

# Define discretization properties
Dx = k / dx # Difusive Flux
Dy = k / dy # Difusive Flux
V = dx * dy * t # Cell volume
Ax = dy * t # Cell cross-sectional area in x
Ay = dx * t # Cell cross-sectional area in y

# Initialize variables
u = ones(Float32, Nx*Ny) * U # x velocity vector
v = ones(Float32, Nx*Ny) * V # y velocity vector
T = zeros(Float32, Nx*Ny) # Temperature vector 
A = zeros(Float32, (Nx*Ny, Nx*Ny)) # A matrix 
b = zeros(Float32, Nx*Ny) # Source vector

# Defining cell struct
mutable struct Cell
    N::String # Internal, Dirichlet, Neumann
    S::String # Internal, Dirichlet, Neumann
    E::String # Internal, Dirichlet, Neumann
    W::String # Internal, Dirichlet, Neumann
    BC_value::Tuple{Float32, Float32, Float32, Float32} # B.C. values for each face
    source_value::Float32 # Source value
end

# Initialize cells
cells = [Cell("Internal", "Internal", "Internal", "Internal", (0, 0, 0, 0), S̄) for _ in 1:Ny, _ in 1:Nx]

# Apply B.C.
# B.C at North
[cells[1, j] = Cell("Dirichlet", "Internal", "Internal", "Internal", (T_N, 0, 0, 0), S̄) for j in 1:Nx]
# B.C at South
[cells[end, j] = Cell("Internal", "Dirichlet", "Internal", "Internal", (0, T_S, 0, 0), S̄) for j in 1:Nx]
# B.C at East
[cells[j, end] = Cell("Internal", "Internal", "Dirichlet", "Internal", (0, 0, T_E, 0), S̄) for j in 1:Ny]
# B.C at West
[cells[j, 1] = Cell("Internal", "Internal", "Internal", "Dirichlet", (0, 0, 0, T_W), S̄) for j in 1:Ny]

# Corner cases
# B.C at North-East
cells[1, end] = Cell("Dirichlet", "Internal", "Dirichlet", "Internal", (T_N, 0, T_E, 0), S̄)
# B.C at North-West
cells[1, 1] = Cell("Dirichlet", "Internal", "Internal", "Dirichlet", (T_N, 0, 0, T_W), S̄)
# B.C at South-East
cells[end, end] = Cell("Internal", "Dirichlet", "Dirichlet", "Internal", (0, T_S, T_E, 0), S̄)
# B.C at South-West
cells[end, 1] = Cell("Internal", "Dirichlet", "Internal", "Dirichlet", (0, T_S, 0, T_W), S̄)

# stack cells
cells = vcat(permutedims(cells)...)

function main()
    # Assemble Matrix
    for i in 1:Nx*Ny
        # North (top) face
        if cells[i].N == "Internal"
            v_n = (v[i] + v[i-Nx]) / 2
            F_n = ρ * cp * v_n * Ay
            a_n = Dy * Ay + max(-F_n, 0)
            Sp_n = 0
            Su_n = 0
        elseif cells[i].N == "Dirichlet" 
            v_n = V_N
            F_n = ρ * cp * v_n * Ay
            a_n = 0
            Sp_n = -(2 * Dy * Ay + max(-F_n, 0))
            Su_n = cells[i].BC_value[1] * (2 * Dy * Ay + max(-F_n, 0))
        elseif cells[i].N == "Neumann"
            a_n = 0
            Sp_n = 0
            Su_n = -cells[i].BC_value[1] * Ay 
        end

        # South (bottom) face
        if cells[i].S == "Internal"
            v_s = (v[i] + v[i+Nx]) / 2
            F_s = ρ * cp * v_s * Ay
            a_s = Dy * Ay + max(F_s, 0)
            Sp_s = 0
            Su_s = 0
        elseif cells[i].S == "Dirichlet"
            v_s = V_S
            F_s = ρ * cp * v_s * Ay
            a_s = 0
            Sp_s = -(2 * Dy * Ay + max(F_s, 0))
            Su_s = cells[i].BC_value[2] * (2 * Dy * Ay + max(F_s, 0))
        elseif cells[i].N == "Neumann"
            a_s = 0
            Sp_s = 0
            Su_s = -cells[i].BC_value[2] * Ay 
        end

        # East (right) face
        if cells[i].E == "Internal"
            u_e = (u[i] + u[i+1]) / 2
            F_e = ρ * cp * u_e * Ax
            a_e = Dx * Ax + max(-F_e, 0)
            Sp_e = 0 
            Su_e = 0
        elseif cells[i].E == "Dirichlet" 
            u_e = U_E
            F_e = ρ * cp * u_e * Ay
            a_e = 0
            Sp_e = -(2 * Dx * Ax + max(-F_e, 0))
            Su_e = cells[i].BC_value[3] * (2 * Dx * Ax + max(-F_e, 0))
        elseif cells[i].N == "Neumann"
            a_e = 0
            Sp_e = 0
            Su_e = -cells[i].BC_value[3] * Ax
        end

        # West (left) face
        if cells[i].W == "Internal"
            u_w = (u[i] + u[i-1]) / 2
            F_w = ρ * cp * u_w * Ax
            a_w = Dx * Ax + max(F_w, 0)
            Sp_w = 0
            Su_w = 0
        elseif cells[i].W == "Dirichlet" 
            u_w = U_W
            F_w = ρ * cp * u_w * Ay
            a_w = 0
            Sp_w = -(2 * Dx * Ax + max(F_w, 0))
            Su_w = cells[i].BC_value[4] * (2 * Dx * Ax + max(F_w, 0))
        elseif cells[i].N == "Neumann"
            a_w = 0
            Sp_w = 0
            Su_w = -cells[i].BC_value[4] * Ax
        end

        # Matrix Parameters
        Sp = -(Sp_n + Sp_s + Sp_e + Sp_w)
        ap = a_n + a_s + a_e + a_w + (F_e - F_w + F_n - F_s) + Sp
        Su = Su_n + Su_s + Su_e + Su_w + cells[i].source_value * V

        # Aloccating matrix Parameters
        A[i, i] = ap
        if i > Nx
            A[i, i-Nx] = -a_n
        end
        if i <= Nx*Ny - Nx
            A[i, i+Nx] = -a_s
        end
        if i < Nx*Ny
            A[i, i+1] = -a_e
        end
        if i > 1
            A[i, i-1] = -a_w
        end

        b[i] = Su

    end

    # Solve system
    T = A \ b

    T = reshape(T, (Nx, Ny))

    #contour(Xp, Yp, T', label="Temperature", xlabel="X [m]", ylabel="Y [m]", title="Temperature Distribution", c=:viridis, fill=true, levels=20, yflip=true)
    heatmap(Xp, Yp, T', label="Temperature", xlabel="X [m]", ylabel="Y [m]", title="Temperature Distribution", c=:viridis, yflip=true, aspect_ratio=Lx/Ly)
end

@time main()