using DrWatson
@quickactivate "Heisenberg"


abstract type AbstractGrid end
#and this sadly doesnt work for an n-dimensional case
# carefull touching this will need a restart
mutable struct Grid <: AbstractGrid
    dimension::Vector{Int64} 
    J_coup::Float16
    H_field::Float16
    matrix::Array{Tuple{Float64, Float64, Float64}}
    E::Float64
    M::Tuple{Float64, Float64, Float64}
end


function Grid(N,J,H)
    dimension = N
    J_coup = J 
    H_field = H 
    matrix = random_grid(N)
    E = Heisenberg_Hamilt(matrix,J,H)
    M = Magnetisation(matrix)
    
    return Grid(dimension,J_coup,H_field,matrix,E,M)
end

function update_EM(c::AbstractGrid)
    c.E  = Heisenberg_Hamilt(c.matrix,c.J_coup,c.H_field)
    c.M  = Magnetisation(c.matrix)
end


function assertCorrectness(c::AbstractGrid)
    @assert isapprox(c.E , Hamilt(c.matrix,c.J_coup,c.H_field)) "The Energy diverged $(c.E ) and $(Hamilt(c.matrix,c.J_coup,c.H_field))"
    @assert isapprox(abs(c.M), Magnetisation(c.matrix)) "The Magnetisation diverged $(c.M) and $(Magnetisation(c.matrix))"
end

Base.show(io::IO, person::AbstractGrid) = println(io, "Dimension: $(person.dimension) J_coup: $(person.J_coup) H_field: $(person.H_field) Energy: $(person.E) Mag: $(person.M)")