using Colors
using DrWatson
@quickactivate "Heisenberg"


#adding vectors
Base.:+(a::Tuple{Float64, Float64, Float64}, b::Vector{Float64}) = (a[1]+b[1],a[2]+b[2],a[3]+b[3])
Base.:+(a::Tuple{Float64, Float64, Float64}, b::Tuple{Float64, Float64, Float64}) = (a[1]+b[1],a[2]+b[2],a[3]+b[3])
Base.:+(a::Tuple{Float64, Float64, Float64}, b::Tuple{Int64, Int64, Int64}) = (a[1]+b[1],a[2]+b[2],a[3]+b[3])

Base.:-(a::Tuple{Float64, Float64, Float64}, b::Tuple{Int64, Int64, Int64}) = (a[1]-b[1],a[2]-b[2],a[3]-b[3])
Base.:-(a::Tuple{Float64, Float64, Float64}, b::Tuple{Float64, Float64, Float64}) = (a[1]-b[1],a[2]-b[2],a[3]-b[3])


# real scalar product
Base.:*(a::Tuple{Float64, Float64, Float64},b::Tuple{Float64, Float64, Float64}) = a[1]*b[1]+a[2]*b[2]+a[3]*b[3]
Base.:*(a::Vector{Float64},b::Tuple{Float64, Float64, Float64}) = a[1]*b[1]+a[2]*b[2]+a[3]*b[3]
Base.:*(a::Tuple{Float64, Float64, Float64},b::Vector{Float64}) = a[1]*b[1]+a[2]*b[2]+a[3]*b[3]

# scaling a vector
Base.:*(a::Int64,b::Tuple{Float64, Float64, Float64}) = (a*b[1],a*b[2],a*b[3])
Base.:*(a::Float64,b::Tuple{Float64, Float64, Float64}) = (a*b[1],a*b[2],a*b[3])

Base.:/(a::Tuple{Float64, Float64, Float64},b::Int64) = (a[1]/b,a[2]/b,a[3]/b)

Base.:abs(a::Tuple{Float64, Float64, Float64}) = (abs(a[1]),abs(a[2]),abs(a[3]))


abstract type AbstractGrid end
#and this sadly doesnt work for an n-dimensional case
# carefull touching this will need a restart
mutable struct Grid <: AbstractGrid
    dimension::Vector{Int64} 
    J_coup::Float16
    H_field::Tuple{Float64, Float64, Float64}
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


function normalise_spin(input::Vector{Float64})
    x,y,z = input
    norm = sqrt(x^2 + y^2 + z^2)
    return (x/norm,y/norm,z/norm)
end

function normalise_spin(input::Tuple{Float64, Float64, Float64})
    x,y,z = input
    norm = sqrt(x^2 + y^2 + z^2)
    return (x/norm,y/norm,z/norm)
end


"""
 Calculates the Magnetisation for a given Grid 

"""
function Magnetisation(A::Array{Tuple{Float64, Float64, Float64}})
    x_s,y_s,z_s = 0,0,0
    for  cord in CartesianIndices(A)
        x,y,z = A[Tuple(cord)...]     
        x_s += x
        y_s += y
        z_s += z
    end

    return x_s,y_s,z_s
end

"""
 Creates an a grid for the given tuple of dimensions N. 
    This can do more than 3 dimensions. 

"""
function random_grid(N)
    Tensor_state =  Array{Tuple{Float64, Float64, Float64}}(undef,  N...)

    for  cord in CartesianIndices(Tensor_state)
        Tensor_state[Tuple(cord)...] = normalise_spin((rand(-100:100)/10.0,rand(-100:100)/10.0,rand(-100:100)/10.0))      
              
    end
    
    return Tensor_state
end


"""
  This is a a very slow version of determining all the nearest nearest neighbors
   for a given cordinate in a matrix defined by the axes. Whilst simulating periodic boundary conditions.
"""
function returnNN(axes, cord)
    
    NNlist = Array{Array{}}(undef, length(axes) * 2)
    
    #go through each dimension with index ind and total size ax
    for (ind, ax) in enumerate(axes)
        # test for each dimension if the upper and lower neighbors are in the tensor
        for (chI, change) in enumerate([-1, 1])

            safe_index = (ind - 1) * 2 + chI

            if (cord[ind] + change <= 0)
                NNlist[safe_index] = copy(cord)
                NNlist[safe_index][ind] = ax #maybe pm 1
            elseif (cord[ind] + change > ax)
                NNlist[safe_index] = copy(cord)
                NNlist[safe_index][ind] = 1 #maybe pm 1
            else
                NNlist[safe_index] = copy(cord)
                NNlist[safe_index][ind] = cord[ind] + change
            end
        end
    end
    return NNlist
end


"""
  This calculates/applies the hamiltonian for the complete grid/matrix
"""
function Heisenberg_Hamilt(A::Array{Tuple{Float64, Float64, Float64}}, J::Float64, H::Tuple{Float64, Float64, Float64})
    Ha = 0
    for cords in CartesianIndices(A)
        cord = Tuple(cords)
        NNlist = returnNN(size(A), collect(cord))

        for neigh in NNlist
            Ha += (-J * 0.5 * (A[cord...] * A[neigh...]))
        end
        #Ha += -H * A[cord...]
    end
    return Ha
end

"""
  This calculates/applies the hamiltonian for the complete grid/matrix
"""
function Heisenberg_Hamilt(A::Array{Tuple{Float64, Float64, Float64}}, J::Float16, H::Tuple{Float64, Float64, Float64})
    Ha = 0
    for cords in CartesianIndices(A)
        cord = Tuple(cords)
        NNlist = returnNN(size(A), collect(cord))

        for neigh in NNlist
            Ha += (-J * 0.5 * (A[cord...] * A[neigh...]))
        end
        #Ha += -H * A[cord...]
    end
    return Ha
end

"""
  Calculates the difference in Energy for a given flip and the new angle
"""
function Heisenberg_Delta(grid::AbstractGrid,pos::Vector{Int64},New_spin::Tuple{Float64, Float64, Float64})
    current = grid.matrix[pos...]
    n = length(grid.dimension)
    N = grid.dimension
    Ha = 0
    
    for i in 1:n
        
        nnm,nnp = copy(pos),copy(pos)
        nnm[i] = mod1(nnm[i]-1,N[i]) 
        nnp[i] = mod1(nnp[i]+1,N[i]) 
        
        Ha -=   grid.J_coup  * ( (-1* current + New_spin) * grid.matrix[nnp...])
        Ha -=   grid.J_coup  * ( (-1* current + New_spin) * grid.matrix[nnm...])
    end
    Ha += 2 * (grid.H_field *(-1* current + New_spin)) 
    return Ha
end

"""
  Calculates the difference in Magnetisation(call before doing the flip....)
"""
function Magnetisation_Delta(grid::AbstractGrid,pos::Vector{Int64},New_spin::Tuple{Float64, Float64, Float64})
    current = grid.matrix[pos...]
    return (-1*current+New_spin)
end

function Thermalisation(grid::AbstractGrid,T::Float64,Steps::Int64)
    for _ in 1:Steps
        flip = Array{Int}(undef, length(grid.dimension))

        for ind in 1:length(grid.dimension)
            flip[ind] = rand(1:grid.dimension[ind])                  # chose a random spin to flip  
        end

        proposed_spin = normalise_spin((rand(-100:100)/10.0,rand(-100:100)/10.0,rand(-100:100)/10.0))
        
        Delta_energy = Heisenberg_Delta(grid,flip,proposed_spin)

        metroplis_P  = exp(-Delta_energy/(T))
        Glauber_P  = metroplis_P/(1+metroplis_P)
        
        #if (rand()< exp(-Delta_energy/T))
        if (Delta_energy< 0) || (rand()< Glauber_P)
         #   if (Delta_energy< 0) || (rand()< exp(-Delta_energy/T))
            grid.matrix[flip...] = Tuple(proposed_spin)        
        end
        
    end
    #update_EM(grid)
end


function Arrow(xb,yb,xt,yt)
    l1 = Line(xb,yb,xt,yt)
    dx = (xb-xt)
    dy = (yb-yt)
    Norm  = sqrt(dx^2+dy^2)
    udx =   dx/Norm
    udy = dy/Norm

    ax = udx * sqrt(3)/2 - udy * 1/2

    ay = udx * 1/2 + udy * sqrt(3)/2

    bx = udx * sqrt(3)/2 + udy * 1/2

    by =  - udx * 1/2 + udy * sqrt(3)/2

    l2 = Line(xt,yt,trunc(Int, xt + 20 * ax),trunc(Int, yt + 20 * ay))
    l3 = Line(xt,yt,trunc(Int, xt + 20 * bx),trunc(Int, yt + 20 * by))
   
    return [l1,l2,l3]	
end


function Arrow_Dir(xb,yb,Dx,Dy,c)
    #xt = trunc(Int, xb+Dx) 
    xt =  xb+Dx
    #yt = trunc(Int, yb+Dy) 
    yt = yb+Dy
    l1 = LineF(xb,yb,xt,yt,c)
    dx = (xb-xt)
    dy = (yb-yt)
    Norm  = sqrt(dx^2+dy^2)
    if Norm == 0
         Norm = 1
    end
    
    udx =   dx/Norm
    udy = dy/Norm

    ax = udx * sqrt(3)/2 - udy * 1/2

    ay = udx * 1/2 + udy * sqrt(3)/2

    bx = udx * sqrt(3)/2 + udy * 1/2

    by =  - udx * 1/2 + udy * sqrt(3)/2

    l2 = LineF(xt,yt, xt + 20 * ax, yt + 20 * ay,c)
    l3 = LineF(xt,yt, xt + 20 * bx, yt + 20 * by,c)
   
    return [l1,l2,l3]	
end


############################################################################################################

#Plot parameters
HEIGHT = 150
WIDTH = 650
BACKGROUND = colorant"antiquewhite"

#Simulation parameters
N = [5,1]                   # Amount of Spins
J = 100.01                    # Coupling of the Spins
H = (0.0,1500.0,0.0)        # Field

T = 300.0                   # Temperature
Steps_per_Frame = 1         # Amount of Simualtions steps per Frame


twoD_grid = Grid(N,J,H)     # create a Grid with the given Parameters



#Inital Data preperations
A = twoD_grid.matrix        

As = []

for cord in CartesianIndices(A)
    c =  (A[Tuple(cord)...][3] +1)*0.5
    push!(As,Arrow_Dir(cord[1]*900+10,cord[2]*1800+10,A[Tuple(cord)...][1]*40,A[Tuple(cord)...][2]*40,c))
    
end

norm = sqrt(H[1]^2 + H[2]^2 +H[3]) * 0.02
#push!(As,Arrow_Dir(600,100,-H[1]/norm,-H[2]/norm,-H[3]/norm))
push!(As,Arrow_Dir(600,100,-H[1]/norm,-H[2]/norm,-1))

#Inital Plotting
function draw(g::Game) 
    for A in As
        for a in A
            draw(a)
        end
    end
end

#Update Plotting
function update(g::Game)
    Thermalisation(twoD_grid,T,Steps_per_Frame)
    
    An = twoD_grid.matrix    
    for (ind,cord) in enumerate(CartesianIndices(An))
        c = (A[Tuple(cord)...][3] +1)*0.5
        As[ind]= Arrow_Dir(cord[1]*90+7,cord[2]*90+5,An[Tuple(cord)...][1]*50,An[Tuple(cord)...][2]*50,c)
    end
   

end