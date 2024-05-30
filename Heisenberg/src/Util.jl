using DrWatson
@quickactivate "Heisenberg"

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
        Tensor_state[Tuple(cord)...] = normalise_spin(rand([-1.0, 1.0], 3))      
              
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
function Heisenberg_Hamilt(A::Array{Tuple{Float64, Float64, Float64}}, J::Float64, H::Float64)
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
function Heisenberg_Hamilt(A::Array{Tuple{Float64, Float64, Float64}}, J::Float16, H::Float16)
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
    return Ha
end

"""
  Calculates the difference in Magnetisation(call before doing the flip....)
"""
function Magnetisation_Delta(grid::AbstractGrid,pos::Vector{Int64},New_spin::Tuple{Float64, Float64, Float64})
    current = grid.matrix[pos...]
    return (-1*current+New_spin)
end