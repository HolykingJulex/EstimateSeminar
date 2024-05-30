using DrWatson
@quickactivate "Julian-Beisch"

function Metropolis_Simulation(grid::AbstractGrid)

    N_thermal = 1000 * grid.dimension[1]^3
    N_sample = 500
    N_subsweep = 3 * grid.dimension[1]^3

    Temp_step::Int16 = 20

    M_data = Array{Tuple{Float64, Float64, Float64}}(undef,  N_sample)
    E_data = zeros(N_sample)
    B_data_1 = zeros(N_sample)
    B_data_2 = zeros(N_sample)

    #M_mean = Array{Tuple{Float64, Float64, Float64}}(undef,  Temp_step)
    M_mean = zeros(Temp_step)
    E_mean = zeros(Temp_step)
    Temps = zeros(Temp_step)
    Binder = zeros(Temp_step)
    Sus = zeros(Temp_step)
    Ts = range(0.4, 4, length=Temp_step)

    for (index,T) in enumerate(Ts)
        #T = TH/2
        Thermalisation(grid,T,N_thermal)

        for ind in (1:N_sample)
            Subsweep(grid,T,N_subsweep)
            
            #E_data[1] = 
            M_data[ind] = abs.(Magnetisation(grid.matrix))
            E_data[ind] = Heisenberg_Hamilt(grid.matrix,grid.J_coup,grid.H_field)
            B_data_1[ind] = M_data[ind][1]^4 + M_data[ind][2]^4 + M_data[ind][3]^4
            B_data_2[ind] = M_data[ind][1]^2 + M_data[ind][2]^2 + M_data[ind][3]^2
        end
        M_mean[index]= mean(mean(M_data))
        E_mean[index]= mean(E_data)
        Temps[index]=T
        Sus[index]=std(E_data)
        Binder[index] = 1 - mean(B_data_1)/(3*mean(B_data_2)^2)
    end
    return Temps, M_mean, E_mean, Binder, Sus
end

function Thermalisation(grid::AbstractGrid,T::Float64,Steps::Int64)
    for _ in 1:Steps
        flip = Array{Int}(undef, length(grid.dimension))

        for ind in 1:length(grid.dimension)
            flip[ind] = rand(1:grid.dimension[ind])                  # chose a random spin to flip  
        end

        proposed_spin = normalise_spin(rand([-1.0, 1.0], 3))
        
        Delta_energy = Heisenberg_Delta(grid,flip,proposed_spin)
        
        if (Delta_energy< 0) || (rand()< exp(-Delta_energy/T))
            grid.matrix[flip...] = Tuple(proposed_spin)        
        end
        
    end
    update_EM(grid)
end

function Subsweep(grid::AbstractGrid,T::Float64,Steps::Int64)

    for _ in 1:Steps
        flip = Array{Int}(undef, length(grid.dimension))

        for ind in 1:length(grid.dimension)
            flip[ind] = rand(1:grid.dimension[ind])                  # chose a random spin to flip  
        end
    
        proposed_spin = normalise_spin(rand([-1.0, 1.0], 3))
        
        Delta_energy = Heisenberg_Delta(grid,flip,proposed_spin)
        
        if (Delta_energy< 0) || (rand()< exp(-Delta_energy/T))
            Delta_M = Magnetisation_Delta(grid,flip,proposed_spin)
    
            grid.matrix[flip...] = Tuple(proposed_spin)
            
            grid.M -= Delta_M 
            grid.E -= Delta_energy
       
        end 
    end
end

# function Metropolis_Step(grid::AbstractGrid,T::Float64)
#     flip = Array{Int}(undef, length(grid.dimension))

#     for ind in 1:length(grid.dimension)
#         flip[ind] = rand(1:grid.dimension[ind])                  # chose a random spin to flip  
#     end

#     proposed_spin = normalise_spin(rand([-1.0, 1.0], 3))
    
#     Delta_energy = Heisenberg_Delta(grid,flip,proposed_spin)

    

#     #if (Delta_energy< 0) || (rand()< exp(-Delta_energy/T))
#     if (rand()< exp(-Delta_energy/T))
#         Delta_M = Magnetisation_Delta(grid,flip,proposed_spin)

#         grid.matrix[flip...] = Tuple(proposed_spin)
        
#         #grid.M += Delta_M 
#         #grid.E -= Delta_energy
        
#     else
#         Delta_M = (0,0,0)
#         Delta_energy = 0.0
#     end

#     return Delta_M, Delta_energy
    
# end

