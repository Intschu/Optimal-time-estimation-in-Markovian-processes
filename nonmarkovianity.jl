using QuantumOptics
using LinearAlgebra
using SparseArrays
using IterativeSolvers
using Plots
using LaTeXStrings
using QuantumFCS
#include("FCS_functions.jl")
println("detecting non-markovianity")

function transition_time(escape_rate::Float64) # sample time to jump out of state with escape rate (escape_rate)
    return -log(rand())/escape_rate # Gillespie algorithm
end

function transition_position(jump_rates::Vector{Float64}) # sample next post-jump state
    # jump_rates[j] is a vector of rates from current state to state j
    r = rand()
    escape_rate = sum(jump_rates)
    next_pos = 1 # post-jump state
    s = jump_rates[next_pos]/escape_rate
    while s < r # Gellispie algorithm
        next_pos += 1
        s += jump_rates[next_pos]/escape_rate
    end
    return next_pos
end

function sample_times(transition_rates::Matrix{Float64}, visible_weights::Matrix{Int64}, init_pos::Int64, runs::Int64)
    dwell_time = 0.0 # waiting time of the first visible transition (visible dwell time)
    wait_time = 0.0 # wainting time of the second visible transition (visible waiting time)
    for j = 1:runs # ensemble average
        pos = init_pos #initial state init_pos
        counter = 0 # counter of visible transitions
        while counter == 0
            # transition_rates[next_pos, pos] is rate from state pos to next_pos
            dwell_time += transition_time(sum(transition_rates[:, pos]))
            next_pos = transition_position(transition_rates[:, pos])
            #if next_pos == pos
            #    println("error")
            #end
            counter += visible_weights[next_pos, pos] # visible_weights[next_pos, pos] is 1 if transition from pos to next_pos is visible (otherwise is 0)
            pos = next_pos
        end
        while counter == 1
            wait_time += transition_time(sum(transition_rates[:, pos]))
            next_pos = transition_position(transition_rates[:, pos])
            counter += visible_weights[next_pos, pos]
            pos = next_pos
        end
    end
    return [dwell_time/runs, wait_time/runs]
end



function ring_network(n::Int64, transition_rates::Matrix{Float64}, visible_weights::Matrix{Int64}, runs::Int64)
    # n is dimension of the system
    H = 0.0*transition(b, 1, 1) # zero Hamiltonian
    J = (sqrt.(vec(transition_rates))).*vec(transition_ops) # vectorized transition operators
    steady_state = steadystate.eigenvector(H, J) # steady state
    cm_1, cm_2 = fcscumulants_recursive(H, J, [sqrt(transition_rates[1, n])*transition(b, 1, n), sqrt(transition_rates[n, 1])*transition(b, n, 1)], 2, steady_state, [1, 1])#, iterative=:false)
    times = [0.0, 0.0] # visible mean dwell and waiting times
    times_markov = [0.0, 0.0] # general mean dwell and waiting times
    for pos = 1:n
        times .+= sample_times(transition_rates, visible_weights, pos, runs)*real(expect(transition(b, pos, pos), steady_state))
        times_markov .+= sample_times(transition_rates, ones(Int64, n, n) - Matrix(1I, n, n), pos, runs)*real(expect(transition(b, pos, pos), steady_state))
    end
    #return [times[1], times[2], times_markov[1], times_markov[2], current[1], current[2]]
    return [times[1]/times[2], (cm_1*cm_1*times[1])/cm_2, (cm_1*cm_1*times[2])/cm_2, times_markov[1]/times_markov[2], (cm_1*cm_1*times_markov[1])/cm_2, (cm_1*cm_1*times_markov[2])/cm_2]
end


n = 6 # dimension of system
m = 50
runs = 1000
b = NLevelBasis(n) # hilbert space
transition_ops = Matrix{typeof(transition(b, 2, 1))}(undef, n, n) # 
for j = 1:n
    for k = 1:n
        transition_ops[j, k] = transition(b, j, k)
    end
end


cur_axis_dwell = []
kur_axis_dwell = []
cur_axis_wait = []
kur_axis_wait = []

visible_weights = zeros(Int64, n, n)
visible_weights[2, 5] = 1; visible_weights[5, 2] = 1 # set visible transitions

transition_rates = zeros(Float64, n, n)

for k = 1:m
    delta = 5.0*rand()
    for pos = 1:n
        r = rand()
        transition_rates[mod(pos + 1 - 1, n) + 1, pos] = r
        transition_rates[pos, mod(pos + 1 - 1, n) + 1] = r*exp(-delta)
    end    
    ring_network_temp = ring_network(n, transition_rates, visible_weights, runs)
    if ring_network_temp[1] < 1.0
        append!(cur_axis_wait, ring_network_temp[2])
        append!(kur_axis_wait, ring_network_temp[3])
    else
        append!(cur_axis_dwell, ring_network_temp[2])
        append!(kur_axis_dwell, ring_network_temp[3])
    end
end

println(sample_times(transition_rates, visible_weights, 1, 10))


outfile_dwell = "C:\\Users\\Downloads\\ring_hmm_dwell.txt"
outfile_wait = "C:\\Users\\Downloads\\ring_hmm_wait.txt"

open(outfile_wait, "w") do f
    for j = 1:length(cur_axis_wait)
      println(f, string(cur_axis_wait[j])*" "*string(kur_axis_wait[j]))
    end
end

open(outfile_dwell, "w") do f
    for j = 1:length(cur_axis_dwell)
      println(f, string(cur_axis_dwell[j])*" "*string(kur_axis_dwell[j]))
    end
end

println("Done")

ring_network_data = [cur_axis_dwell, kur_axis_dwell, cur_axis_wait, kur_axis_wait]
scatter(ring_network_data[1], ring_network_data[2], xlabel = L"\mathcal{S}\tilde{\mathcal{T}}", ylabel = L"\mathcal{S}/\tilde{\mathcal{A}} ", label=L"\tilde{\mathcal{T}} > 1/\tilde{\mathcal{A}}")
scatter!(ring_network_data[3], ring_network_data[4],  label=L"\tilde{\mathcal{T}} < 1/\tilde{\mathcal{A}}")


#=

transition_rates[1, 7] = 160.0; transition_rates[1, 2] = 50.0; transition_rates[3, 2] = 500.0; transition_rates[2, 3] = 100.0; transition_rates[4, 3] = 1000.0; transition_rates[3, 4] = 100.0; transition_rates[5, 4] = 10000.0; transition_rates[4, 5] = 500.0; transition_rates[6, 5] = 5000.0; transition_rates[7, 6] = 5000.0; transition_rates[6, 7] = 10.0
for k = 1:m
    transition_rates[7, 1] = 2.7*rand()*1000.0
    transition_rates[2, 1] = 2.0*rand()*1000.0
    transition_rates[5, 6] = 0.01*1000.0*rand()
    ring_network_temp = dynein_network(n, transition_rates, visible_weights, runs)
    if ring_network_temp[1] < 1.0
        append!(cur_axis_wait, ring_network_temp[2])
        append!(kur_axis_wait, ring_network_temp[3])
    else
        append!(cur_axis_dwell, ring_network_temp[2])
        append!(kur_axis_dwell, ring_network_temp[3])
    end
end

ring_network_data = [cur_axis_dwell, kur_axis_dwell, cur_axis_wait, kur_axis_wait]
scatter(ring_network_data[1], ring_network_data[2], xlabel = L"\mathcal{S}\tilde{\mathcal{T}}", ylabel = L"\mathcal{S}/\tilde{\mathcal{A}} ", label=L"\tilde{\mathcal{T}} > 1/\tilde{\mathcal{A}}")
scatter!(ring_network_data[3], ring_network_data[4],  label=L"\tilde{\mathcal{T}} < 1/\tilde{\mathcal{A}}")

=#



#=
function dynein_network(n::Int64, transition_rates::Matrix{Float64}, visible_weights::Matrix{Int64}, runs::Int64)
    H = 0.0*transition(b, 1, 1) # zero Hamiltonian
    J = (sqrt.(vec(transition_rates))).*vec(transition_ops)
    steady_state = steadystate.eigenvector(H, J)
    current = fcscumulants_recursive(H, J, [sqrt(transition_rates[n-1, n])*transition(b, n-1, n), sqrt(transition_rates[n, n-1])*transition(b, n, n-1)], 2, steady_state, iterative=:false)
    times = [0.0, 0.0]
    times_markov = [0.0, 0.0]
    for pos = 1:n
        times .+= sample_times(transition_rates, visible_weights, pos, runs)*real(expect(transition(b, pos, pos), steady_state))
        times_markov .+= sample_times(transition_rates, ones(Int64, n, n) - Matrix(1I, n, n), pos, runs)*real(expect(transition(b, pos, pos), steady_state))
    end
    #return [times[1], times[2], times_markov[1], times_markov[2], current[1], current[2]]
    return [times[1]/times[2], (current[1]*current[1]*times[1])/current[2], (current[1]*current[1]*times[2])/current[2], times_markov[1]/times_markov[2], (current[1]*current[1]*times_markov[1])/current[2], (current[1]*current[1]*times_markov[2])/current[2]]
end

function kinesin_network(n::Int64, transition_rates::Matrix{Float64}, visible_weights::Matrix{Int64}, runs::Int64)
    H = 0.0*transition(b, 1, 1) # zero Hamiltonian
    J = (sqrt.(vec(transition_rates))).*vec(transition_ops)
    steady_state = steadystate.eigenvector(H, J)
    current = fcscumulants_recursive(H, J, [sqrt(transition_rates[2, 5])*transition(b, 2, 5), sqrt(transition_rates[5, 2])*transition(b, 5, 2)], 2, steady_state, iterative=:false)
    times = [0.0, 0.0]
    times_markov = [0.0, 0.0]
    for pos = 1:n
        times .+= sample_times(transition_rates, visible_weights, pos, runs)*real(expect(transition(b, pos, pos), steady_state))
        times_markov .+= sample_times(transition_rates, ones(Int64, n, n) - Matrix(1I, n, n), pos, runs)*real(expect(transition(b, pos, pos), steady_state))
    end
    #return [times[1], times[2], times_markov[1], times_markov[2], current[1], current[2]]
    return [times[1]/times[2], (current[1]*current[1]*times[1])/current[2], (current[1]*current[1]*times[2])/current[2], times_markov[1]/times_markov[2], (current[1]*current[1]*times_markov[1])/current[2], (current[1]*current[1]*times_markov[2])/current[2]]
end

=#



#=

transition_rates[1, 2] = 100.0; transition_rates[3, 2] = 100.0; transition_rates[6, 5] = 100.0; transition_rates[2, 5] = 0.24; transition_rates[5, 2] = 300000.0; transition_rates[4, 3] = 100.0; transition_rates[1, 6] = 100.0
for k = 1:m
    r = rand() # atp
    transition_rates[2, 1] = 2.0*r*1000.0
    transition_rates[5, 4] = 2.0*r*1000.0
    r = rand() # adp
    transition_rates[2, 3] = 0.02*r*1000.0
    transition_rates[5, 6] = 0.02*r*1000.0
    r = rand() # pi
    transition_rates[3, 4] = 0.02*r*1000.0
    transition_rates[6, 1] = 0.02*r*1000.0
    transition_rates[4, 5] = transition_rates[1, 2]*(transition_rates[2, 5]/transition_rates[5, 2])^2
    ring_network_temp = kinesin_network(n, transition_rates, visible_weights, runs)
    if ring_network_temp[1] < 1.0
        append!(cur_axis_wait, ring_network_temp[2])
        append!(kur_axis_wait, ring_network_temp[3])
    else
        append!(cur_axis_dwell, ring_network_temp[2])
        append!(kur_axis_dwell, ring_network_temp[3])
    end
end

ring_network_data = [cur_axis_dwell, kur_axis_dwell, cur_axis_wait, kur_axis_wait]
scatter(ring_network_data[1], ring_network_data[2], xlabel = L"\mathcal{S}\tilde{\mathcal{T}}", ylabel = L"\mathcal{S}/\tilde{\mathcal{A}} ", label=L"\tilde{\mathcal{T}} > 1/\tilde{\mathcal{A}}")
scatter!(ring_network_data[3], ring_network_data[4],  label=L"\tilde{\mathcal{T}} < 1/\tilde{\mathcal{A}}")

=#