from ortools.sat.python import cp_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from Environment import Environment
import time

import pandas as pd

# Load the Excel file
file_path = r"C:\Users\thodoris\Documents\Pfizer\result.xlsx"  # Replace with your Excel file path




#lines_list = [2,5,10, 20,50,100,200,500,800,1000,1500]

def Google_OR(env):

    num_tasks = env.num_tasks

    task_to_bit = env.task_to_bit
    task_machine_mapping = env.task_machine_mapping
    inactivity_window_mapping = env.inactivity_window_mapping
    task_order_pairs =env.task_order_pairs
    tuples = env.tuples
    unique_mapping = env.unique_mapping



    Jobs = env.Jobs
    p = env.p
    u_r = env.u_r
    delta = env.delta
    delta_star = env.delta_star
    t_c = env.t_c
    t_c_star = env.t_c_star

    n = env.n
    M = env.M
    num_of_variables = []
    num_of_constraints =  []
    elapsed_time = []

    scale = 1000
    # Define the model
    model = cp_model.CpModel()

    # Decision variables
    S = [model.NewIntVar(0, 1500, f'S_{i}') for i in range(num_tasks)]  # Start times

    #machine assignment variable
    # X = [[None for m in range(num_machines)] for i in range(num_tasks)]
    # for i in range(num_tasks):
    #     for m in range(num_machines):
    #         if m in task_machine_mapping[i]:
    #             X[i][m]=model.NewBoolVar(f'X_{i}_{m}')

    #sequencing variables
    Y = [[None for k in range(num_tasks)] for i in range(num_tasks)]
    for tuple in tuples:
            i = tuple[0]
            k = tuple[1]
            Y[i][k]=model.NewBoolVar(f'Y_{i}_{k}')
            Y[k][i]=model.NewBoolVar(f'Y_{k}_{i}')

    #inactivity period sequencing variables
    #Z = [[model.NewBoolVar(f'Z_{i}_{l}') for l in range(3)] for i in range(num_tasks)]  # Maintenance

    C_max = model.NewIntVar(0, 1500, 'C_max')  # Makespan

    # Objective: Minimize makespan
    model.Minimize(C_max)

    # Constraints
    #Completion Time Constraint
    for i in range(num_tasks):
        for m in task_machine_mapping[i]:
            model.Add(scale*C_max >= scale *  (S[i] +int( p[i] * (2 - u_r[m]))))

    # Task Order Constraint
    for pair in task_order_pairs:
                i = pair[0]
                k = pair[1]
                for m in task_machine_mapping[i]:
                    model.Add(scale * S[k] >= scale * (S[i] + int(p[i] * (2 - u_r[m]))))

    # Single Machine Assignment
    # for i in range(num_tasks):
    #     model.Add(sum(X[i][m] for m in task_machine_mapping[i]) == 1)

    # Maintenance / Holidays Constraints
    # for i in range(num_tasks):
    #     for m in task_machine_mapping[i]:
    #         for l in range(3):
    #             start, end = 50, 60
    #             model.Add(scale * (S[i] + int(p[i] * (2 - u_r[m]))) <= scale * (start + M*(1-Z[i][l]))).OnlyEnforceIf(X[i][m])
    #             model.Add(scale * S[i] >= scale * (end- M*(Z[i][l]))).OnlyEnforceIf(X[i][m])

    # Task Precedence
    for tuple in tuples:
            i = tuple[0]
            k = tuple[1]
            model.Add(Y[i][k] + Y[k][i] == 1)

    # Job Deadline Constraint
    # for j in range(num_jobs):
    #     for i in range(num_tasks):
    #         if Jobs[i] == j:
    #             for m in task_machine_mapping[i]:
    #                 model.Add(scale * (S[i] + int(p[i]* (2 - u_r[m]))) <= scale * D[j]).OnlyEnforceIf(X[i][m])

    # No Machine Overlap
    for tuple in tuples:
            i = tuple[0]
            k = tuple[1]
            for m in (set(task_machine_mapping[i]) & set(task_machine_mapping[k])) :
                print("helloooo")
                model.Add(scale * S[k] >= scale * (S[i] + int(p[i] * (2 - u_r[m]) + int(delta[i][k]*t_c + (1-delta[i][k])*delta_star[i][k]*t_c_star)))).OnlyEnforceIf([Y[i][k]])
                model.Add(scale * S[i] >= scale * (S[k] + int(p[k] * (2 - u_r[m]) + int(delta[i][k]*t_c + (1-delta[i][k])*delta_star[i][k]*t_c_star)))).OnlyEnforceIf([Y[k][i]])

    # Solve the model

    proto = model.Proto()
    print(len(proto.variables))
    print(len(proto.constraints))

    # Get the number of variables
    num_of_variables.append(len(proto.variables))
    num_of_constraints.append(len(proto.constraints))

    start_time = time.time()
    solver = cp_model.CpSolver()
    end_time = time.time()

    elapsed_time.append(end_time - start_time)
    status = solver.Solve(model)

    #print(solver.Solve(model))


    # plt.plot(lines_list, num_of_variables)

    # plt.xlabel('number of tasks')
    # plt.ylabel('number of variables')
    # plt.title("Number of Variables")
    # plt.show()


    # plt.plot(lines_list, num_of_constraints)


    # plt.xlabel('number of tasks')
    # plt.ylabel('number of constraints')
    # plt.title("Number of Constraints")
    # plt.show()

    # plt.plot(lines_list, elapsed_time)


    # plt.xlabel('number of tasks')
    # plt.ylabel('time')
    # plt.title("Elapsed time")
    # plt.show()
    #Check and plot results
    start_times = []
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution Found:")
        tasks_schedule = []
        C_max = solver.Value(C_max)
        for i in range(num_tasks):
            start_time = solver.Value(S[i])
            start_times.append(start_time)
            duration = p[i]
            assigned_machine = None
            for m in task_machine_mapping[i]:
                
                    assigned_machine = f'Machine {m }'
                    duration = p[i] * (2 - u_r[m])
                    break
            tasks_schedule.append((i, assigned_machine, start_time, duration))
            print(f"Task {i}: Start = {start_time},Job = {Jobs[i]} Duration = {duration}, Assigned Machine = {assigned_machine}")

    def int_to_bits_little_endian(value, n):
        return [ (value >> q) & 1 for q in range(n) ]

    def build_bitstring(C_max, start_times, n):
        # Convert C_max to n bits
        
        bits_cmax = int_to_bits_little_endian(C_max, n)
        print(f"Cmax {C_max } Cmaxbit {bits_cmax}")
        # Convert each s[i] to n bits and accumulate
        bits_tasks = []
        for s_val in start_times:
            bits_tasks.extend(int_to_bits_little_endian(s_val, n))
        
        # Concatenate and return as a list of 0/1 or a string of '0'/'1'
        final_bits = bits_cmax + bits_tasks

        bitstring = ''.join(str(b) for b in final_bits)
        for tuple in unique_mapping.keys():
            i = tuple[0]
            k = tuple[1]
            y = solver.Value(Y[i][k])
            print(y)
            bitstring+=str(y)

        print(unique_mapping)
        return bitstring
    
    return build_bitstring(C_max,start_times, n)



if __name__ == "__main__":
    # Example usage:
    env = Environment(num_jobs=6,
                 num_tasks=6,
                 num_machines=6,
                 Horizon=12) 
    print(env)
    print("Task order pairs:", env.task_order_pairs)
    print("Tuples (conflicts):", env.tuples)
    print("Task->Bit mapping:", env.task_to_bit)

    print(Google_OR(env))