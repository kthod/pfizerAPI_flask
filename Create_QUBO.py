

import numpy as np  # Import NumPy for numerical operations
#import networkx as nx  # Import NetworkX for graph operations (not used in the shown code)
from Environment import Environment
import math
from collections import defaultdict

def create_qubo_matrix(env, lambdas :list, num_qubits):
    """
    Function to create a QUBO matrix using the Augmented Lagrangian Method.

    Args:
        lambdas (list): Lagrangian multipliers for the constraints.
        num_qubits (int): Number of qubits in the QUBO problem.

    Returns:
        np.ndarray: QUBO matrix representing the optimization problem.
    """

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

    num_task_order_constraints = len(task_order_pairs)
    num_last_task_constraints = num_tasks
    num_overlap_constraints = 2 * len(tuples)

    # Initialize the QUBO matrix with zeros
    Q = np.zeros((num_qubits,num_qubits))

    def objective_function():
        """
        Define the objective function contribution to the QUBO matrix.

        Returns:
            np.ndarray: Contribution of the objective function to the QUBO matrix.
        """
        Q = np.zeros((num_qubits, num_qubits))
        for q in range(n):
            Q[q][q] += 2**q

        return Q

    def last_task_quad(lambdas):
        """
        Define the quadratic contribution for the 'last task' constraint.

        Args:
            lambdas (list): Lagrangian multipliers for this constraint.

        Returns:
            np.ndarray: Contribution of the 'last task' constraint to the QUBO matrix.
        """
        Q = np.zeros((num_qubits,num_qubits))

        for j in range(num_tasks):
            i = task_to_bit[j]
            m=1
            for machine in task_machine_mapping[j]:
                for q in range(n):
                    Q[q][q] += (2**q) * (2**q  -2 * (p[j]*(2 - u_r[machine])) )*lambdas[j+m-1] #2Μ

                for q in range(n-1):
                    for r in range(q+1, n):
                        Q[q][r] +=  2*(2**q)*(2**r) * lambdas[j+m-1]

                for q in range(n):
                    Q[i+q][i+q] += (2**q) * (2**q  +2 * (p[j]*(2 - u_r[machine]) ) ) * lambdas[j+m-1]#-2Μ

                for q in range(n-1):
                    for r in range(q+1, n):
                        Q[i+q][i+r] +=  2 * (2**q) * (2**r) * lambdas[j+m-1]
   
                for q in range(n):
                    for r in range(n):
                        Q[q][i+r] += -2 * (2**q)*(2**r) * lambdas[j+m-1]


                # Q[i+1+m][i+1+m] += (-M**2 +2*( p[j]*(2 - u_r[machine]) - 1)*M) * lambdas[j+m-1]

                # for q in range(n):
                #     Q[q][i+m+1] += -2 * (2**q) * M * lambdas[j+m-1]

                # for q in range(n):
                #     Q[i+q][i+m+1] += 2 * (2**q) * M * lambdas[j+m-1]


                m+=1 
        
        return Q
    
    def last_task(lambdas: list):
        """
        Define the quadratic contribution for the 'task order' constraint.

        Args:
            lambdas (list): Lagrangian multipliers for this constraint.

        Returns:
            np.ndarray: Contribution of the 'task order' constraint to the QUBO matrix.
        """
        Q = np.zeros((num_qubits,num_qubits))

        for j in range(num_tasks):
            i = task_to_bit[j]
            m=1
            for machine in task_machine_mapping[j+m-1]:

                for q in range(n):
                    Q[q][q] += -(2**q) * lambdas[j+m-1]

                for q in range(n):
                    Q[i+q][i+q] += (2**q) * lambdas[j+m-1]



               # Q[i+1+m][i+1+m] += M*lambdas[j+m-1] 
                m+=1
        
        return Q

    def task_order_quad(lambdas):
        """
        Define the quadratic contribution for the 'task order' constraint.

        Args:
            lambdas (list): Lagrangian multipliers for this constraint.

        Returns:
            np.ndarray: Contribution of the 'task order' constraint to the QUBO matrix.
        """

        Q = np.zeros((num_qubits,num_qubits))

        l = 0
        # for task_i in range(num_tasks):
        #     for task_k in range(task_i + 1, num_tasks):
        for pair in task_order_pairs:
                task_i = pair[0]
                task_k = pair[1]
                print(f"task_i {task_i} task_k {task_k}")
                if Jobs[task_i] == Jobs[task_k]:
        #for j  in [0]:
                    i = task_to_bit[task_i]
                    k = task_to_bit[task_k]

                    print(f"i {i} k {k}")
                    for m in range(1,len(task_machine_mapping[task_i])+1):
                        for q in range(n):
                            Q[k+q][k+q] += (2**q) * (2**q  -2 * p[task_i]*(2 - u_r[m]) )*lambdas[l]

                        for q in range(n-1):
                            for r in range(q+1, n):
                                Q[k+q][k+r] +=  2*(2**q)*(2**r) * lambdas[l]

                        for q in range(n):
                            Q[i+q][i+q] += (2**q) * (2**q  +2 * p[task_i]*(2 - u_r[m])) * lambdas[l]

                        for q in range(n-1):
                            for r in range(q+1, n):
                                Q[i+q][i+r] +=  2 * (2**q) * (2**r) * lambdas[l]

                        for q in range(n):
                            for r in range(n):
                                Q[i+q][k+r] += -2 * (2**q)*(2**r) * lambdas[l]


                        #Q[i+1+m][i+1+m] += (-M**2 + 2*p[task_i]*(2 - u_r[m])*M) * lambdas[l]

                        # for q in range(n):
                        #     Q[k+q][i+m+1] += -2 * (2**q) * M * lambdas[l]

                        # for q in range(n):
                        #     Q[i+q][i+m+1] += 2 * (2**q) * M * lambdas[l]

                    l+=1
                    task_i = task_k
                    task_k = task_i+1
        return Q
    
    def task_order(lambdas):
        """
        Define the linear contribution for the 'task order' constraint.

        Args:
            lambdas (list): Lagrangian multipliers for this constraint.

        Returns:
            np.ndarray: Contribution of the 'task order' constraint to the QUBO matrix.
        """

        Q = np.zeros((num_qubits,num_qubits))

        l=0
        # for task_i in range(num_tasks):
        #     for task_k in range(task_i + 1, num_tasks):
        for pair in task_order_pairs:
                task_i = pair[0]
                task_k = pair[1]
                if Jobs[task_i] == Jobs[task_k]:
        #for j  in [0]:
                    i = task_to_bit[task_i]
                    k = task_to_bit[task_k]
                    for m in range(1,len(task_machine_mapping[task_i])+1):

                        for q in range(n):
                            Q[k+q][k+q] += -(2**q) * lambdas[l]

                        for q in range(n):
                            Q[i+q][i+q] += (2**q) * lambdas[l]


                        #Q[i+1+m][i+1+m] += M*lambdas[l] 
                    l+=1

                    task_i = task_k
                    task_k = task_i+1
        
        return Q
    

    def no_overlap_quad(lambdas):
        """
        Generate a quadratic matrix Q to enforce no-overlap constraints for tasks assigned to the same machine.

        This function ensures that tasks scheduled on the same machine do not overlap by considering:
        - Task durations
        - Cleaning times for tasks of different brands or API strengths
        - Machine-specific configurations

        Since the order of two tasks is not known in advance, this function handles both possible scenarios:
        1. Task i precedes Task k.
        2. Task k precedes Task i.

        Two separate constraints are included to capture these cases.

        Parameters:
            lambdas (list): Coefficients for the penalty terms in the quadratic formulation.

        Returns:
            numpy.ndarray: A quadratic matrix Q representing the no-overlap constraints.
        """


        Q = np.zeros((num_qubits,num_qubits))
        l = 0
        for tuple  in tuples:
            #locate the bit address of the corresponding y variable
            y = unique_mapping[tuple] +task_to_bit[num_tasks-1] +n#+ len(task_machine_mapping[num_tasks-1])+len(inactivity_window_mapping[num_tasks-1]) 
            i = task_to_bit[tuple[0]] #locate the bit address of task i 
            k = task_to_bit[tuple[1]] #locate the bit address of task k

            task_i = tuple[0]
            task_k = tuple[1]

            print(f"tuple {tuple} i {i} k {k} y {y}")
            for m in (set(task_machine_mapping[task_i]) & set(task_machine_mapping[task_k])) :
                print(f"machine {m}")
                A = p[task_i]*(2 - u_r[m]) +delta[tuple[0]][tuple[1]]*t_c +  (1-delta[tuple[0]][tuple[1]])*delta_star[tuple[0]][tuple[1]]*t_c_star
                B = p[task_k]*(2 - u_r[m]) +delta[tuple[0]][tuple[1]]*t_c +  (1-delta[tuple[0]][tuple[1]])*delta_star[tuple[0]][tuple[1]]*t_c_star

                #First Constraint

                for q in range(n):
                    Q[k+q][k+q] += (2**q) * (2**q - 2 * A + 2 * M)*lambdas[l]#6Μ

                for q in range(n-1):
                    for r in range(q+1, n):
                        Q[k+q][k+r] +=  2*(2**q)*(2**r) * lambdas[l]

                #Second Constraint

                for q in range(n):
                    Q[k+q][k+q] += (2**q) * (2**q + 2 * B ) * lambdas[l+1]#-4Μ

                for q in range(n-1):
                    for r in range(q+1, n):
                        Q[k+q][k+r] +=  2*(2**q)*(2**r) * lambdas[l+1]


                #First Constraint
                for q in range(n):
                    Q[i+q][i+q] += (2**q) * (2**q + 2 * A - 2 * M)*lambdas[l] #-6Μ

                for q in range(n-1):
                    for r in range(q+1, n):
                        Q[i+q][i+r] +=  2*(2**q)*(2**r) * lambdas[l]

                #Second Constraint
                for q in range(n):
                    Q[i+q][i+q] += (2**q) * (2**q - 2 * B ) * lambdas[l+1]

                for q in range(n-1):
                    for r in range(q+1, n):
                        Q[i+q][i+r] +=  2*(2**q)*(2**r) * lambdas[l+1]

                #First Constraint
                for q in range(n):
                    for r in range(n):
                        Q[i+q][k+r] += -2 * (2**q)*(2**r) * lambdas[l]
                #Second Constrainta
                for q in range(n):
                    for r in range(n):
                        Q[i+q][k+r] += -2 * (2**q)*(2**r) * lambdas[l+1]

                #First Constraint
                # Q[i+1+m][i+1+m] += (-5*M**2 + 2 * M * A) * lambdas[l]
                # Q[k+1+m][k+1+m] += (-5*M**2 + 2 * M * A) * lambdas[l]
                Q[y][y] += (-1*M**2 + 2 * M * A) * lambdas[l]
                #Second Constraint
                # Q[i+1+m][i+1+m] += (-3*M**2 + 2 * M * B) * lambdas[l+1]
                # Q[k+1+m][k+1+m] += (-3*M**2 + 2 * M * B) * lambdas[l+1]
                Q[y][y] += (1*M**2 - 2 * M * B) * lambdas[l+1]

                #First Constraint
                # for q in range(n):
                #     Q[i+m+1][k+q] += -2 * (2**q) * M * lambdas[l]

                #Second Constraint
                # for q in range(n):
                #     Q[i+m+1][k+q] += 2 * (2**q) * M * lambdas[l+1]

                #First Constraint 
                # for q in range(n):
                #     Q[i+q][i+1+m] += 2 * (2**q) * M * lambdas[l]
                #Second Constrant
                # for q in range(n):
                #     Q[i+q][i+1+m] += -2 * (2**q) * M * lambdas[l+1]

                #First Constraint
                # for q in  range(n):
                #     Q[k+m+1][k+q] += -2 * (2**q) * M * lambdas[l]

                #Second Constraint
                # for q in range(n):
                #     Q[k+m+1][k+q] += 2 * (2**q) * M * lambdas[l+1]

                #First Constraint
                # for q in range(n):
                #     Q[i+q][k+1+m] += 2 * (2**q) * M * lambdas[l]

                #Second Constraint
                # for q in range(n):
                #     Q[i+q][k+1+m] += -2 * (2**q) * M * lambdas[l+1]

                #First Constraint
                for q in range(n):
                    Q[y][k+q] += -2 * (2**q) * M * lambdas[l]

                #Second Constraint
                for q in range(n):
                    Q[y][k+q] += -2 * (2**q) * M * lambdas[l+1]

                
                #First Constraint
                for q in range(n):
                    Q[i+q][y] += 2 * (2**q) * M * lambdas[l]

                #Second Constraint
                for q in range(n):
                    Q[i+q][y] += 2 * (2**q) * M * lambdas[l+1]


                #First Constraint
                # Q[i+m+1][y] += 2 * M**2* lambdas[l]
                # Q[k+m+1][y] += 2 * M**2* lambdas[l]
                # Q[k+m+1][i+m+1] += 2 * M**2* lambdas[l]
                # #Second Constraint
                # Q[i+m+1][y] += -2 * M**2* lambdas[l+1]
                # Q[k+m+1][y] += -2 * M**2* lambdas[l+1]
                # Q[k+m+1][i+m+1] += 2 * M**2* lambdas[l+1]
            
                l+=2
        return Q
    
    def no_overlap(lambdas):
        """
        Generate a matrix Q to enforce no-overlap constraints for tasks assigned to the same machine.

        Parameters:
            lambdas (list): Coefficients for the penalty terms in the quadratic formulation.

        Returns:
            numpy.ndarray: A quadratic matrix Q representing the no-overlap constraints.
        """

        Q = np.zeros((num_qubits,num_qubits))

        l = 0
        for tuple  in tuples:
            y = unique_mapping[tuple] +task_to_bit[num_tasks-1] + n# + len(task_machine_mapping[num_tasks-1])+len(inactivity_window_mapping[num_tasks-1])
            i = task_to_bit[tuple[0]]
            k = task_to_bit[tuple[1]] 
            task_i = tuple[0]
            task_k = tuple[1]
            for m in (set(task_machine_mapping[task_i]) & set(task_machine_mapping[task_k])) :
                #First Constraint
                for q in range(n):
                    Q[k+q][k+q] += -(2**q) *lambdas[l]  
                
                for q in range(n):
                    Q[i+q][i+q] += (2**q) *lambdas[l]  

                # Q[i+1+m][i+1+m] += M*lambdas[l] 
                # Q[k+1+m][k+1+m] += M*lambdas[l] 
                Q[y][y] += M*lambdas[l] 

                #Second Constraint
                for q in range(n):
                    Q[k+q][k+q] +=  (2**q) *lambdas[l+1]  
               
                for q in range(n):
                    Q[i+q][i+q] += -(2**q) *lambdas[l+1] 

                # Q[i+1+m][i+1+m] += -M*lambdas[l+1] 
                # Q[k+1+m][k+1+m] += -M*lambdas[l+1] 
                Q[y][y] += -M*lambdas[l+1] 

                l+=2

        
        return Q
    
    def inactivity_windows_quad(lambdas):
        """
        Generate a quadratic matrix Q to enforce inactivity window constraints for tasks assigned to specific machines.

        This function ensures that tasks respect the inactivity windows defined for their assigned machines. For each task, it considers:
        - Task durations
        - Machine-specific configurations
        - The defined start and end times of inactivity windows

        Two constraints are included for each inactivity window:
        1. The task ends before the start of the inactivity window.
        2. The task starts after the end of the inactivity window.

        Parameters:
            lambdas (list): Coefficients for the penalty terms in the quadratic formulation.

        Returns:
            numpy.ndarray: A quadratic matrix Q representing the inactivity window constraints.
        """

        Q = np.zeros((num_qubits,num_qubits))

        l=0
        for j in [0,1,2,3]:
            i = task_to_bit[j]
            len_machines = len(task_machine_mapping[j])
            z = i+n+len_machines
            m=0
            for w in inactivity_window_mapping[j]:
                
                for machine in task_machine_mapping[j]:

                    #First Constraint
                    A = p[j]*(1-u_r[machine])-w[0]-2*M
                    for q in range(n):
                        Q[i+q][i+q] += (2**q) * (2**q  +2 * A)*lambdas[l]

                    for q in range(n-1):
                        for r in range(q+1, n):
                            Q[i+q][i+r] +=  2*(2**q)*(2**r) * lambdas[l]


                    Q[i+n+m][i+n+m] += (M**2 + 2*A*M) * lambdas[l]
                    Q[z][z] += (M**2 + 2*A*M) * lambdas[l]



                    for q in range(n):
                        Q[i+q][i+n+m] += 2 * (2**q) * M * lambdas[l]

                    for q in range(n):
                        Q[i+q][z] += 2 * (2**q) * M * lambdas[l]

                    Q[i+n+m][z] += 2*(M**2) * lambdas[l]

                    #Second Constraint
                    B = w[1]-M
                    for q in range(n):
                        Q[i+q][i+q] += (2**q) * (2**q  -2 * B)*lambdas[l+1]

                    for q in range(n-1):
                        for r in range(q+1, n):
                            Q[i+q][i+r] +=  2*(2**q)*(2**r) * lambdas[l+1]
                       


                    Q[i+n+m][i+n+m] += (M**2 + 2*B*M) * lambdas[l+1]
                    Q[z][z] += (M**2 - 2*B*M) * lambdas[l+1]



                    for q in range(n):
                        Q[i+q][i+n+m] += -2 * (2**q) * M * lambdas[l+1]

                    for q in range(n):
                        Q[i+q][z] += 2 * (2**q) * M * lambdas[l+1]

                    Q[i+n+m][z] += -2*(M**2) * lambdas[l+1]

                    l+=2
                    m+=1
            
        
        return Q
    
    def inactivity_windows(lambdas: list):
        
        Q = np.zeros((num_qubits,num_qubits))

        i = n
        l=0
        for j in [0,1,2,3]:
            i = task_to_bit[j]
            len_machines = len(task_machine_mapping[j])
            z = i+n+len_machines
            
            m=1
            for machine in task_machine_mapping[j+m-1]:
                
                #First Constraint
                for q in range(n):
                    Q[i+q][i+q] += (2**q) * lambdas[l]



                Q[i+n+m][i+n+m] += M*lambdas[l] 
                Q[z][z] += M*lambdas[l]

                #Second Constraint
                for q in range(n):
                    Q[i+q][i+q] += -(2**q) * lambdas[l+1]



                Q[i+n+m][i+n+m] += M*lambdas[l+1] 
                Q[z][z] += -M*lambdas[l+1]
                m+=1
                l+=2
        
        
        return Q

    # Adjacency is essentially a matrix which tells you which nodes are connected.

    first_batch = num_task_order_constraints
    second_batch = first_batch + num_last_task_constraints
    third_batch = second_batch + num_overlap_constraints

    Q = objective_function() +5*( 1*task_order(lambdas[0:first_batch]) +1/M*task_order_quad(lambdas[0:first_batch])+no_overlap(lambdas[second_batch:third_batch]) + 1/M*no_overlap_quad(lambdas[second_batch:third_batch]) + last_task(lambdas[first_batch:second_batch])+ 1/M*last_task_quad(lambdas[first_batch:second_batch]))#++ inactivity_windows(lambdas[10:]) + 0.5*inactivity_windows_quad(lambdas[10:])#+last_task(lambdas[2:6])+ 0.5*last_task_quad(lambdas[2:6]) #+ inactivity_windows(lambdas[10:]) + 0.5*inactivity_windows_quad(lambdas[10:]) #
        
    return  Q