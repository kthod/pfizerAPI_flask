from ortools.linear_solver import pywraplp
import time
from Environment import Environment

def Google_OR_SCIP(env):

    # Unpack environment data
    num_tasks               = env.num_tasks
    task_machine_mapping    = env.task_machine_mapping
    task_order_pairs        = env.task_order_pairs
    tuples                  = env.tuples            # pairs with potential overlap
    unique_mapping          = env.unique_mapping
    Jobs = env.Jobs
    p       = env.p
    u_r     = env.u_r
    delta   = env.delta
    delta_star = env.delta_star
    t_c     = env.t_c
    t_c_star= env.t_c_star

    # A sufficiently large M for big-M linearization
    M = 10**6

    # Create SCIP solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise RuntimeError("SCIP solver unavailable")

    # Decision variables
    S = [solver.IntVar(0, 1500, f'S_{i}') for i in range(num_tasks)]
    Y = {}
    for (i, k) in tuples:
        Y[(i,k)] = solver.BoolVar(f'Y_{i}_{k}')
        Y[(k,i)] = solver.BoolVar(f'Y_{k}_{i}')
        # exactly one ordering
        solver.Add(Y[(i,k)] + Y[(k,i)] == 1)

    C_max = solver.IntVar(0, 1500, 'C_max')

    # Objective: minimize makespan
    solver.Minimize(C_max)

    # 1) Makespan ≥ every task’s completion on its *longest* eligible machine
    for i in range(num_tasks):
        max_d = max(p[i] * (2 - u_r[m]) for m in task_machine_mapping[i])
        solver.Add(C_max >= S[i] + max_d)

    # 2) Precedence pairs: i → k
    for (i, k) in task_order_pairs:
        # enforce on the worst-case machine duration
        max_d = max(p[i] * (2 - u_r[m]) for m in task_machine_mapping[i])
        solver.Add(S[k] >= S[i] + max_d)

    # 3) No‐overlap on shared machines via big-M + Y vars
    for (i, k) in tuples:
        # only if they share a machine
        if set(task_machine_mapping[i]) & set(task_machine_mapping[k]):
            # compute sequence‐dependent durations
            c_ik = p[i] * (2 - u_r[ task_machine_mapping[i][0] ]) \
                   + (delta[i][k]*t_c + (1-delta[i][k])*delta_star[i][k]*t_c_star)
            c_ki = p[k] * (2 - u_r[ task_machine_mapping[k][0] ]) \
                   + (delta[i][k]*t_c + (1-delta[i][k])*delta_star[i][k]*t_c_star)
            # if Y[i,k]==1 then i before k; else no restriction
            solver.Add(S[k] >= S[i] + c_ik - M*(1 - Y[(i,k)]))
            # if Y[k,i]==1 then k before i
            solver.Add(S[i] >= S[k] + c_ki - M*(1 - Y[(k,i)]))

    # Solve
    start = time.time()
    result = solver.Solve()
    elapsed = time.time() - start

    if result not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        print("No feasible solution found.")
        return None

    # Extract solution
    start_times = [int(S[i].solution_value()) for i in range(num_tasks)]
    cmax_val    = int(C_max.solution_value())
    print(f"SCIP MIP solved in {elapsed:.2f}s, C_max={cmax_val}")

    for i in range(num_tasks):
        # just report first eligible machine and its duration
        m = task_machine_mapping[i][0]
        dur = int(p[i] * (2 - u_r[m]))
        print(f"Task {i}: Start = {start_times[i]},Job = {Jobs[i]} Duration = {dur}, Assigned Machine = {m}")

    # Build bitstring
    def int_to_bits_le(x, n):
        return [(x >> b) & 1 for b in range(n)]

    bits = int_to_bits_le(cmax_val, env.n)
    for s in start_times:
        bits += int_to_bits_le(s, env.n)
    for (i,k) in unique_mapping.keys():
        bits.append(int(Y[(i,k)].solution_value()))

    return ''.join(str(b) for b in bits)


if __name__ == "__main__":
    env = Environment(num_jobs=6,
                      num_tasks=6,
                      num_machines=6,
                      Horizon=12)
    bs = Google_OR_SCIP(env)
    print("Bitstring:", bs)
