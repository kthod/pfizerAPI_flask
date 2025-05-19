import math
import numpy as np
from collections import defaultdict

class Environment:
    def __init__(self,
                 num_jobs=6,
                 num_tasks=6,
                 num_machines=6,
                 Horizon=12,
                 p=None,
                 Jobs=None,
                 Brands=None,
                 mg=None,
                 task_machine_mapping = None):
        """
        Initialize the environment with default or user-specified values.
        """
        # Basic parameters
        self.num_jobs = num_jobs
        self.num_tasks = num_tasks
        self.num_machines = num_machines
        self.Horizon = Horizon
        #self.Deadlines = {job: Horizon for job in Jobs}

        # Derived parameter
        self.n = math.ceil(np.log2(self.Horizon))

        # Initialize arrays/lists if not provided
        self.p = p if p is not None else 2*[3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
        self.Jobs = Jobs if Jobs is not None else 2*[0, 0, 0, 1, 0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.Brands = Brands if Brands is not None else 2*[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.mg = mg if mg is not None else 2*[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Dictionary mapping job -> list of tasks in ascending order
        self.value_to_indices = defaultdict(list)
        for i, val in enumerate(self.Jobs[:self.num_tasks]):
            self.value_to_indices[val].append(i)

        # Sort task indices and build consecutive pairs (task_order_pairs)
        self.task_order_pairs = []
        for val, indices in self.value_to_indices.items():
            indices.sort()
            for i in range(len(indices) - 1):
                self.task_order_pairs.append((indices[i], indices[i+1]))

        # Define task-machine mapping ( random initialization if no input is given)
        self.task_machine_mapping = task_machine_mapping if task_machine_mapping is not None else {
            0: [0],
            1: [0],
            2: [1],
            3: [1],
            4: [1],
            5: [2],
            6: [2],
            7: [2],
            8: [3],
            9: [3],
            10: [3],
            11: [4],
            12: [4],
            13: [4],
            14: [4],
            15: [5],
            16: [5],
            17: [5]
        }

        # Inactivity window mapping (example as empty lists)
        self.inactivity_window_mapping = [[(2,3)] for _ in range(num_tasks)]

        # Machine utilization rate (u_r)
        self.u_r = self.num_machines * [1]


        # M = Horizon + 2
        self.M = self.Horizon + 2

        # Compute delta array (brand-based cleaning requirements)
        self.delta = np.zeros((self.num_tasks, self.num_tasks))
        for i in range(self.num_tasks):
            for k in range(self.num_tasks):
                if self.Brands[i] != self.Brands[k]:
                    self.delta[i][k] = 1
                else:
                    self.delta[i][k] = 0

        # Compute delta_star array (API strength-based cleaning requirements)
        self.delta_star = np.zeros((self.num_tasks, self.num_tasks))
        for i in range(self.num_tasks):
            for k in range(self.num_tasks):
                if self.mg[i] != self.mg[k]:
                    self.delta_star[i][k] = 1
                else:
                    self.delta_star[i][k] = 0

        # Cleaning times
        self.t_c = 2      # Cleaning time for different brands
        self.t_c_star = 0 # Cleaning time for different API strengths

        # Build tuples for conflicting tasks (sharing a machine)
        self.tuples = []
        for i in range(self.num_tasks):
            for k in range(i + 1, self.num_tasks):
                if set(self.task_machine_mapping.get(i, [])) & set(self.task_machine_mapping.get(k, [])):
                    self.tuples.append((i, k))

        # Unique mapping for each tuple
        self.unique_mapping = {t: i for i, t in enumerate(set(self.tuples))}

        # Build dictionary for task -> bit location
        self.task_to_bit = {}
        prev = self.n
        for i in range(self.num_tasks):
            self.task_to_bit[i] = prev
            prev += self.n

        # Example constraint counts
        self.num_task_order_constraints = len(self.task_order_pairs)
        self.num_last_task_constraints = self.num_tasks
        self.num_overlap_constraints = 2 * len(self.tuples)

    def __repr__(self):
        """
        A quick string representation so you can see some key attributes.
        """
        return (f"Environment(num_jobs={self.num_jobs}, num_tasks={self.num_tasks}, "
                f"num_machines={self.num_machines}, Horizon={self.Horizon})")


if __name__ == "__main__":
    # Example usage:
    env = Environment()
    print(env)
    print("Task order pairs:", env.task_order_pairs)
    print("Tuples (conflicts):", env.tuples)
    print("Task->Bit mapping:", env.task_to_bit)
