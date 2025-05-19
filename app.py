from flask import Flask, request, jsonify
import os
import numpy as np
#from typing import List, Dict, Any
import logging
from CompressedVQE_Class import CompressedVQE
from Create_QUBO import create_qubo_matrix
from Environment import Environment
from GoogleOR import Google_OR
    # import qiskit
    # import matplotlib.pyplot as plt
    # import io
    # import base64

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce Qiskit's logging verbosity
qiskit_logger = logging.getLogger('qiskit')
qiskit_logger.setLevel(logging.WARNING)

app = Flask(__name__)

# Configuration
app.config['JSON_SORT_KEYS'] = False  # Maintain JSON response order

# def create_optimization_plot(cost_history: List[float], start_times: List[int], assigned_machines: List[int]) -> Dict[str, Any]:
#     """Create plot data for the optimization results."""
#     # Create a figure with two subplots
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
#     # Plot cost function evolution
#     if cost_history:
#         ax1.plot(cost_history)
#         ax1.set_title('Cost Function Evolution')
#         ax1.set_xlabel('Iteration')
#         ax1.set_ylabel('Cost')
#         ax1.grid(True)
    
#     # Plot Gantt chart
#     tasks = range(len(start_times))
#     ax2.barh(tasks, [1] * len(tasks), left=start_times, height=0.5)
#     ax2.set_title('Task Schedule')
#     ax2.set_xlabel('Time')
#     ax2.set_ylabel('Task')
#     ax2.grid(True)
    
#     # Save plot to a bytes buffer
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight')
#     buf.seek(0)
    
#     # Convert plot to base64 string
#     plot_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    
#     # Create plot data for client-side plotting
#     plot_data = {
#         'cost_history': cost_history,
#         'start_times': start_times,
#         'assigned_machines': assigned_machines,
#         'tasks': list(tasks)
#     }
    
#     plt.close()
    
#     return {
#         'plot_data': plot_data,
#         'plot_image': plot_image
#     }

@app.route('/', methods=['GET'])
def root():
    """Root endpoint that returns API information."""
    return jsonify({
        "message": "Welcome to the Quantum Optimization API",
        "documentation": "/docs",
        "endpoints": {
            "GET /": "This information page",
            "POST /optimize": "Submit optimization problem"
        }
    })

@app.route('/optimize', methods=['POST'])
def optimize():
    """Optimize a scheduling problem using quantum computing."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Extract parameters from request
        num_jobs = data.get('num_jobs')
        num_tasks = data.get('num_tasks')
        num_machines = data.get('num_machines')
        p = data.get('p')
        Jobs = data.get('Jobs')
        Brands = data.get('Brands')
        mg = data.get('mg')
        task_machine_mapping = data.get('task_machine_mapping')
        layers = data.get('layers', 3)
        n_measurements = data.get('n_measurements', 20000)
        number_of_experiments = data.get('number_of_experiments', 1)
        maxiter = data.get('maxiter', 300)
        Horizon = data.get('Horizon', 32)
        # Validate required fields
        required_fields = ['num_jobs', 'num_tasks', 'num_machines', 'p', 'Jobs', 'Brands', 'mg', 'task_machine_mapping']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        logger.info("Received optimization request")
        logger.info(f"Number of tasks: {num_tasks}")
        logger.info(f"Number of machines: {num_machines}")
        logger.info(f"Number of jobs: {num_jobs}")
        
        # Log the first few entries of each list for debugging
        logger.info(f"First 3 processing times: {p[:3]}")
        logger.info(f"First 3 jobs: {Jobs[:3]}")
        logger.info(f"First 3 brands: {Brands[:3]}")
        logger.info(f"First 3 machine groups: {mg[:3]}")
        logger.info(f"First 3 task-machine mappings: {dict(list(task_machine_mapping.items())[:3])}")

        # Create environment
        logger.info("Creating environment...")
        try:
            env = Environment(
                num_jobs=num_jobs,
                num_tasks=num_tasks,
                num_machines=num_machines,
                Horizon=Horizon,
                p=p,
                Jobs=Jobs,
                Brands=Brands,
                mg=mg,
                task_machine_mapping={int(k): set(v) for k, v in task_machine_mapping.items()}
            )
            logger.info("Environment created successfully")
        except Exception as e:
            logger.error(f"Error creating environment: {str(e)}")
            return jsonify({'error': f'Error creating environment: {str(e)}'}), 400

        # Create QUBO matrix
        logger.info("Creating QUBO matrix...")
        try:
            print(f"env.n: {env.n} env.num_tasks: {env.num_tasks} env.num_machines: {env.num_machines}")
            lambdas = np.array([1]*1000)
            Q = create_qubo_matrix(env, lambdas, env.n + env.num_tasks*(env.n) + len(env.tuples))
            logger.info("QUBO matrix created successfully")
        except Exception as e:
            logger.error(f"Error creating QUBO matrix: {str(e)}")
            return jsonify({'error': f'Error creating QUBO matrix: {str(e)}'}), 400

        bitstring_or = Google_OR(env)
        bitstring = np.array([int(x) for x in bitstring_or])
        opt_value = bitstring.T @ Q @ bitstring

        # Run optimization
        logger.info("Starting optimization...")
        try:
            inst = CompressedVQE(Q, layers=layers, na=int(env.n))
            initial_vector = inst.optimize(
                n_measurements=n_measurements,
                number_of_experiments=number_of_experiments,
                maxiter=maxiter
            )
            logger.info("Optimization completed successfully")
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            return jsonify({'error': f'Error during optimization: {str(e)}'}), 400

        # Get solution
        logger.info("Getting solution...")
        try:
            bitstring = inst.show_solution(shots=1000000)
            logger.info("Solution obtained successfully")
        except Exception as e:
            logger.error(f"Error getting solution: {str(e)}")
            return jsonify({'error': f'Error getting solution: {str(e)}'}), 400

        # Get cost evolution data
        upper_bound, lower_bound, mean = inst.plot_evolution(normalization=[opt_value], label=f"CompressedVQE na={env.n}, shots={n_measurements}")

        cost_evolution = {
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
            "mean": mean
        }

        logger.info("Optimization completed successfully")
        return jsonify({
            'solution': bitstring,
            'cost_function_value': inst.compute_expectation({bitstring: 1}),
            'cost_evolution': cost_evolution
        })

    except Exception as e:
        logger.error(f"Unexpected error during optimization: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # Get port from environment variable (AWS EB sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # In production, we don't want debug mode
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=debug) 