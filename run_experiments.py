"""Run experiments programmatically: compute policy via FittedValueIteration,
save cost/policy images, and measure max control + final error.
"""
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pydrake.all import StartMeshcat, FittedValueIteration, PeriodicBoundaryCondition
from on_a_mesh import (
    discretize_state_and_input,
    PENDULUM_DISCRETIZATION,
    PendulumPlant,
    extract_policy_grid,
    simulate_and_measure_policy,
)

from pydrake.all import Simulator, DynamicProgrammingOptions


def run_one(experiment, output_dir="."):
    meshcat = StartMeshcat()
    # copy discretization and override if provided
    disc = PENDULUM_DISCRETIZATION.copy()
    for k in ["nq", "nqdot", "nu", "q_range", "qdot_range", "u_range"]:
        if k in experiment:
            disc[k] = experiment[k]

    qbins, qdotbins, _ubins, state_grid, input_grid, Q, Qdot = discretize_state_and_input(
        disc["q_range"], disc["qdot_range"], disc["u_range"], disc["nq"], disc["nqdot"], disc["nu"]
    )

    # numerical integration / VI timestep and DP options
    time_step = 0.01
    options = DynamicProgrammingOptions()
    # angle (index 0) is periodic from 0 to 2pi
    options.periodic_boundary_conditions = [PeriodicBoundaryCondition(0, 0.0, 2.0 * np.pi)]
    # use a high discount to approximate long-horizon cost
    options.discount_factor = 0.999

    def min_time_cost(context):
        x = context.get_continuous_state_vector().CopyToVector()
        x[0] = x[0] - np.pi
        if x.dot(x) < 0.05:
            return 0.0
        return 1.0

    def quadratic_regulator_cost(context):
        x = context.get_continuous_state_vector().CopyToVector()
        x[0] = x[0] - np.pi
        u = (context.get_numeric_parameter(0) if False else None)
        # fallback: use simple quadratic on state (we won't use u here in VI)
        return 2 * x.dot(x)

    # select cost function and appropriate convergence tolerance
    if experiment.get("min_time", True):
        cost_function = min_time_cost
        # tighter tolerance for min-time problems (discontinuous cost)
        options.convergence_tol = 0.001
    else:
        cost_function = quadratic_regulator_cost
        # looser tolerance for smooth quadratic cost
        options.convergence_tol = 0.1

    simulator = Simulator(PendulumPlant())

    # run value iteration and time the operation
    t0 = time.time()
    policy, cost_to_go = FittedValueIteration(simulator, cost_function, state_grid, input_grid, time_step, options)
    duration = time.time() - t0

    J = np.reshape(cost_to_go, Q.shape)
    Pi = extract_policy_grid(policy, Q.shape)

    os.makedirs(output_dir, exist_ok=True)
    cost_path = os.path.join(output_dir, f"cost_{experiment['name']}.png")
    policy_path = os.path.join(output_dir, f"policy_{experiment['name']}.png")

    fig_cost = plt.figure(figsize=(6, 4))
    ax_cost = fig_cost.subplots(1, 1)
    ax_cost.imshow(J, cmap=cm.jet, aspect="auto", extent=(qbins[0], qbins[-1], qdotbins[-1], qdotbins[0]))
    ax_cost.invert_yaxis()
    fig_cost.tight_layout()
    fig_cost.savefig(cost_path, dpi=150)
    plt.close(fig_cost)

    fig_pi = plt.figure(figsize=(6, 4))
    ax_pi = fig_pi.subplots(1, 1)
    ax_pi.imshow(Pi, cmap=cm.jet, aspect="auto", extent=(qbins[0], qbins[-1], qdotbins[-1], qdotbins[0]))
    ax_pi.invert_yaxis()
    fig_pi.tight_layout()
    fig_pi.savefig(policy_path, dpi=150)
    plt.close(fig_pi)

    # measure closed-loop behavior from a small offset initial condition
    # (theta=0.1 rad, theta_dot=0) to quantify control effort and final error
    max_u, final_err = simulate_and_measure_policy(policy, [0.1, 0.0], duration=8.0)

    result = {
        "name": experiment["name"],
        "min_time": experiment.get("min_time", True),
        "nq": disc["nq"],
        "nqdot": disc["nqdot"],
        "nu": disc["nu"],
        "runtime_s": duration,
        "max_u": max_u,
        "final_error_norm": final_err,
        "cost_path": cost_path,
        "policy_path": policy_path,
    }
    return result


if __name__ == "__main__":
    experiments = [
        {"name": "min_time_base", "min_time": True},
        {"name": "quad_base", "min_time": False},
        {"name": "min_time_coarse", "min_time": True, "nq": 31, "nqdot": 31, "nu": 7},
    ]

    out = []
    for e in experiments:
        print(f"Running {e['name']}...")
        r = run_one(e, output_dir="./results/experiments/mild")
        print(r)
        out.append(r)

    import json

    os.makedirs("./results", exist_ok=True)
    with open("./results/experiment_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results to ./results/experiment_results.json")
