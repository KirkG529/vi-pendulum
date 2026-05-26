"""Value iteration examples for double integrator and pendulum (Drake).

Converted from the original notebook to a standalone script.

This module provides:
- grid construction utilities (`discretize_state_and_input`)
- example pipelines for double-integrator and pendulum problems
- a small closed-loop simulation helper to measure control magnitudes

The code is organized so the core solver call is `FittedValueIteration` and
the surrounding functions handle visualization, saving figures, and metrics.
"""

from __future__ import annotations

import argparse
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib import cm
from pydrake.all import (
    DiagramBuilder,
    DynamicProgrammingOptions,
    FittedValueIteration,
    LinearSystem,
    PeriodicBoundaryCondition,
    Rgba,
    Simulator,
    StartMeshcat,
    WrapToSystem,
)
from pydrake.examples import PendulumPlant

from underactuated.double_integrator import DoubleIntegratorVisualizer
from underactuated.jupyter import running_as_notebook
from underactuated.pendulum import PendulumVisualizer
from underactuated.pyplot_utils import AdvanceToAndVisualize


def discretize_state_and_input(q_range, qdot_range, u_range, nq, nqdot, nu):
    """Return (qbins, qdotbins, ubins, state_grid, input_grid, Q, Qdot)."""
    qbins = np.linspace(q_range[0], q_range[1], nq)
    qdotbins = np.linspace(qdot_range[0], qdot_range[1], nqdot)
    ubins = np.linspace(u_range[0], u_range[1], nu)
    state_grid = [set(qbins), set(qdotbins)]
    input_grid = [set(ubins)]
    Q, Qdot = np.meshgrid(qbins, qdotbins)
    return qbins, qdotbins, ubins, state_grid, input_grid, Q, Qdot


def extract_policy_grid(policy, q_shape):
    """Extract the optimal feedback policy pi* on the grid as a 2D array."""
    return np.reshape(policy.get_output_values(), q_shape)


def simulate_and_measure_policy(policy, initial_state, duration=8.0):
    """Module-level simulation to measure max control and final error for a given policy.

    Returns (max_u, final_error_norm)
    """
    # Build a simple closed-loop diagram: PendulumPlant <- policy <- WrapToSystem
    builder = DiagramBuilder()
    pendulum = builder.AddSystem(PendulumPlant())

    # Wrap angle output to [0, 2pi) before feeding policy (policy expects wrapped angle)
    wrap = builder.AddSystem(WrapToSystem(2))
    wrap.set_interval(0, 0, 2 * np.pi)
    builder.Connect(pendulum.get_output_port(0), wrap.get_input_port(0))

    # Insert the policy as a subsystem and wire inputs/outputs
    vi_policy = builder.AddSystem(policy)
    builder.Connect(wrap.get_output_port(0), vi_policy.get_input_port(0))
    builder.Connect(vi_policy.get_output_port(0), pendulum.get_input_port(0))

    # Add a headless visualizer (not shown) so we can reuse AdvanceToAndVisualize when needed
    visualizer = builder.AddSystem(PendulumVisualizer(show=False))
    builder.Connect(pendulum.get_output_port(0), visualizer.get_input_port(0))

    diagram = builder.Build()
    sim = Simulator(diagram)
    # set initial continuous state (theta, theta_dot)
    sim.get_mutable_context().SetContinuousState(initial_state)

    # Step the simulator in small increments and sample the policy output each step.
    # This manual sampling avoids dependency on SignalLogger which may not be available
    dt = 0.01
    t_end = duration
    samples = []
    t = 0.0
    while t < t_end:
        next_t = min(t_end, t + dt)
        sim.AdvanceTo(next_t)
        # evaluate policy output using the policy's subsystem context
        try:
            policy_ctx = diagram.GetSubsystemContext(vi_policy, sim.get_context())
            out = vi_policy.get_output_port(0).Eval(policy_ctx)
            # first try to get a numerical vector
            try:
                val = out.CopyToVector()
                samples.append(float(val[0]))
            except Exception:
                # fallback for abstract-valued outputs
                try:
                    samples.append(float(out.get_value()))
                except Exception:
                    # if extraction fails, skip this sample
                    pass
        except Exception:
            # if any lower-level call fails, continue the simulation loop
            pass
        t = next_t

    # compute maximum absolute control magnitude seen during the rollout
    if len(samples) == 0:
        max_u = 0.0
    else:
        max_u = float(np.max(np.abs(np.array(samples))))

    # read final pendulum state from the pendulum subsystem context
    pendulum_ctx = diagram.GetSubsystemContext(pendulum, sim.get_context())
    x_final = pendulum_ctx.get_continuous_state_vector().CopyToVector()
    # target state is theta = pi (upright), theta_dot = 0
    x_target = np.array([np.pi, 0.0])
    err = x_final.copy()
    # wrap angle error into [-pi, pi]
    err[0] = ((err[0] - x_target[0] + np.pi) % (2 * np.pi)) - np.pi
    final_error_norm = float(np.linalg.norm(err - x_target))
    return max_u, final_error_norm


PENDULUM_DISCRETIZATION = {
    "q_range": (0.0, 2.0 * np.pi),
    "qdot_range": (-10.0, 10.0),
    "u_range": (-3.0, 3.0),
    "nq": 71,
    "nqdot": 71,
    "nu": 11,
}


def double_integrator():
    return LinearSystem(
        A=np.array([[0, 1], [0, 0]]),
        B=np.array([[0], [1]]),
        C=np.eye(2),
        D=np.zeros((2, 1)),
    )


def double_integrator_example(meshcat, cost_function, convergence_tol, animate=True, plot=True):
    simulator = Simulator(double_integrator())
    options = DynamicProgrammingOptions()

    qbins = np.linspace(-3.0, 3.0, 31)
    qdotbins = np.linspace(-4.0, 4.0, 51)
    state_grid = [set(qbins), set(qdotbins)]

    input_limit = 1.0
    input_grid = [set(np.linspace(-input_limit, input_limit, 9))]
    time_step = 0.01

    Q, Qdot = np.meshgrid(qbins, qdotbins)

    def draw(iteration, _mesh, cost_to_go, policy):
        if iteration % 20 != 0:
            return

        meshcat.PlotSurface(
            "Cost-to-go",
            Q,
            Qdot,
            np.reshape(cost_to_go, Q.shape),
            wireframe=True,
        )
        meshcat.PlotSurface(
            "Policy",
            Q,
            Qdot,
            np.reshape(policy, Q.shape),
            rgba=Rgba(0.3, 0.3, 0.5),
        )

        sleep(0.1)

    def simulate(policy):
        builder = DiagramBuilder()
        plant = builder.AddSystem(double_integrator())

        vi_policy = builder.AddSystem(policy)
        builder.Connect(plant.get_output_port(0), vi_policy.get_input_port(0))
        builder.Connect(vi_policy.get_output_port(0), plant.get_input_port(0))

        visualizer = builder.AddSystem(DoubleIntegratorVisualizer(show=False))
        builder.Connect(plant.get_output_port(0), visualizer.get_input_port(0))

        diagram = builder.Build()
        sim = Simulator(diagram)
        sim.get_mutable_context().SetContinuousState([-10.0, 0.0])
        AdvanceToAndVisualize(sim, visualizer, 10.0)

    if running_as_notebook:
        options.visualization_callback = draw
    options.convergence_tol = convergence_tol
    options.discount_factor = 1.0

    policy, cost_to_go = FittedValueIteration(
        simulator, cost_function, state_grid, input_grid, time_step, options
    )

    J = np.reshape(cost_to_go, Q.shape)
    Pi = extract_policy_grid(policy, Q.shape)
    meshcat.PlotSurface("Cost-to-go", Q, Qdot, J, wireframe=True)

    if animate:
        print("Simulating...")
        simulate(policy)

    if plot:
        fig = plt.figure(1, figsize=(9, 4))
        ax1, ax2 = fig.subplots(1, 2)
        ax1.set_xlabel("q")
        ax1.set_ylabel("qdot")
        ax1.set_title("Cost-to-Go")
        ax2.set_xlabel("q")
        ax2.set_ylabel("qdot")
        ax2.set_title("Policy")
        ax1.imshow(
            J,
            cmap=cm.jet,
            extent=(qbins[0], qbins[-1], qdotbins[-1], qdotbins[0]),
        )
        ax1.invert_yaxis()
        ax2.imshow(
            Pi,
            cmap=cm.jet,
            extent=(qbins[0], qbins[-1], qdotbins[-1], qdotbins[0]),
        )
        ax2.invert_yaxis()
        display(plt.show())


def pendulum_swingup_example(
    meshcat,
    min_time=True,
    animate=True,
    movie_filename=None,
    save_policy_path=None,
    save_cost_path=None,
):
    plant = PendulumPlant()
    simulator = Simulator(plant)
    options = DynamicProgrammingOptions()

    qbins, qdotbins, _ubins, state_grid, input_grid, Q, Qdot = discretize_state_and_input(
        PENDULUM_DISCRETIZATION["q_range"],
        PENDULUM_DISCRETIZATION["qdot_range"],
        PENDULUM_DISCRETIZATION["u_range"],
        PENDULUM_DISCRETIZATION["nq"],
        PENDULUM_DISCRETIZATION["nqdot"],
        PENDULUM_DISCRETIZATION["nu"],
    )
    time_step = 0.01

    options.periodic_boundary_conditions = [
        PeriodicBoundaryCondition(0, 0.0, 2.0 * np.pi),
    ]
    options.discount_factor = 0.999

    meshcat.Delete()
    meshcat.SetProperty("/Background", "visible", False)

    def draw(iteration, _mesh, cost_to_go, policy):
        if iteration % 20 != 0:
            return

        meshcat.PlotSurface(
            "Cost-to-go",
            Q,
            Qdot,
            np.reshape(cost_to_go, Q.shape),
            wireframe=True,
        )
        meshcat.PlotSurface(
            "Policy",
            Q,
            Qdot,
            np.reshape(policy, Q.shape),
            rgba=Rgba(0.3, 0.3, 0.5),
        )

        sleep(0.1)

    def simulate(policy):
        builder = DiagramBuilder()
        pendulum = builder.AddSystem(PendulumPlant())

        wrap = builder.AddSystem(WrapToSystem(2))
        wrap.set_interval(0, 0, 2 * np.pi)
        builder.Connect(pendulum.get_output_port(0), wrap.get_input_port(0))
        vi_policy = builder.AddSystem(policy)
        builder.Connect(wrap.get_output_port(0), vi_policy.get_input_port(0))
        builder.Connect(vi_policy.get_output_port(0), pendulum.get_input_port(0))

        visualizer = builder.AddSystem(PendulumVisualizer(show=False))
        builder.Connect(pendulum.get_output_port(0), visualizer.get_input_port(0))

        diagram = builder.Build()
        sim = Simulator(diagram)
        sim.get_mutable_context().SetContinuousState([0.1, 0.0])
        # Use full simulation time when running as a script
        AdvanceToAndVisualize(
            sim,
            visualizer,
            8.0,
            time_if_running_headless=0.0,
            movie_filename=movie_filename,
        )

    def simulate_and_measure(policy, initial_state, duration=8.0, record_rate=0.01):
        """Simulate closed-loop from initial_state, record control history and return metrics.

        Returns (max_u, final_error_norm)
        """
        builder = DiagramBuilder()
        pendulum = builder.AddSystem(PendulumPlant())

        wrap = builder.AddSystem(WrapToSystem(2))
        wrap.set_interval(0, 0, 2 * np.pi)
        builder.Connect(pendulum.get_output_port(0), wrap.get_input_port(0))
        vi_policy = builder.AddSystem(policy)
        builder.Connect(wrap.get_output_port(0), vi_policy.get_input_port(0))
        builder.Connect(vi_policy.get_output_port(0), pendulum.get_input_port(0))

        # Logger to record control input
        logger = builder.AddSystem(SignalLogger(1))
        builder.Connect(vi_policy.get_output_port(0), logger.get_input_port(0))

        visualizer = builder.AddSystem(PendulumVisualizer(show=False))
        builder.Connect(pendulum.get_output_port(0), visualizer.get_input_port(0))

        diagram = builder.Build()
        sim = Simulator(diagram)
        sim.get_mutable_context().SetContinuousState(initial_state)

        # Advance simulation (no blocking visualization)
        sim.AdvanceTo(duration)

        # Extract logged control signal and compute max magnitude
        data = logger.data()
        if data.size == 0:
            max_u = 0.0
        else:
            max_u = float(np.max(np.abs(data)))

        # Get pendulum subsystem context to read final state
        pendulum_ctx = diagram.GetSubsystemContext(pendulum, sim.get_context())
        x_final = pendulum_ctx.get_continuous_state_vector().CopyToVector()
        # target is theta = pi, theta_dot = 0
        x_target = np.array([np.pi, 0.0])
        err = x_final.copy()
        err[0] = ((err[0] - x_target[0] + np.pi) % (2 * np.pi)) - np.pi
        final_error_norm = float(np.linalg.norm(err - x_target))
        return max_u, final_error_norm

    if running_as_notebook:
        options.visualization_callback = draw

    def min_time_cost(context):
        x = context.get_continuous_state_vector().CopyToVector()
        x[0] = x[0] - np.pi
        if x.dot(x) < 0.05:
            return 0.0
        return 1.0

    def quadratic_regulator_cost(context):
        x = context.get_continuous_state_vector().CopyToVector()
        x[0] = x[0] - np.pi
        u = plant.EvalVectorInput(context, 0).CopyToVector()
        return 2 * x.dot(x) + u.dot(u)

    if min_time:
        cost_function = min_time_cost
        options.convergence_tol = 0.001
    else:
        cost_function = quadratic_regulator_cost
        options.convergence_tol = 0.1

    policy, cost_to_go = FittedValueIteration(
        simulator, cost_function, state_grid, input_grid, time_step, options
    )

    J = np.reshape(cost_to_go, Q.shape)
    Pi = extract_policy_grid(policy, Q.shape)
    meshcat.PlotSurface("Cost-to-go", Q, Qdot, J, wireframe=True)

    if save_cost_path:
        fig_cost = plt.figure(figsize=(6, 4))
        ax_cost = fig_cost.subplots(1, 1)
        ax_cost.set_xlabel("q")
        ax_cost.set_ylabel("qdot")
        ax_cost.set_title("Cost-to-Go")
        ax_cost.imshow(
            J,
            cmap=cm.jet,
            aspect="auto",
            extent=(qbins[0], qbins[-1], qdotbins[-1], qdotbins[0]),
        )
        ax_cost.invert_yaxis()
        fig_cost.tight_layout()
        fig_cost.savefig(save_cost_path, dpi=150)
        plt.close(fig_cost)

    if save_policy_path:
        fig_pi = plt.figure(figsize=(6, 4))
        ax_pi = fig_pi.subplots(1, 1)
        ax_pi.set_xlabel("q")
        ax_pi.set_ylabel("qdot")
        ax_pi.set_title("Policy")
        ax_pi.imshow(
            Pi,
            cmap=cm.jet,
            aspect="auto",
            extent=(qbins[0], qbins[-1], qdotbins[-1], qdotbins[0]),
        )
        ax_pi.invert_yaxis()
        fig_pi.tight_layout()
        fig_pi.savefig(save_policy_path, dpi=150)
        plt.close(fig_pi)

    if animate:
        print("Simulating...")
        simulate(policy)

    fig = plt.figure(figsize=(9, 4))
    ax1, ax2 = fig.subplots(1, 2)
    ax1.set_xlabel("q")
    ax1.set_ylabel("qdot")
    ax1.set_title("Cost-to-Go")
    ax2.set_xlabel("q")
    ax2.set_ylabel("qdot")
    ax2.set_title("Policy")
    ax1.imshow(
        J,
        cmap=cm.jet,
        aspect="auto",
        extent=(qbins[0], qbins[-1], qdotbins[-1], qdotbins[0]),
    )
    ax1.invert_yaxis()
    ax2.imshow(
        Pi,
        cmap=cm.jet,
        aspect="auto",
        extent=(qbins[0], qbins[-1], qdotbins[-1], qdotbins[0]),
    )
    ax2.invert_yaxis()
    display(plt.show())


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Value iteration examples (Drake)")
    parser.add_argument(
        "--example",
        choices=["pendulum", "double-integrator"],
        default="pendulum",
        help="Which example to run",
    )
    parser.add_argument(
        "--min-time",
        action="store_true",
        help="Use min-time cost (pendulum only)",
    )
    parser.add_argument(
        "--no-animate",
        action="store_true",
        help="Disable animation",
    )
    parser.add_argument(
        "--no-keepalive",
        action="store_true",
        help="Exit immediately after completion (disable keep-alive)",
    )
    parser.add_argument(
        "--save-animation",
        default=None,
        help="Save pendulum animation to an HTML file (e.g. pendulum.html)",
    )
    parser.add_argument(
        "--compare-costs",
        action="store_true",
        help="Run min-time and quadratic costs and save policy/cost plots",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save comparison plots",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    meshcat = StartMeshcat()

    if args.example == "double-integrator":
        def min_time_cost(context):
            x = context.get_continuous_state_vector().CopyToVector()
            if x.dot(x) < 0.05:
                return 0.0
            return 1.0

        double_integrator_example(
            meshcat, min_time_cost, convergence_tol=0.001, animate=not args.no_animate
        )
    else:
        if args.compare_costs:
            base = args.output_dir.rstrip("/")
            pendulum_swingup_example(
                meshcat,
                min_time=True,
                animate=False,
                save_policy_path=f"{base}/policy_min_time.png",
                save_cost_path=f"{base}/cost_min_time.png",
            )
            pendulum_swingup_example(
                meshcat,
                min_time=False,
                animate=False,
                save_policy_path=f"{base}/policy_quadratic.png",
                save_cost_path=f"{base}/cost_quadratic.png",
            )
            print("Saved comparison plots to:")
            print(f"- {base}/policy_min_time.png")
            print(f"- {base}/cost_min_time.png")
            print(f"- {base}/policy_quadratic.png")
            print(f"- {base}/cost_quadratic.png")
        else:
            pendulum_swingup_example(
                meshcat,
                min_time=args.min_time,
                animate=not args.no_animate,
                movie_filename=args.save_animation,
            )

    if not args.no_keepalive:
        input("运行完成，按回车退出...")


if __name__ == "__main__":
    main()
