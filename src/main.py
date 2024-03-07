"""
Top-level script to run all baseline environments and algorithms used in the paper
"Robust Adversarial Reinforcement Learning via Bounded Rationality Curricula"
"""

import argparse
import datetime
import os

from run_experiment import experiment

BASE_RESULTS_DIRECTORY = os.path.dirname(os.path.realpath(__file__)) + "/results"


def main():
    parser = argparse.ArgumentParser(
        "Curriculum Adversarial RL Experiment Runner",
        description="Launches an adversarial RL experiment",
    )

    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument(
        "--algorithm",
        type=str,
        default="qarl",
        choices=[
            "baseline",
            "rarl",
            "mas",
            "sgld",
            "force",
            "qarl_spdl",
            "qarl_linear",
            "qarl_point",
            "qarl_single",
            "fixed_force",
            "fixed_temp",
            "random",
        ],
    )
    parser.add_argument(
        "--domain_name",
        type=str,
        default="cartpole_two_players",
        choices=[
            "acrobot_two_players",
            "ball_in_cup_two_players",
            "cartpole_two_players",
            "cheetah_two_players",
            "hopper_two_players",
            "pendulum_two_players",
            "quadruped_two_players",
            "reacher_two_players",
            "walker_two_players",
        ],
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="balance_vs_adversary",
        choices=[
            "balance_vs_adversary",
            "balance_sparse_vs_adversary",
            "swingup_vs_adversary",
            "swingup_sparse_vs_adversary",
            "balance_vs_adversary",
            "catch_vs_adversary",
            "walk_vs_adversary",
            "run_vs_adversary",
            "hop_vs_adversary",
            "reach_goal_vs_adversary",
            "reach_goal_vs_wind_adversary",
            "easy_vs_adversary",
            "hard_vs_adversary",
        ],
    )
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--bool_render", type=bool, default=False)
    parser.add_argument("--new_adv_max_force", type=float, default=None)
    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument("--n_total_iterations", type=int, default=200)
    parser.add_argument("--n_evaluation_episodes", type=int, default=20)
    args = parser.parse_args()

    exp_name = f"{args.domain_name}-{args.task_name}{datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')}"
    for i in range(args.seeds):
        results_dir = (
            BASE_RESULTS_DIRECTORY + f"/{exp_name}/algorithm___{args.algorithm}/{i}"
        )
        experiment_args = {
            "domain_name": args.domain_name,
            "task_name": args.task_name,
            "horizon": args.horizon,
            "gamma": args.gamma,
            "bool_render": args.bool_render,
            "new_adv_max_force": args.new_adv_max_force,
            "use_cuda": args.use_cuda,
            "n_total_iterations": args.n_total_iterations,
            "n_evaluation_episodes": args.n_evaluation_episodes,
        }
        experiment(
            algorithm=args.algorithm, seed=i, results_dir=results_dir, **experiment_args
        )


if __name__ == "__main__":
    main()
