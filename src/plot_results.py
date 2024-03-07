import argparse
import re
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.signal import savgol_filter
from scipy.stats import t as t_distribution

sns.set_theme()
plt.rcParams["axes.xmargin"] = 0
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15


def main():
    parser = argparse.ArgumentParser(
        "Curriculum Adversarial RL Experiment Runner",
        description="Launches an adversarial RL experiment",
    )

    parser.add_argument("--results_path", type=str)
    parser.add_argument("--bool_smooth_data", type=bool, default=False)
    parser.add_argument("--bool_show_variance", type=bool, default=True)
    parser.add_argument("--bool_only_needed", type=bool, default=True)
    args = parser.parse_args()

    # Graph Layout
    FN_SPLIT = args.results_path.split("/")[-1].split("-")
    DOMAIN_NAME = FN_SPLIT[0]
    try:
        TASK_NAME = FN_SPLIT[1]
        TITLE = f"{DOMAIN_NAME} - {TASK_NAME}"
    except:
        TITLE = ""
    COLORS = {
        "baseline": "green",
        "rarl": "blue",
        "qarl": "red",
        "mas": "orange",
        "sgld": "brown",
        "fixed_temperature": "purple",
        "force": "purple",
        "linear": "olive",
        "default": "black",
    }
    LABELS = {
        "baseline": "SAC",
        "rarl": "RARL",
        "qarl": "QARL",
        "mas": "CAT",
        "sgld": "SAC MixedNE-LD",
        "force": "Force Curriculum",
        "linear": "Linear Curriculum",
        "default": "algorithm",
    }
    ORDER = {
        "baseline": 5,
        "rarl": 1,
        "qarl": 0,
        "mas": 3,
        "sgld": 2,
        "force": 4,
        "linear": -1,
        "default": -1,
    }

    NEEDED_PLOTS = [
        "mean_reward_first_metric",
        "mean_reward_second_metric",
        "mean_reward_robustness",
        "mean_reward_vs_adversary_per_iteration",
        "mean_reward_without_adversary_per_iteration",
        "mean_reward_vs_worst_adversary",
        "mean_reward_without_adversary",
        "teacher_mean_temp_per_iteration",
    ]

    # Metrics
    if DOMAIN_NAME in ("Cheetah", "Walker", "Hopper", "Quadruped"):
        FIRST_METRIC = "Mass of Torso"
        SECOND_METRIC = "Friction Coefficient"
    elif DOMAIN_NAME == "Cartpole":
        FIRST_METRIC = "Mass of Pole"
        SECOND_METRIC = "Mass of Cart"
    elif DOMAIN_NAME == "Reacher":
        FIRST_METRIC = "Mass of Arm"
        SECOND_METRIC = "Mass of Hand"
    elif DOMAIN_NAME == "Ball in Cup":
        FIRST_METRIC = "Mass of Ball"
        SECOND_METRIC = "Mass of Cup"
    elif DOMAIN_NAME == "Pendulum":
        FIRST_METRIC = "Mass of Pole"
        SECOND_METRIC = "Mass of Pendulum"
    elif DOMAIN_NAME == "Acrobot":
        FIRST_METRIC = "Mass of Upper Arm"
        SECOND_METRIC = "Mass of Lower Arm"
    else:
        FIRST_METRIC = "First Metric"
        SECOND_METRIC = "Second Metric"

    def get_npy_list_from_paths(paths_list):
        """
        Get list of npy arrays from list of PosixPaths
        """
        npy_list = [np.load(path) for path in paths_list]
        return npy_list

    def calculate_mean_std(npy_list):
        # npy_list = list of npy arrays
        mean = np.mean(npy_list, axis=0)
        std = np.std(npy_list, axis=0)
        return mean, std

    def smooth_npy(array, window_size=15, order=3):
        """
        Smooth npy array with savgol filter
        """
        try:
            smooth_npy = savgol_filter(array, window_size, order)
        except:
            raise ValueError("Experiment is too short for smoothing!")
        return smooth_npy

    def calculate_95_confidence_interval(mean, std, n_samples):
        # using student's t-distribution as the sample size is small

        # Student T distribution
        confidence_lvl = 0.95
        degrees_of_freedom = n_samples - 1
        t_crit = np.abs(
            t_distribution.ppf((1 - confidence_lvl) / 2, degrees_of_freedom)
        )

        # Normal distribution
        # t_crit = 0.96
        lower_bound = mean - t_crit * (std / np.sqrt(n_samples))
        upper_bound = mean + t_crit * (std / np.sqrt(n_samples))
        return lower_bound, upper_bound

    def calculate_bounds(npy_list, bool_smooth_data):
        # npy_list = list of npy arrays
        mean, std = calculate_mean_std(npy_list)
        if mean.shape[-1] == 2:  # critic double data
            mean = mean.mean(-1)
            std = std.mean(-1)
        lower_bound, upper_bound = calculate_95_confidence_interval(
            mean, std, len(npy_list)
        )
        if bool_smooth_data:
            mean = smooth_npy(mean)
            lower_bound = smooth_npy(lower_bound)
            upper_bound = smooth_npy(upper_bound)

        mean_ci_bounds = {
            "mean": mean,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
        return mean_ci_bounds

    def calculate_bounds_changing_plot(npy_list, bool_smooth_data):
        # npy_list = list of npy arrays
        mean, std = calculate_mean_std(npy_list)
        lower_bound, upper_bound = calculate_95_confidence_interval(
            mean[:, 1], std[:, 1], len(npy_list)
        )
        if bool_smooth_data:
            mean[:, 1] = savgol_filter(mean[:, 1], window_length=5, polyorder=3)
            lower_bound[:] = savgol_filter(lower_bound[:], window_length=5, polyorder=3)
            upper_bound[:] = savgol_filter(upper_bound[:], window_length=5, polyorder=3)

        mean_ci_bounds = {
            "x": mean[:, 0],
            "mean": mean[:, 1],
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }
        return mean_ci_bounds

    def rearrange_custom(data_list):
        data_list_rearranged = [0] * len(data_list)
        for data in data_list:
            for category in ORDER.keys():
                if category in data["category"]:
                    category_order = ORDER[category]
                    data_list_rearranged[category_order] = data
                    break
        return data_list_rearranged

    def gen_training_plot(
        category_mean_ci_bounds_list,
        results_path,
        bool_smooth_data,
        bool_show_variance,
        title,
    ):
        fig, ax = plt.subplots()
        for category_mean_ci_bounds in category_mean_ci_bounds_list:
            x = range(1, len(category_mean_ci_bounds["mean"]) + 1)
            category_label = LABELS["default"]
            for category in LABELS.keys():
                if category in category_mean_ci_bounds["category"]:
                    category_label = LABELS[category]
            category_color = COLORS["default"]
            for category in COLORS.keys():
                if category in category_mean_ci_bounds["category"]:
                    category_color = COLORS[category]

            ax.plot(
                x,
                category_mean_ci_bounds["mean"],
                color=category_color,
                linewidth=1,
                label=category_label,
            )
            if bool_show_variance:
                ax.fill_between(
                    x,
                    category_mean_ci_bounds["lower_bound"],
                    category_mean_ci_bounds["upper_bound"],
                    color=category_color,
                    alpha=0.2,
                )

        category_mean_ci_bounds_0 = category_mean_ci_bounds_list[0]
        fn = f"{category_mean_ci_bounds_0['experiment_phase']}_{category_mean_ci_bounds_0['data_key']}"
        # xlabel
        if "iteration" in category_mean_ci_bounds_0["data_key"]:
            xlabel = "Iteration"
        elif "training_step" or "progression" in category_mean_ci_bounds_0["data_key"]:
            xlabel = "Transition"
        else:
            xlabel = ""
        # ylabel
        if "vs" in category_mean_ci_bounds_0["data_key"]:
            ylabel = "Return vs Adversary"
        elif "without" in category_mean_ci_bounds_0["data_key"]:
            ylabel = "Return"
        elif "temp" in category_mean_ci_bounds_0["data_key"]:
            ylabel = "Temperature"
        else:
            ylabel = fn

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="lower right", prop={"size": 15})

        if bool_smooth_data:
            plt.savefig(fname=str(results_path) + f"/smoothed_{fn}.png")
        else:
            plt.savefig(fname=str(results_path) + f"/{fn}.png")
        plt.close("all")

    def gen_eval_plot(category_data_eval_list, results_path, title):
        fig, ax = plt.subplots(figsize=(10, 5))
        data = []
        labels = []
        for i, category_data_eval in enumerate(category_data_eval_list):
            data.append(np.array(category_data_eval["data"]))
            category_label = LABELS["default"]
            for category in LABELS.keys():
                if category in category_data_eval["category"]:
                    category_label = LABELS[category]
            y_mean = np.mean(data[-1])
            category_label += f"\n({round(y_mean)})"
            labels.append(category_label)

        ax.boxplot(data, labels=labels, whis=[0, 100])
        for i, category_data_eval in enumerate(category_data_eval_list):
            x = np.random.normal(i + 1, 0.015, size=len(data))
            category_color = COLORS["default"]
            for category in COLORS.keys():
                if category in category_data_eval["category"]:
                    category_color = COLORS[category]
            ax.scatter(x, data, color=category_color, marker=".", alpha=0.5, s=100)

        category_data_eval_0 = category_data_eval_list[0]
        fn = category_data_eval_0["data_key"]
        if "vs" in category_data_eval_0["data_key"]:
            ylabel = "Return vs Adversary"
        elif "without" in category_data_eval_0["data_key"]:
            ylabel = "Return"
        else:
            ylabel = fn

        # ax.set_xlabel("Algorithm")
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_title(title, fontsize=15)
        plt.xticks(rotation=20)
        fn = f"{title}-{ylabel}"
        plt.savefig(fname=str(results_path) + f"/{fn}.png", bbox_inches="tight")
        plt.close("all")

    def gen_changing_plot(
        category_mean_ci_bounds_list,
        results_path,
        bool_smooth_data,
        bool_show_variance,
        title,
    ):
        fig, ax = plt.subplots()
        for category_mean_ci_bounds in category_mean_ci_bounds_list:
            x = category_mean_ci_bounds["x"]
            category_label = LABELS["default"]
            for category in LABELS.keys():
                if category in category_mean_ci_bounds["category"]:
                    category_label = LABELS[category]
            category_color = COLORS["default"]
            for category in COLORS.keys():
                if category in category_mean_ci_bounds["category"]:
                    category_color = COLORS[category]
            ax.plot(
                x,
                category_mean_ci_bounds["mean"],
                color=category_color,
                linewidth=1,
                label=category_label,
            )
            if bool_show_variance:
                ax.fill_between(
                    x,
                    category_mean_ci_bounds["lower_bound"],
                    category_mean_ci_bounds["upper_bound"],
                    color=category_color,
                    alpha=0.2,
                )

        category_mean_ci_bounds_0 = category_mean_ci_bounds_list[0]
        fn = f"{category_mean_ci_bounds_0['experiment_phase']}_{category_mean_ci_bounds_0['data_key']}"
        if "first_metric" in category_mean_ci_bounds_0["data_key"]:
            xlabel = FIRST_METRIC
        elif "second_metric" in category_mean_ci_bounds_0["data_key"]:
            xlabel = SECOND_METRIC
        ylabel = "Return"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="lower right", prop={"size": 15})

        if bool_smooth_data:
            plt.savefig(fname=str(results_path) + f"/smoothed_{fn}.png")
        else:
            plt.savefig(fname=str(results_path) + f"/{fn}.png")
        plt.close("all")

    def gen_heatmap(
        category_mean_list,
        results_path,
        title,
    ):
        category_mean_0 = category_mean_list[0]
        x = category_mean_0["data"][0, :, 1]
        y = category_mean_0["data"][:, 0, 0]
        all_data = np.array(
            [category_mean["data"] for category_mean in category_mean_list]
        )
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        v_min = np.min(all_data[:, :, :, 2])
        v_max = np.max(all_data[:, :, :, 2])
        aspect_ratio = (x_max - x_min) / (y_max - y_min)
        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(
            fig,
            111,
            nrows_ncols=(1, len(category_mean_list)),
            axes_pad=0.5,
            share_all=True,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="10%",
            cbar_pad=0.25,
        )
        for i, ax in enumerate(grid):
            category_mean = category_mean_list[i]
            category_label = LABELS["default"]
            for category in LABELS.keys():
                if category in category_mean["category"]:
                    category_label = LABELS[category]
            data = category_mean["data"][:, :, 2]
            data_mean = np.mean(data)
            im = ax.imshow(
                data,
                cmap="coolwarm",
                interpolation="nearest",
                extent=[x_min, x_max, y_max, y_min],
                vmin=v_min,
                vmax=v_max,
            )
            xlabel = SECOND_METRIC
            ylabel = FIRST_METRIC
            ax.set_xlabel(xlabel, fontsize=20)
            ax.set_ylabel(ylabel, fontsize=20)
            ax.set_title(category_label + f" ({round(data_mean)})", fontsize=20)
            ax.set_aspect(aspect_ratio)
            ax.grid(False)

        cbar = ax.cax.colorbar(im)
        cbar.set_label("Return", rotation=270, labelpad=25, fontsize=20)
        ax.cax.toggle_label(True)
        fig.suptitle(title, fontsize=20)
        fn = f"{title}-robustness"
        plt.savefig(fname=str(results_path) + f"/{fn}.png")
        plt.close("all")

    def gen_data_key_paths(results_path, bool_only_needed):
        results_path = Path(results_path)
        data_key_paths = {}
        for npy_file_path in results_path.glob("**/*.npy"):
            if "exclude" in str(npy_file_path):  # use "exclude" to remove certain runs
                continue
            data_key_npy = re.split("exp_\d*_", npy_file_path.name)[-1]
            data_key = data_key_npy.split(".npy")[0]  # e.g. adv_temp_progression
            # Only generate required plots
            if bool_only_needed:
                if data_key not in NEEDED_PLOTS:
                    continue
            data_category_list = re.findall(
                "(?<=___)(.*?)(?=/)", str(npy_file_path)
            )  # e.g. ["rarl", "sac"]
            data_category = "_".join(data_category_list)  # e.g. rarl_sac
            try:
                data_key_paths[data_key][data_category].append(npy_file_path)
            except:
                try:
                    data_key_paths[data_key][data_category] = [npy_file_path]
                except:
                    data_key_paths[data_key] = {data_category: [npy_file_path]}
        return data_key_paths

    def gen_graphs(
        results_path, bool_smooth_data, bool_show_variance, bool_only_needed, title
    ):
        data_key_paths = gen_data_key_paths(results_path, bool_only_needed)
        for data_key in data_key_paths.keys():
            category_0 = list(data_key_paths[data_key].keys())[0]
            npy_file_path_0 = data_key_paths[data_key][category_0][0]
            experiment_phase = npy_file_path_0.parent.absolute().name

            if "Training" in experiment_phase:
                category_mean_ci_bounds_list = []
                for category in list(data_key_paths[data_key].keys()):
                    category_data = data_key_paths[data_key][category]
                    category_data_npy = get_npy_list_from_paths(category_data)

                    category_mean_ci_bounds = calculate_bounds(
                        category_data_npy, bool_smooth_data
                    )
                    category_mean_ci_bounds["data_key"] = data_key
                    category_mean_ci_bounds["category"] = category
                    category_mean_ci_bounds["experiment_phase"] = experiment_phase
                    category_mean_ci_bounds_list.append(category_mean_ci_bounds)
                gen_training_plot(
                    rearrange_custom(category_mean_ci_bounds_list),
                    results_path,
                    bool_smooth_data,
                    bool_show_variance,
                    title,
                )

            elif "Evaluation" in experiment_phase:
                if ("changing" in data_key) or ("metric" in data_key):
                    # changing just one metric
                    category_mean_ci_bounds_list = []
                    for category in list(data_key_paths[data_key].keys()):
                        category_data = data_key_paths[data_key][category]
                        category_data_npy = get_npy_list_from_paths(category_data)
                        category_mean_ci_bounds = calculate_bounds_changing_plot(
                            category_data_npy, bool_smooth_data
                        )
                        category_mean_ci_bounds["data_key"] = data_key
                        category_mean_ci_bounds["category"] = category
                        category_mean_ci_bounds["experiment_phase"] = experiment_phase
                        category_mean_ci_bounds_list.append(category_mean_ci_bounds)
                    gen_changing_plot(
                        rearrange_custom(category_mean_ci_bounds_list),
                        results_path,
                        bool_smooth_data,
                        bool_show_variance,
                        title,
                    )
                elif "robustness" in data_key:
                    category_mean_list = []
                    for category in list(data_key_paths[data_key].keys()):
                        category_data = data_key_paths[data_key][category]
                        category_data_npy = get_npy_list_from_paths(category_data)
                        category_mean_arr = np.copy(category_data_npy[0])
                        category_mean_arr[:, :, 2] = np.mean(category_data_npy, axis=0)[
                            :, :, 2
                        ]
                        category_mean = {"data": category_mean_arr}
                        category_mean["data_key"] = data_key
                        category_mean["category"] = category
                        category_mean["experiment_phase"] = experiment_phase
                        category_mean_list.append(category_mean)
                    gen_heatmap(
                        rearrange_custom(category_mean_list),
                        results_path,
                        title,
                    )
                else:
                    # normal evaluation whisker plot
                    category_data_eval_list = []
                    for category in list(data_key_paths[data_key].keys()):
                        category_data = data_key_paths[data_key][category]
                        category_data_npy = get_npy_list_from_paths(category_data)
                        category_data_eval = {"data": category_data_npy}
                        category_data_eval["data_key"] = data_key
                        category_data_eval["category"] = category
                        category_data_eval["experiment_phase"] = experiment_phase
                        category_data_eval_list.append(category_data_eval)
                    gen_eval_plot(
                        rearrange_custom(category_data_eval_list), results_path, title
                    )

    gen_graphs(
        results_path=args.results_path,
        bool_smooth_data=args.bool_smooth_data,
        bool_show_variance=args.bool_show_variance,
        bool_only_needed=args.bool_only_needed,
        title=TITLE,
    )


if __name__ == "__main__":
    main()
