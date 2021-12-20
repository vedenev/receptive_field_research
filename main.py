import fire

DEFAULT_SCRIPT = "plot_field_size_vs_depth"

def run_script(script : str = DEFAULT_SCRIPT):
    if script == "experiment_field_size_vs_depth":
        from experiments import experiment_field_size_vs_depth
        experiment_field_size_vs_depth()
    if script == "experiment_field_size_vs_depth_res":
            from experiments import experiment_field_size_vs_depth_res
            experiment_field_size_vs_depth_res()
    elif script == "plot_field_size_vs_depth":
        from visualization_utils import plot_field_size_vs_depth
        plot_field_size_vs_depth()


if __name__ == '__main__':
    fire.Fire(run_script)