import fire

DEFAULT_SCRIPT = "plot_field_size_by_forward_pass"

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
    elif script == "experiment_field_size_vs_depth_thiner":
        from experiments import experiment_field_size_vs_depth_thiner
        experiment_field_size_vs_depth_thiner()
    elif script == "experiment_field_size_by_forward_pass":
        from experiments import experiment_field_size_by_forward_pass
        experiment_field_size_by_forward_pass()
    elif script == "plot_field_size_by_forward_pass":
        from visualization_utils import plot_field_size_by_forward_pass
        plot_field_size_by_forward_pass()
    elif script == "experiment_field_size_by_forward_pass_constant":
        from experiments import experiment_field_size_by_forward_pass_constant
        experiment_field_size_by_forward_pass_constant()




if __name__ == '__main__':
    fire.Fire(run_script)