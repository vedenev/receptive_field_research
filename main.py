import fire

DEFAULT_SCRIPT = "scripts_list"


def run_script(script: str = DEFAULT_SCRIPT) -> None:

    """
    Type python3 main.py to get list of script names
    """

    if script == DEFAULT_SCRIPT:
        from utils import print_scripts_list
        print_scripts_list()
    elif script == "experiment_field_size_vs_depth":
        from experiments import experiment_field_size_vs_depth
        experiment_field_size_vs_depth()
    elif script == "experiment_field_size_vs_depth_res":
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
    elif script == 'experiment_field_size_by_forward_pass_for_shifted':
        from experiments import experiment_field_size_by_forward_pass_for_shifted
        experiment_field_size_by_forward_pass_for_shifted()
    elif script == "plot_field_size_by_forward_pass":
        from visualization_utils import plot_field_size_by_forward_pass
        plot_field_size_by_forward_pass()
    elif script == "experiment_field_size_by_forward_pass_constant":
        from experiments import experiment_field_size_by_forward_pass_constant
        experiment_field_size_by_forward_pass_constant()
    elif script == "experiment_field_size_resnet50":
        from experiments import experiment_field_size_resnet50
        experiment_field_size_resnet50()
    elif script == "experiment_field_size_vs_depth_res_decomposed_init":
        from experiments import experiment_field_size_vs_depth_res_decomposed_init
        experiment_field_size_vs_depth_res_decomposed_init()
    elif script == "experiment_field_size_by_forward_pass_decomposed_init":
        from experiments import experiment_field_size_by_forward_pass_decomposed_init
        experiment_field_size_by_forward_pass_decomposed_init()
    elif script == "plot_field_size_by_forward_pass_decomposed_init":
        from visualization_utils import plot_field_size_by_forward_pass_decomposed_init
        plot_field_size_by_forward_pass_decomposed_init()
    elif script == "show_dataset":
        from visualization_utils import show_dataset
        show_dataset()
    elif script == "experiment_field_size_vs_depth_dot_circular":
        from experiments import experiment_field_size_vs_depth_dot_circular
        experiment_field_size_vs_depth_dot_circular()
    elif script == "experiment_field_size_resnet50_by_forward_pass":
        from experiments import experiment_field_size_resnet50_by_forward_pass
        experiment_field_size_resnet50_by_forward_pass()
    elif script == "plot_field_size_resnet":
        from visualization_utils import plot_field_size_resnet
        plot_field_size_resnet()
    elif script == "experiment_field_size_by_forward_pass_constant_output_image":
        from experiments import experiment_field_size_by_forward_pass_constant_output_image
        experiment_field_size_by_forward_pass_constant_output_image()
    else:
        print("unexpected script name, type python3 main.py to get help")


if __name__ == '__main__':
    fire.Fire(run_script)