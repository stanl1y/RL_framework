from off_policy import off_policy_training_stage
from collect_expert import collect_expert


def get_main_stage(config):
    if config.main_stage_type == "off_policy":
        return off_policy_training_stage(config)
    elif config.main_stage_type == "collect_expert":
        return collect_expert(config)
    else:
        raise TypeError(
            f"training stage type : {config.training_stage_type} not supported"
        )
