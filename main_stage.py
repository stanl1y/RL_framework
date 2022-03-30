from off_policy import off_policy_training_stage

def get_main_stage(config):
    if config.main_stage_type=="off_policy":
        return off_policy_training_stage(config)
    else:
        raise TypeError(f"training stage type : {config.training_stage_type} not supported")