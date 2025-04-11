def config_dataset(config):
    """
    Update configuration parameters based on the dataset.

    Args:
        config (dict): Input configuration dictionary.

    Returns:
        dict: Updated configuration with dataset-specific values.
    """
    if config["dataset"] == 'histo':
        config["topK"] = -1
        config["n_class"] = 6
    elif config["dataset"] == 'cifar':
        config["topK"] = -1
        config["n_class"] = 10
    elif config["dataset"] == 'RadImageNet-CT':
        config["topK"] = -1
        config["n_class"] = 16
    return config


def get_config():
    """
    Return the default configuration for training.

    Returns:
        dict: Configuration dictionary.
    """
    config = {
        "info": "[ACIRwoOOD]",
        "resize_size": 224,
        "batch_size": 256,
        "dataset": "RadImageNet-CT",
        "epoch": 150,
        "test_freq": 1,
        "bit_list": [8, 16, 64, 128, 512],
        "gpus": [0, 1],
        "device": "cuda",
        "save_path": "save/ACIR_woOOD/"
    }
    config = config_dataset(config)
    return config
