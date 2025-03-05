import toml

config_path = "config.toml"
def load_from_config():
    # Load configuration file
    config = toml.load(config_path)
    return config