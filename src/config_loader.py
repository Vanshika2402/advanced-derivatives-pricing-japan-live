import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError("Config file is empty.")

    return config
