# test_hydra_env.py
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path=".", config_name="simple_config")
def my_app(cfg: DictConfig):
    print("Hydra successfully loaded the config.")
    print(f"Config content: {cfg}")
    print(f"My variable: {cfg.my_variable}")

if __name__ == "__main__":
    my_app()