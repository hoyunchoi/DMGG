import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from DMGG.experiment_setup import get_random_id, save, setup_training
from DMGG.hyperparameter import get_hp
from DMGG.net import count_trainable_params
from path import RESULT_DIR


def main() -> None:
    hp = get_hp()
    model, tracker_callback = setup_training(hp)

    # --------- Ready to train ---------
    exp_id = get_random_id()
    print(f"Ready to train {exp_id}")
    print("Trainable parameters:", count_trainable_params(model))

    result_dir = RESULT_DIR / exp_id
    result_dir.mkdir(parents=True, exist_ok=True)
    hp.save(result_dir)

    # --------- Train ---------
    model.learn(
        total_timesteps=hp.total_timesteps,
        callback=tracker_callback,
        tb_log_name="PPO",
        reset_num_timesteps=not bool(hp.resume),
        progress_bar=hp.progress_bar,
    )

    save(result_dir, model, tracker_callback)


if __name__ == "__main__":
    main()
