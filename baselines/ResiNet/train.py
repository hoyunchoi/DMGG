import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from baselines.ResiNet.experiment_setup import save, setup_training
from baselines.ResiNet.hyperparameter import RESINET_RESULT_DIR, get_hp
from DMGG.experiment_setup import get_random_id
from DMGG.net import count_trainable_params


def main() -> None:
    hp = get_hp()
    model, tracker_callback = setup_training(hp)

    # --------- Ready to train ---------
    experiment_id = get_random_id()
    print(f"Ready to train {experiment_id}")
    print("Trainable parameters:", count_trainable_params(model))

    result_dir = RESINET_RESULT_DIR / experiment_id
    result_dir.mkdir(parents=True, exist_ok=False)
    hp.save(result_dir)

    # --------- Train ---------
    model.learn(
        total_timesteps=hp.total_timesteps,
        callback=tracker_callback,
        tb_log_name="ResiNet",
        reset_num_timesteps=not bool(hp.resume),
        progress_bar=hp.progress_bar,
    )

    save(result_dir, model, tracker_callback)


if __name__ == "__main__":
    main()
