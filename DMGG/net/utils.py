from stable_baselines3.common.base_class import BaseAlgorithm


def count_trainable_params(model: BaseAlgorithm) -> int:
    seen, total = set(), 0
    for group in model.policy.optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total
