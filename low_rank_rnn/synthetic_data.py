import random

import torch


def generate_go_nogo(
    n_samples: int = 1_000, prob: float = 0.7
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns synthetic input signals U, and corresponding expected output vectors O
    for the Go / NoGo task.

    U is of shape (n_samples, 1).
    Every sample is composed of a scalar which can be either:
        * 0 if no stimulus, then the expected action is 0
        * 1 means the expected action is 1 (a yellow square in the original experiment)
        * -1 means the expected action is 0 (a blue square)
    The temporal dynamic is decomposed in 2 phases:
        * first, nothing happens: u_t = 0
        * then, there is a stimulus with noise around 1 or -1: u_t = 0.94

    Each phase lasts for 100 units of time.
    """
    phase_duration = 100
    std = 0.05

    inputs: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    for _ in range(n_samples):
        action = int(random.random() < prob) * 2.0 - 1.0

        mean_u = torch.tensor(
            [0.0 for _ in range(phase_duration)]
            + [action for _ in range(phase_duration)]
        )
        u = torch.normal(mean_u, std=std).unsqueeze(1)
        o = torch.tensor(
            [0.0 for _ in range(phase_duration)]
            + [0.0 if action == -1.0 else 1.0 for _ in range(phase_duration)]
        )

        inputs.append(u)
        outputs.append(o)

    return torch.stack(inputs), torch.stack(outputs)
