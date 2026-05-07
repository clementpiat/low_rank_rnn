import pathlib

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from torch import nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, random_split

from low_rank_rnn.rnn import LowRankRNN
from low_rank_rnn.synthetic_data import generate_go_nogo


def train(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    model: LowRankRNN,
    batch_size: int = 8,
    epochs: int = 1,
    lr: float = 1e-4,
    print_every_k_batch: int = 10,
) -> None:
    # dataset & dataloader
    dataset = TensorDataset(inputs, outputs)
    train_dataset, val_dataset = random_split(dataset, (0.8, 0.2))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # loss & optimizer
    loss_fn = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=lr)

    # training loop
    for e in range(epochs):
        print(f"EPOCH {e}")
        model = model.train(True)
        for i, (_input, target) in enumerate(train_loader, start=1):
            out = model.forward(_input)
            optimizer.zero_grad()
            loss = loss_fn(out.squeeze(), target)
            loss.backward()
            optimizer.step()

            if i % print_every_k_batch == 0:
                print(f"\tLoss: {loss.item()}")

        model = model.eval()
        val_loss = []
        with torch.no_grad():
            for _input, target in val_loader:
                out = model.forward(_input)
                val_loss.append(loss_fn(out.squeeze(), target).item())

            print(f"Validation loss: {np.mean(val_loss)}")


def visualize(model: LowRankRNN, prob: float = 0):
    x = []
    y = []

    u, o = generate_go_nogo(n_samples=1, prob=prob)
    x_t = torch.zeros(model.N)
    for u_t in u[0]:
        a, b = model.n(F.tanh(x_t))
        x.append(float(a))
        y.append(float(b))

        x_t = model.m(model.n(F.tanh(x_t))) + model.I(u_t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.set_title(
        f"Evolution of m1 and m2 components over time with target = {int(o[0][-1]) * 2 - 1}"
    )
    ax1.set_xlabel("m1")
    ax1.set_ylabel("m2")
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)

    ax2.set_title(f"Stimulus")
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    action = int(o[0][-1])
    color = "blue" if action == 0 else "yellow"
    patch = ax2.add_patch(Rectangle((5, 5), 2, 2, fc=color))  # out of bounds

    def update(frame: int):
        _x = x[:frame]
        _y = y[:frame]
        ax1.scatter(_x, _y)
        if frame == 100:
            patch.set_x(-1)
            patch.set_y(-1)

    anim = animation.FuncAnimation(fig=fig, func=update, frames=len(x), repeat=True)
    writer = animation.PillowWriter(fps=30)
    output_name = "figures/" + ("go" if int(o[0][-1]) == 1 else "nogo") + ".gif"
    anim.save(pathlib.Path(__file__).parent.resolve() / output_name, writer=writer)


if __name__ == "__main__":
    inputs, outputs = generate_go_nogo()
    model = LowRankRNN(rank=2, neurons=1_000, input_dim=1)
    train(inputs, outputs, model, epochs=3)
    visualize(model, 0)
    visualize(model, 1)
