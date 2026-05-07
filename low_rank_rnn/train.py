import pathlib

import numpy as np
import torch
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


def visualize(model: LowRankRNN, output_name: str):
    pass



if __name__ == "__main__":
    inputs, outputs = generate_go_nogo()
    model = LowRankRNN(rank=2, neurons=1_000, input_dim=1)
    train(inputs, outputs, model, epochs=3)
