"""
Code to analyze the gradient tensor during training.

Provides callback to save gradients to file.

Run this file to plot figures.
"""

import matplotlib.pyplot as plt

import torch
from transformers import TrainerCallback


class SaveGradients(TrainerCallback):
    """
    Save the gradients of the model during training, and exit.
    """

    def __init__(self, save_path="gradients.pt", step_thres=50):
        super().__init__()

        self.save_path = save_path
        self.step_thres = step_thres

    def on_optimizer_step(self, args, state, control, **kwargs):
        if state.global_step >= self.step_thres:
            grads = {
                name: param.grad.cpu()
                for name, param in kwargs["model"].named_parameters()
                if param.grad is not None
            }

            torch.save(grads, self.save_path)
            print("Saved gradients to", self.save_path)

            raise KeyboardInterrupt("Exiting after saving gradients.")


def load_gradients():
    print("Loading from gradients.pt")
    grads = torch.load("gradients.pt")
    # Flatten and concat
    grads = torch.cat([g.flatten() for g in grads.values()])
    return grads


def plot_hist(grads, scale=1, y_max=1e4):
    """
    Histogram of gradients.

    scale: Multiply gradients by this value. Can be used to simulate quantization.
    """
    grads = grads.cpu().numpy() * scale
    plt.hist(grads, bins=100)
    plt.ylim(0, y_max)
    plt.savefig("grads.png")


def main():
    grads = load_gradients()
    plot_hist(grads)


if __name__ == "__main__":
    main()
