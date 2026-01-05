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


def plot_hist(grads):
    plt.hist(grads.cpu().numpy(), bins=100)
    plt.title("Gradients distribution")
    plt.ylim(0, 1e5)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    #plt.show()
    plt.savefig("asdf.png")


def main():
    print("Loading from gradients.pt")
    grads = torch.load("gradients.pt")
    # Flatten and concat
    grads = torch.cat([g.flatten() for g in grads.values()])

    plot_hist(grads)


if __name__ == "__main__":
    main()
