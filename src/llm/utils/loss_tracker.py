from pathlib import Path
import matplotlib.pyplot as plt

def moving_average(values, window_size: int):
    if len(values) < window_size:
        return values
    return [
        sum(values[i - window_size:i]) / window_size
        for i in range(window_size, len(values) + 1)
    ]

class LossTracker:
    def __init__(self, save_dir: Path):
        self.train_steps = []
        self.train_losses = []
        self.val_steps = []
        self.val_losses = []

        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def update_train(self, step: int, loss: float):
        self.train_steps.append(step)
        self.train_losses.append(loss)

    def update_val(self, step: int, loss: float):
        self.val_steps.append(step)
        self.val_losses.append(loss)

    def plot(self, filename: str = "loss.png", smooth_window: int = 10):
        if not self.train_losses and not self.val_losses:
            return

        plt.figure()

        # -------- train --------
        if self.train_losses:
            # plt.plot(
            #     self.train_steps,
            #     self.train_losses,
            #     alpha=0.3,
            #     label="train_loss (raw)",
            # )

            if len(self.train_losses) >= smooth_window:
                smooth_train = moving_average(self.train_losses, smooth_window)
                smooth_steps = self.train_steps[smooth_window - 1:]
                plt.plot(
                    smooth_steps,
                    smooth_train,
                    label=f"train_loss (smooth={smooth_window})",
                )

        # -------- val --------
        if self.val_losses:
            # plt.plot(
            #     self.val_steps,
            #     self.val_losses,
            #     alpha=0.3,
            #     label="val_loss (raw)",
            # )

            if len(self.val_losses) >= smooth_window:
                smooth_val = moving_average(self.val_losses, smooth_window)
                smooth_steps = self.val_steps[smooth_window - 1:]
                plt.plot(
                    smooth_steps,
                    smooth_val,
                    label=f"val_loss (smooth={smooth_window})",
                )

        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True)

        plt.savefig(self.save_dir / filename)
        plt.close()