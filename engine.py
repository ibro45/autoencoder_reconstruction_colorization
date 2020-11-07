import sys
from pathlib import Path

import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader

from data import get_cifar_train_val_test, YCbCrToRGB, GrayscaleToRGB
from model import AutoEncoder

class Engine:
    def __init__(self, num_epochs, batch_size, lr, experiment_name, mode="reconstruction", num_workers=4):
        if torch.cuda.is_available(): 
            torch.backends.cudnn.benchmark = True

        self.lr = lr
        self.batch_size = batch_size
        self.output_dir = f"checkpoints/{experiment_name}"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.mode = mode.lower()
        if self.mode == "reconstruction":
            in_channels, out_channels = 3, 3
        elif self.mode == "colorization":
            in_channels, out_channels = 1, 2
        else:
            raise NotImplemented("Only `colorization` and `reconstruction` modes available.")

        self.num_epochs = num_epochs
        self.model = AutoEncoder(in_channels, out_channels, device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.criterion = nn.MSELoss()

        train_set, val_set, test_set = get_cifar_train_val_test(root="./dataset/", 
                                                                split_ratio=(0.8, 0.1, 0.1))
        self.train_loader = DataLoader(train_set, 
                                       batch_size=batch_size, 
                                       num_workers=num_workers, 
                                       shuffle=True)
        self.val_loader = DataLoader(val_set, 
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     shuffle=False)
        self.test_loader = DataLoader(test_set, 
                                      batch_size=batch_size, 
                                      num_workers=num_workers, 
                                      shuffle=False)

    def _run_epoch(self, stage):
        assert stage in ['train', 'val', 'test']

        is_train = True if stage == 'train' else False

        torch.set_grad_enabled(is_train)
        self.model.train() if is_train else self.model.eval()
        
        loader = getattr(self, f"{stage}_loader")
        running_loss = 0
        for data, _ in loader:
            data = data.to(self.device)
            input = data
            target = data

            if self.mode == "colorization":
                input = data[:, 0:1, ...]  # The first channel of YCbCr is luminance (grayscale)
                target = data[:, 1:, ...]  # The second two channels of YCbCr are color information
            if is_train:
                self.optimizer.zero_grad()

            output = self.model(input)  
            loss = self.criterion(output, target)
            
            if is_train:
                loss.backward()
                self.optimizer.step()  
            
            running_loss += loss.item()

        # save the last batch of the images for the epoch
        if stage in ["val", "test"]:
            self._log_images(data, input, output, stage)

        return running_loss / len(loader)

    def train(self):
        self.train_loss_log = []
        self.val_loss_log = []
        best_val_loss = sys.maxsize

        print("Training started.")
        for i in range(self.num_epochs):
            self.epoch = i + 1
            # Training
            train_loss = self._run_epoch(stage='train')            
            self.train_loss_log.append(train_loss)
            
            # Validation
            val_loss = self._run_epoch(stage='val')     
            self.val_loss_log.append(val_loss)
            
            print(f"\nEpoch: {self.epoch}/{self.num_epochs}.. "
                  f"Training Loss: {train_loss:.6f}   "
                  f"Validation Loss: {val_loss:.6f}")
            
            # Model Checkpoint
            if self.val_loss_log[-1] < best_val_loss:
                new_best = self.val_loss_log[-1]
                print(f'Val loss improved from {best_val_loss:.6f} to {new_best:.6f}, checkpointing.')
                best_val_loss = new_best
                self._save_checkpoint("best")

        self._save_checkpoint("last")

    def test(self, best=True):
        print('Testing...')
        if best:
            self.load_checkpoint(f'{self.output_dir}/best_checkpoint.pth')
        test_loss = self._run_epoch(stage='test')
        print(f'Test loss: {test_loss:.6f}')

    def _save_checkpoint(self, name):
        checkpoint = {
            "epoch": self.epoch,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "model": self.model.state_dict(),
            "model_class_name": type(self.model).__name__,
            "optimizer": self.optimizer.state_dict(),
            "train_loss_log": self.train_loss_log,
            "val_loss_log": self.val_loss_log,
            "mode": self.mode
        }
        torch.save(checkpoint, f'{self.output_dir}/{name}_checkpoint.pth')

    def load_checkpoint(self, checkpoint):
        print(f'Loading checkpoint from {checkpoint}')
        checkpoint = torch.load(checkpoint)
        assert self.mode == checkpoint["mode"], \
            f"Checkpoint was trained for {checkpoint['mode']}, but the current mode is {self.mode}"
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def _log_images(self, data, input, output, stage):
        ycbr_to_rgb = YCbCrToRGB()
        gray_to_rgb = GrayscaleToRGB()

        if self.mode == "colorization":
            output = torch.cat((input, output), dim=1)  # add the luminance channel to the output
            
            data = ycbr_to_rgb(data)
            output = ycbr_to_rgb(output)
            input  = gray_to_rgb(input)

            grid = torch.cat((data, input, output), dim=3)

        else:
            grid = torch.cat((input, output), dim=3)
            grid = ycbr_to_rgb(grid)
        
        grid = torchvision.utils.make_grid(grid, nrow=4)
        
        path = Path(f"{self.output_dir}/{stage}/{self.epoch}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        torchvision.utils.save_image(grid, path)


if __name__ == "__main__":
    engine = Engine(10, 16, 0.001, mode="reconstruction", experiment_name="reconstruction1")
    engine.train()
    engine.test()