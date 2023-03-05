import logging
import os
from logging import handlers
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd


CODE_ROOT = Path.cwd()
RESULT_FOLDER = CODE_ROOT.joinpath(CODE_ROOT, 'result')

TRAIN_INPUT_MAT = CODE_ROOT.joinpath('train_data.mat')  # should be in the same folder with this .py file
TRAIN_RATIO = 0.8  # the ratio of training set
EPOCHS = 100
USE_CPU = True

BATCH_SIZE = 32
WINDOW_SIZE = 20
CONV_UNITS = 32
DROP_RATE = 0.1
FC1_UNITS = 32

DEVICE = torch.device("cpu")
if not USE_CPU and torch.cuda.is_available():
    DEVICE = torch.device("cuda")


def to_tensor(arr):
    return torch.Tensor(arr).to(DEVICE)


def init_logger(log_dir):
    fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    logger = logging.getLogger()
    format_str = logging.Formatter(fmt)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = handlers.RotatingFileHandler(filename=os.path.join(log_dir, 'log.txt'), mode='w', encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)
    return logger, sh, th


class MyLogger(object):
    def __init__(self, log_dir):
        self.logger, self.sh, self.th = init_logger(log_dir)

    def info(self, msg):
        self.logger.info(msg)

    def close(self):
        self.logger.handlers.clear()
        del self.sh, self.th
        logging.shutdown()


class MyConvNet(nn.Module):

    def __init__(self, n_units_in: int = None, n_units_out: int = None, suggest=None):
        super().__init__()

        self.suggest = {
            'n_units_in': n_units_in,
            'n_units_out': n_units_out,
            'conv_units': CONV_UNITS,
            'drop_out': DROP_RATE,
            'fc1_units': FC1_UNITS
        } if suggest is None else suggest

        self.n_units_out = self.suggest['n_units_out']
        self.n_units_in = self.suggest['n_units_in']
        self.conv_layer = nn.Conv1d(self.n_units_in, self.suggest['conv_units'], 1)
        self.dropout = nn.Dropout(p=self.suggest['drop_out'])
        self.fc1 = nn.Linear(self.suggest['conv_units'], self.suggest['fc1_units'])
        self.fc2 = nn.Linear(self.suggest['fc1_units'], self.n_units_out)

    def forward(self, x):
        # CNN in Section 3.2.2: one hidden layer (conv_layer, relu, fc1) and one output layer (fc2)
        n_batch = x.shape[0]
        output = self.conv_layer(x.view(n_batch, self.n_units_in, -1)).view(n_batch, -1)
        output = self.dropout(output)
        output = F.relu(output)
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output.view(-1, 1, self.n_units_out)


class Wrapper(nn.Module):
    def __init__(self, model, wrapper_name='wrapper'):
        super().__init__()
        self.model = model
        self.wrapper_name = wrapper_name

    def call(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def get_basic_model(self):
        model = self.model
        while isinstance(model, Wrapper):
            model = model.model
        return model


class ResidualWrapper(Wrapper):
    # ResNet in Section 3.1
    def __init__(self, model):
        super().__init__(model, wrapper_name='residual')

    def forward(self, x, *args, **kwargs):
        delta = self.model(x, *args, **kwargs)
        inputs = x[:, -delta.shape[1]:, :delta.shape[2]]
        return inputs + delta


class AutoRegressiveWrapper(Wrapper):
    # Autoregressive process in Section 3.3
    def __init__(self, model):
        super().__init__(model, wrapper_name='auto_regressive')

    def forward(self, x, *args, **kwargs):
        basic_model = self.get_basic_model()
        n_states = basic_model.n_units_out
        init_states = x[:, :1, :n_states]
        x_external = x[:, :, n_states:]
        previous_x = torch.cat((init_states, x_external[:, :1, :]), 2)
        n_steps = x.shape[1]

        predictions = None
        for step in range(n_steps):
            next_state = self.model(previous_x, *args, **kwargs)
            predictions = torch.cat((predictions, next_state), 1) if predictions is not None else next_state[:]
            if step == n_steps - 1:
                break
            external = x_external[:, step + 1:step + 2, :]
            previous_x = torch.cat((previous_x[:, 1:, :n_states], next_state), 1)
            previous_x = torch.cat((previous_x, external), 2)
        return predictions


class MyDataset(Dataset):

    def __init__(self, src: Path, window_size: int, show_data: bool = False) -> None:
        super().__init__()
        self.src = src
        self.window_size = window_size

        if not self.src.exists():
            print('Wrong data source. Path does not exist.')
            return

        data = loadmat(str(self.src))
        # data_x features: z_k-1, u_k, du_k; data_y feature: z_k (Section 4.1)
        self.data, self.target = data['data_x'], data['data_y']

        if show_data:
            for i in range(4):
                plt.subplot(2, 2, i + 1)
                if i <= 2:
                    plt.title(f'X Feature {i + 1}')
                    for j in range(self.data.shape[0]):
                        plt.plot(self.data[j, :, i])
                else:
                    plt.title('Y')
                    for j in range(self.target.shape[0]):
                        plt.plot(self.target[j, :])
            plt.show()

        self.data, self.target = to_tensor(self.data).float(), to_tensor(self.target).float()
        # reshape data_x, data_y to keep the shape of [n_batch, sequence_length, features]
        if len(self.data.shape) == 2:
            self.data = self.data[:, :, None]  # for the case sequence=1
        if len(self.target.shape) == 2:
            self.target = self.target[:, :, None]  # for the case sequence=1

        self.time_steps, self.n_units_in = self.data.shape[1:]
        self.n_units_out = self.target.shape[-1]

        data_, target_ = None, None
        for start in range(self.data.shape[1]-self.window_size):
            dat_window = self.data[:, start:start+self.window_size, :]
            tar_window = self.target[:, start:start+self.window_size, :]
            if data_ is None:
                data_ = dat_window[:]
                target_ = tar_window[:]
            else:
                data_ = torch.cat((data_, dat_window), dim=0)
                target_ = torch.cat((target_, tar_window), dim=0)
        if data_ is None:
            print('Wrong window size')
            return
        self.data, self.target = data_[:], target_[:]
        torch.save({'data': self.data, 'target': self.target}, RESULT_FOLDER.joinpath('dataset.pt'))

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)


def train_model(model, optimizer, train_loader):
    model.train()
    loss_records = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE).float(), target.to(DEVICE).float()
        optimizer.zero_grad()
        pred = model(data)
        loss = F.mse_loss(pred, target)
        loss_records.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.average(loss_records)


def eval_model(model, eval_loader):
    model.eval()
    loss = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(eval_loader):
            data, target = data.to(DEVICE).float(), target.to(DEVICE).float()
            pred = model(data)
            loss.append(F.mse_loss(pred, target).item())
    return np.average(loss)


if __name__ == '__main__':
    print(CODE_ROOT)
    if not RESULT_FOLDER.exists():
        os.mkdir(RESULT_FOLDER)

    ds = MyDataset(src=TRAIN_INPUT_MAT, window_size=WINDOW_SIZE, show_data=False)
    n_train = int(len(ds) * TRAIN_RATIO)
    n_eval = len(ds) - n_train
    train_ds, eval_ds = torch.utils.data.random_split(ds, [n_train, n_eval])
    train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(dataset=eval_ds, batch_size=BATCH_SIZE, shuffle=True)
    print('Dataset loaded. Now training...')

    model = MyConvNet(n_units_in=ds.n_units_in, n_units_out=ds.n_units_out).to(DEVICE)
    model = ResidualWrapper(model).to(DEVICE)
    model = AutoRegressiveWrapper(model).to(DEVICE)
    optimizer = torch.optim.Adam(model.get_basic_model().parameters(), 1e-4)

    save = {'train_loss': [], 'eval_loss': [], 'total_loss': []}
    print(f"Epoch {EPOCHS}\n-------------------------------")
    for epoch in range(EPOCHS):
        train_loss = train_model(model, optimizer, train_loader)
        save['train_loss'].append(train_loss)
        eval_loss = eval_model(model, eval_loader)
        save['eval_loss'].append(eval_loss)
        print(f'Train Loss: {train_loss:>7f}, Eval Loss: {eval_loss:.4f}, [{epoch+1:>5d}/{EPOCHS:>5d}]')

        total_loss = (train_loss + eval_loss) / 2

        if len(save['total_loss']) == 0:
            save['total_loss'].append(total_loss)
            continue

        save_path = RESULT_FOLDER.joinpath(f'model.pth')
        if total_loss < min(save['total_loss']):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.get_basic_model().state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'suggest': model.get_basic_model().suggest,
                'n_units_in': ds.n_units_in,
                'n_units_out': ds.n_units_out,
                'window_size': WINDOW_SIZE,
                'y': ds.target.shape
            }, str(save_path))
        save['total_loss'].append(total_loss)
    pd.DataFrame(save).to_csv(RESULT_FOLDER.joinpath(f'model.csv'))

    print(f'Best loss: {min(save["total_loss"]):.5f}')
