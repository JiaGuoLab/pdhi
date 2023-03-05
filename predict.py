import sys
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import torch
import matplotlib.pyplot as plt
from scipy.io import savemat

sys.path.append(Path.cwd())

from train import MyConvNet, ResidualWrapper, AutoRegressiveWrapper, Wrapper
from mechanical_system import MechanicalSystem

dataset = 'predict_data.mat'
USE_CPU = False

DEVICE = torch.device("cpu")
if not USE_CPU and torch.cuda.is_available():
    DEVICE = torch.device("cuda")


class PDHIRegressiveWrapper(Wrapper):
    def __init__(self, model):
        # build the base-isolated 7-story shear building model according to the first paragraph in Section 4
        super().__init__(model, wrapper_name='pdhi_regressive')
        self.T = np.array([1.0, 0, 0, 0, 0, 0, 0], ndmin=2)
        self.system = MechanicalSystem(nDOF=7, nr=1, m=2e3, k=1e6, h=0.01, T=self.T)
        self.gm = None
        self.dt = 1.0 / 200
        self.system.ZOH_matrix(self.dt)

    def forward(self, gm, *args, **kwargs):
        print(f'-------------------------------------')
        print(f'Input shape (ground motion): {gm.shape}')
        self.gm = gm
        n_steps = self.gm.shape[1] - 1
        nDOF = 7
        batch_size = self.gm.shape[0]
        # four input ground motions: El Centro, Taft, Kobe, and Kumamoto, see Table 1 in Section 4.1
        print(f'Batch size: {batch_size}')
        print(f'Total {n_steps} Steps.')
        print('Running....')

        xz_k = np.zeros((batch_size, nDOF * 2 + 1))
        xz_all = xz_k[:, np.newaxis, :]
        # system state initialization:
        # 1-7 dofs for the displacement from 1st to 7th story;
        # 8-14 dofs for the velocity from 1st to 7th story;
        # 15th dof for the damping force in the 1st story.

        u_indexes = np.array([0, 7])      # 1st & 8th dofs for the damping system's displacement and velocity
        u_params = np.array([0.05, 0.8])  # corresponding normalization parameter
        z_indexes = np.array([14])        # 15th dof for the damping force
        z_params = np.array([1e4])        # corresponding normalization parameter

        for k in range(n_steps):
            # Algorithm 1:
            # (i) p_k, x_k --> next_x_k in Eq.12
            p_k = self.gm[:, k]
            next_x_k = xz_k[:, :2 * nDOF] @ np.transpose(self.system.AA) + \
                       p_k @ np.transpose(self.system.Bf) + \
                       xz_k[:, 2 * nDOF:] @ np.transpose(self.system.Bz)

            # (ii) next_x_k --> next_u_k (u=Hx in Section 2.2.2)
            next_u_k = (next_x_k[:, u_indexes] / u_params).astype(np.float32)

            # (iii) network prediction: z_k, next_u_k --> next_z_k according to Eq.18-1 (Adams_Moulton M=0)
            z_k_us = xz_k[:, z_indexes]
            z_k = (z_k_us / z_params).astype(np.float32) if len(z_indexes) > 0 else None
            network_inputs = np.concatenate((z_k, next_u_k), axis=1)[:, np.newaxis, :]
            # -------------network prediction-------------------
            output = self.model.forward(torch.Tensor(network_inputs).to(DEVICE).float(), *args, **kwargs).to(
                'cpu').numpy()
            next_z_k = output[:, 0, :]
            # save results and update state
            next_xz_k = np.concatenate((next_x_k, next_z_k * z_params[0]), axis=1)
            xz_all = np.concatenate((xz_all, next_xz_k[:, np.newaxis, :]), axis=1)
            xz_k = next_xz_k[:]

        print(f'Output shape (response): {xz_all.shape}')
        print(f'-------------------------------------')
        return xz_all


if __name__ == '__main__':
    CODE_ROOT = Path.cwd()
    RESULT_FOLDER = CODE_ROOT.joinpath(CODE_ROOT, 'result')
    print(CODE_ROOT)
    # --------------------1. load models------------------
    model_file = RESULT_FOLDER.joinpath(f'model.pth')
    ckpt = torch.load(model_file)
    window_size = ckpt['window_size']
    model = MyConvNet(suggest=ckpt['suggest']).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model = ResidualWrapper(model).to(DEVICE)
    model = AutoRegressiveWrapper(model).to(DEVICE)
    model = PDHIRegressiveWrapper(model).to(DEVICE)

    # --------------------2. load trained data-------------------
    data = loadmat(dataset)
    input_gm, response_true = data['gm'], data['response_gt']

    # --------------------3. predict---------------------
    model.eval()
    with torch.no_grad():
        prediction = model.forward(input_gm[:, :, np.newaxis])
    savemat(RESULT_FOLDER.joinpath('prediction.mat'), {'prediction': prediction})
    print(f'Predict completed.')
    if response_true is None:
        sys.exit(0)

    plt.figure()
    time_steps = np.arange(prediction.shape[1]) * model.dt
    indexes_range = min(prediction.shape[0], 4)
    indexes = np.arange(indexes_range)
    for i in range(indexes_range):
        plt.subplot(2, 2, i + 1)
        force_pred = prediction[indexes[i], :, 14]/1000.0
        force_true = response_true[indexes[i], :, 14]/1000.0
        plt.plot(time_steps, force_true, 'b-', label='truth')
        plt.plot(time_steps, force_pred, 'r-', label='prediction')
        plt.ylim([-15, 15])
        plt.ylabel('Damping force (kN)')
        plt.xlabel('Time (s)')
    plt.legend(loc='best')
    plt.suptitle(f"Predict Results for different ground motions")
    plt.show()
