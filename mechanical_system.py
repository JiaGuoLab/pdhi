import numpy as np
from scipy.linalg import expm


class MechanicalSystem(object):

    def __init__(self, nDOF, nr, m, k, h, T):
        self.Mglobal = m * np.eye(nDOF)
        S = 2.0 * np.eye(nDOF) - np.diag(np.ones(nDOF - 1), k=1) - np.diag(np.ones(nDOF - 1), k=-1)
        S[-1, -1] = 1.0
        self.Kglobal = k * S
        self.Fexglobal = -self.Mglobal @ np.ones((nDOF, 1))
        m_ode = np.diag(self.Mglobal)
        self.M_ode = m_ode[:, np.newaxis]

        w1 = 4.6747
        w2 = 13.8197
        p0 = 2 * w1 * w2 * (h * w2 - h * w1) / (w2 ** 2 - w1 ** 2)
        p1 = 2 * (h * w2 - h * w1) / (w2 ** 2 - w1 ** 2)
        self.Cglobal = p0 * self.Mglobal + p1 * self.Kglobal

        self.Tglobal = T
        self.nDOF = nDOF
        self.nr = nr
        u0 = np.zeros((nDOF, 1))
        v0 = np.zeros((nDOF, 1))
        r0 = np.zeros((nr, 1))
        self.y0 = np.concatenate((u0, v0, r0), axis=0)[:, 0]

        self.AA, self.Bf, self.Bz = None, None, None

    def ZOH_matrix(self, dt):
        nDOF = self.nDOF
        M_ode = self.M_ode
        Fex = self.Fexglobal
        K = self.Kglobal
        C = self.Cglobal
        T = self.Tglobal

        # generate matrices for classic exponential time-step scheme according to Eq.11~12
        a1 = np.zeros((nDOF, nDOF))
        a2 = np.eye(nDOF)
        A1 = np.concatenate((a1, a2), axis=1)
        A2 = np.concatenate((-K / M_ode, -C / M_ode), axis=1)
        Ac = np.concatenate((A1, A2), axis=0)
        Bfc = np.concatenate((np.zeros((nDOF, 1)), Fex / M_ode), axis=0)
        Bzc = np.concatenate((np.zeros((nDOF, 1)), -np.transpose(T) / M_ode), axis=0)
        self.AA = expm(Ac * dt)
        inv_Ac = np.linalg.inv(Ac)
        self.Bf = (self.AA - np.eye(self.AA.shape[0])) @ inv_Ac @ Bfc
        self.Bz = (self.AA - np.eye(self.AA.shape[0])) @ inv_Ac @ Bzc

        return self