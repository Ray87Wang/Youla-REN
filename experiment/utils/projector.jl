using PyCall
using LinearAlgebra

includet("./LTI_utils.jl")
includet("./rnn_utils.jl")

py"""
import cvxpy as cp
import numpy as np

class projector(object):
    def __init__(self, Ag, Bg, Cg, nz, nw):
        self.Ag = Ag
        self.Bg = Bg
        self.Cg = Cg
        nx = np.shape(Ag)[0]
        nu = np.shape(Bg)[1]
        ny = np.shape(Cg)[0]
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.nz = nz
        self.nw = nw 
        rho = 0.98
        vX  = cp.Variable((nx, nx), symmetric=True)
        vY  = cp.Variable((nx, nx), symmetric=True)
        vK  = cp.Variable((nx, nx))
        vL  = cp.Variable((nx, ny))
        vM  = cp.Variable((nu, nx))
        vN  = cp.Variable((nu, ny))
        obj = 0
        yPAy = cp.bmat([
            [Ag @ vY + Bg @ vM, Ag + Bg @ vN @ Cg],
            [vK,                vX @ Ag + vL @ Cg]
        ])
        yPy = cp.bmat([
            [vY, np.eye(nx)],
            [np.eye(nx), vX]
        ])
        LMI1 = cp.bmat([
            [rho ** 2 *yPy,  yPAy.T],
            [yPAy, yPy]
        ])
        LMI2 = yPy
        cons = [LMI1 >> 0, LMI2 >> 0]
        p = cp.Problem(cp.Minimize(obj), cons)
        result = p.solve(solver="MOSEK")
        U = vX.value
        V = np.linalg.inv(vX.value) - vY.value
        P = np.linalg.inv(np.block([[vY.value, V],[np.eye(nx), np.zeros((nx, nx))]])) @ \
            np.block([[np.eye(nx), np.zeros((nx, nx))],[vX.value, U]])
        Q10 = np.linalg.inv(P)
        self.Q1 = np.block([ 
            [Q10, np.zeros((2*nx, nz-nx))],
            [np.zeros((nz-nx,2*nx)), np.eye(nz-nx)]
        ])
        self.Q2  = np.eye(nw)

    def get_Q(self):
        return self.Q1, self.Q2

    def project(self, Ak, Bk1, Bk2, Ck1, Dk1, Dk2, Ck2, Dk3):
        c   = 0.5
        eps = 1e-5
        nx  = self.nx
        nu  = self.nu
        ny  = self.ny
        nz  = self.nz
        nw  = self.nw
        Ag  = self.Ag
        Bg  = self.Bg
        Cg  = self.Cg
        Ak  = Ak + c*Bk1 @ Ck2
        Bk2 = Bk2 + c*Bk1 @ Dk3
        Ck1 = Ck1 + c*Dk1 @ Ck2
        Dk2 = Dk2 + c*Dk1 @ Dk3
        Q1i = np.linalg.inv(self.Q1)
        Q2i = np.linalg.inv(self.Q2)
        vAk = cp.Variable((nz, nz))
        vBk1= cp.Variable((nz, nw))
        vBk2= cp.Variable((nz, ny))
        vCk1= cp.Variable((nu, nz))
        vDk1= cp.Variable((nu, nw))
        vDk2= cp.Variable((nu, ny))
        vCk2= cp.Variable((nw, nz))
        vDk3= cp.Variable((nw, ny))
        vQ1 = cp.Variable((nz+nx, nz+nx), symmetric=True)
        vQ2 = cp.diag(cp.Variable(nw))
        obj = cp.sum_squares(vAk - Ak) + \
          cp.sum_squares(vBk1 - Bk1) + \
          cp.sum_squares(vBk2 - Bk2) + \
          cp.sum_squares(vCk1 - Ck1) + \
          cp.sum_squares(vDk1 - Dk1) + \
          cp.sum_squares(vDk2 - Dk2) + \
          cp.sum_squares(vCk2 - Ck2) + \
          cp.sum_squares(vDk3 - Dk3) + \
          cp.sum_squares(vQ1 - self.Q1) + \
          cp.sum_squares(vQ2 - self.Q2)
        A   = cp.bmat([
            [Ag + Bg @ vDk2 @ Cg, Bg @ vCk1],
            [vBk2 @ Cg,            vAk     ]
          ])
        B   = cp.bmat([
            [c*Bg @ vDk1 ],
            [c*vBk1      ]
          ])
        C   = cp.bmat([[vDk3 @ Cg, vCk2]])
        D   = np.zeros((nw, nw))
        LMI = cp.bmat([
            [2 * Q1i - Q1i @ vQ1 @ Q1i , np.zeros((nz+nx, nw)), A.T, C.T],
            [np.zeros((nw, nz+nx)), 2 * Q2i - Q2i @ vQ2 @ Q2i,  B.T, D.T],
            [A, B, vQ1, np.zeros((nz+nx, nw))],
            [C, D, np.zeros((nw, nz+nx)), vQ2]
          ])
        cons = [LMI >> eps * np.eye(2*(nz+nx+nw)), vQ1 >> eps*np.eye(nz+nx), cp.diag(vQ2) >= eps]
        prob = cp.Problem(cp.Minimize(obj), cons)
        res  = prob.solve(solver="MOSEK")
        AK   = vAk.value
        BK1  = vBk1.value
        BK2  = vBk2.value
        CK1  = vCk1.value
        DK1  = vDk1.value
        DK2  = vDk2.value
        CK2  = vCk2.value
        DK3  = vDk3.value
        self.Q1 = vQ1.value
        self.Q2 = vQ2.value
        AK   = AK  - c*BK1 @ CK2
        BK2  = BK2 - c*BK1 @ DK3
        CK1  = CK1 - c*DK1 @ CK2
        DK2  = DK2 - c*DK1 @ DK3
        K    = np.block([ 
            [AK,  BK1, BK2],
            [CK1, DK1, DK2],
            [CK2, np.zeros((nw,nw)), DK3]
        ])
        return K
"""


function projector(G::lti,C::rnn)
    Ag, Bg, Cg = PyObject(G.A), PyObject(G.B), PyObject(G.C)
    nξ, nϕ = C.nξ, C.nϕ
    P = py"projector"(Ag, Bg, Cg, nξ, nϕ)
    return P
end

function projection!(C::rnn, P)
    Ak,  Bk1, Bk2 = PyObject(C.A),  PyObject(C.B1), PyObject(C.B2)
    Ck1, Dk1, Dk2 = PyObject(C.C1), PyObject(C.D1), PyObject(C.D2)
    Ck2, Dk3      = PyObject(C.C2), PyObject(C.D3)
    K = P.project(Ak, Bk1, Bk2, Ck1, Dk1, Dk2, Ck2, Dk3)
    nξ, nϕ, ny, nu = C.nξ, C.nϕ, C.ny, C.nu
    C.A  = K[1:nξ, 1:nξ]
    C.B1 = K[1:nξ, nξ+1:nξ+nϕ]
    C.B2 = K[1:nξ, nξ+nϕ+1:nξ+nϕ+ny]
    C.C1 = K[nξ+1:nξ+nu, 1:nξ]
    C.C2 = K[nξ+nu+1:nξ+nu+nϕ, 1:nξ]
    C.D1 = K[nξ+1:nξ+nu, nξ+1:nξ+nϕ]
    C.D2 = K[nξ+1:nξ+nu, nξ+nϕ+1:nξ+nϕ+ny]
    C.D3 = K[nξ+nu+1:nξ+nu+nϕ, nξ+nϕ+1:nξ+nϕ+ny]
end
