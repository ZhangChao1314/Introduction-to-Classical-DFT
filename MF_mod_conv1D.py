import numpy as np
from numba import jit

@jit(nopython=True)
def Phi_wca(dist,rc,eps,Sig_ff):
    rmin = 2.**(1./6.)*Sig_ff
    if dist>rc:
        return 0
    elif (dist>rmin and dist<=rc):
        Temp1 = (2./5.)*(eps*Sig_ff**12)/(dist**10)-eps*Sig_ff**6/(dist**4)
        Temp2 = (2./5.)*(eps*Sig_ff**12)/(rc**10)-eps*Sig_ff**6/(rc**4)
        return  2.*np.pi*(Temp1 - Temp2)
    else:
        Temp0 = 0.5*eps*(dist**2-rmin**2)
        Temp1 = (2./5.)*(eps*Sig_ff**12)/(rmin**10)-eps*Sig_ff**6/(rmin**4)
        Temp2 = (2./5.)*(eps*Sig_ff**12)/(rc**10)-eps*Sig_ff**6/(rc**4)
        return 2.*np.pi*(Temp0 + Temp1 - Temp2)

@jit(nopython=True)
def MF_phi(Z_vec,rc,eps,Sig_ff):
    Phi_att = np.zeros((len(Z_vec),len(Z_vec)))
    for j in range(len(Z_vec)):
        for i in range(len(Z_vec)):
           Phi_att[i,j] = Phi_wca(abs(Z_vec[i]-Z_vec[j]),rc,eps,Sig_ff)
    return Phi_att

@jit(nopython=True)
def calc_dFAtt(H, MF, rho_vec, Nump, dz, Sigma_ff, eps, Z_vec, rc):
    conv = np.sum(MF * rho_vec, axis=1)
    FAtt = conv * dz
    return FAtt

def MF_weight(Z_vec,rc_MF,Eps_kb,Sigma_ff):
    Phi_att = np.zeros((len(Z_vec),len(Z_vec)))
    for j in range(len(Z_vec)):
        for i in range(len(Z_vec)):
           Phi_att[i,j] = Phi_wca(abs(Z_vec[i]-Z_vec[j]),rc_MF,Eps_kb,Sigma_ff)
    return Phi_att
