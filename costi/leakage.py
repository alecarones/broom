import numpy as np
import healpy as hp
import pymaster as nmt
from sklearn.linear_model import LinearRegression

def purify_master(QU_maps,mask,lmax, return_E=True, return_B=True, purify_E=False):
    print('Performing MASTER purification.')
    maskbin = np.zeros_like(mask)
    maskbin[mask > 0.] = 1.
    nside = hp.get_nside(QU_maps[0])
    
    fp = nmt.NmtField(mask, [(QU_maps[0])*maskbin, (QU_maps[1])*maskbin],lmax=lmax,lmax_mask=lmax)
    alms_p, _ = fp._purify(fp.mask, fp.get_mask_alms(), [(QU_maps[0])*maskbin, (QU_maps[1])*maskbin], n_iter=fp.n_iter,task=[purify_E,True])
    if return_E and return_B:
        return alms_p
    elif return_E and not return_B:
        return alms_p[0]
    elif return_B and not return_E:
        return alms_p[1]
    else:
        raise ValueError("At least one of 'return_E' and 'return_B' must be True.")

def purify_recycling(QU_maps,mask,lmax, return_E=True, return_B=True, purify_E=False, iterations=0):
    print('Performing recycling purification.')
    maskbin = np.zeros_like(mask)
    maskbin[mask > 0.] = 1.
    nside = hp.get_nside(QU_maps[0])
    
    TQU_maps = np.array([0.*QU_maps[0],QU_maps[0],QU_maps[1]])

    if return_E and not purify_E:
        alms_E_p = hp.map2alm(TQU_maps*mask, lmax=lmax, iter=3, pol=True)[1]
    elif return_E and purify_E:
        print("Purification of E not implemented yet for recycling technique.")

    alms_m = hp.map2alm(TQU_maps*maskbin, lmax=lmax, iter=3, pol=True)
    
    alms_E = np.zeros((3,alms_m.shape[1]),dtype=complex)
    alms_B = np.zeros((3,alms_m.shape[1]),dtype=complex)
    alms_E[1] = alms_m[1]
    alms_B[2] = alms_m[2]

    maps_TQU_E=hp.alm2map(alms_E, nside, lmax=lmax, pol=True)
    maps_TQU_B=hp.alm2map(alms_B, nside, lmax=lmax, pol=True)

    alms_E_B=hp.map2alm(maps_TQU_E*maskbin, lmax=lmax, iter=3, pol=True)[2]

    alms_B_m = np.zeros((3,alms_m.shape[1]),dtype=complex)
    alms_B_m[2]=alms_E_B
    maps_TQU_B_temp = hp.alm2map(alms_B_m, nside, lmax=lmax, pol=True)

    reg_Q = LinearRegression(fit_intercept=False).fit(maps_TQU_B_temp[1,maskbin>0].reshape(-1, 1), (maps_TQU_B)[1,maskbin>0])
    reg_U = LinearRegression(fit_intercept=False).fit(maps_TQU_B_temp[2,maskbin>0].reshape(-1, 1), (maps_TQU_B)[2,maskbin>0])

    QU_p = np.zeros((3,12*nside**2))
    QU_p[1] = maps_TQU_B[1]-((reg_Q.coef_)[0])*maps_TQU_B_temp[1]
    QU_p[2] = maps_TQU_B[2]-((reg_U.coef_)[0])*maps_TQU_B_temp[2]
    
    if iterations > 0:
        for it in range(iterations):
            alms_B_m = np.zeros((3,alms_m.shape[1]),dtype=complex)
            alms_B_m[2] = hp.map2alm([0.*(QU_p[0]),(QU_p[1])*maskbin,(QU_p[2])*maskbin], lmax=lmax, iter=3, pol=True)[2]
            QU_p = hp.alm2map(alms_B_m,nside,lmax=lmax,pol=True)
    
    alms_B_p = hp.map2alm([0.*(QU_p[0]),(QU_p[1])*mask,(QU_p[2])*mask], lmax=lmax, iter=3, pol=True)[2]

    if return_E and return_B:
        return np.array([alms_E_p,alms_B_p])
    elif return_E and not return_B:
        return alms_E_p
    elif return_B and not return_E:
        return alms_B_p
    else:
        raise ValueError("At least one of 'return_E' and 'return_B' must be True.")