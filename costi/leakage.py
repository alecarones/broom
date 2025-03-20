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
    lm = hp.Alm.getsize(lmax)

    if QU_maps.ndim == 2:
        QU_maps = QU_maps[...,np.newaxis]
        nside = hp.get_nside(QU_maps[0])
        remove_last_dim = True
    else:
        nside = hp.get_nside(QU_maps[0,:,0])
        remove_last_dim = False

    if return_E and return_B:
        alms_p = np.zeros((2,lm,QU_maps.shape[-1]),dtype=complex)
    elif (return_E and not return_B) or (return_B and not return_E):
        alms_p = np.zeros((1,lm,QU_maps.shape[-1]),dtype=complex)
    else:
        raise ValueError("At least one of 'return_E' and 'return_B' must be True.")
    
    for c in range(QU_maps.shape[-1]):
        TQU_maps = np.array([0.*QU_maps[0,:,c],QU_maps[0,:,c],QU_maps[1,:,c]])

        if return_E and not purify_E:
            alms_p[0,:,c] = hp.map2alm(TQU_maps*mask, lmax=lmax, iter=3, pol=True)[1]
        elif return_E and purify_E:
            print("Purification of E not implemented yet for recycling technique.")

        if return_B:
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

            if c==0:
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
            
            if return_E:
                alms_p[1,:,c] = hp.map2alm([0.*(QU_p[0]),(QU_p[1])*mask,(QU_p[2])*mask], lmax=lmax, iter=3, pol=True)[2]
            else:
                alms_p[0,:,c] = hp.map2alm([0.*(QU_p[0]),(QU_p[1])*mask,(QU_p[2])*mask], lmax=lmax, iter=3, pol=True)[2]
    
    if remove_last_dim:
        alms_p = alms_p[...,0]
        
    if (return_E and not return_B) or (return_B and not return_E):
        return alms_p[0]
    else:
        return alms_p


    