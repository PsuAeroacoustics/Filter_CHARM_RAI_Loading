#!/usr/bin/env python3
import numpy as np
from scipy.fft import fft,ifft
from scipy.optimize import Bounds
import dependencies.resonator.resonator as res

#%%



def get_res_dist(res_params):

    mult = 8

    t = np.arange(res_params['N_r']*mult)/(res_params['N_r']*mult-1)
    f_max = (2*np.diff(t[:2])*mult)**-1
    f_min = (2*res_params['N_r']*mult*np.diff(t[:2]))**-1

    x = (10-10**(1-np.asarray(res_params['dist'])))/10
    # x =np.zeros(5)

    phi = 2*np.pi*(x@np.asarray((1/5*(t-1)**5,
                                t**2*(2-4*t+3*t**2-4/5*t**3),
                                t**3*(2-3*t+6/5*t**2),
                                t**4*(1-4/5*t),
                                1/5*t**5))*(f_max-f_min)+f_min*t)
    # x =[1,1,1]
    # phi = 2*np.pi*(x@np.array((t*(1/3*t**2-t+1),
    #                             t**2*(1-2/3*t),
    #                             1/3*t**3))*(f_max-f_min)+f_min*t)
    filt_ind = np.invert(np.sin(phi)[:,int(mult/2)::mult] < 0).flatten()

    return filt_ind

def get_Z_smeared(saved_params,res_params):
    Z_smeared = np.zeros((saved_params['N_patches'],len(saved_params['f'])),dtype=complex)
    for i in range(saved_params['N_patches']):
        N_res = len(res_params['l'][i])
        Z = np.asarray([init_res(saved_params['f'],a_n = res_params['a'],L_n = res_params['l'][i][res_itr]/2,a_c = res_params['a'],L_c = res_params['l'][i][res_itr]/2).Z for res_itr in range(N_res)])
        Z_smeared[i] = ((res_params['OAR']/N_res*Z**-1).sum(axis = 0))**-1
    saved_params.update({'Z_smeared':Z_smeared.T})

def get_sample_info(filt_ind,A_s,**kwargs):

    def get_res_geom(x):

        # x = [0.0003, 0.07, 0, 1, .1, 1, .2, .2,.2]
        a = np.array([x[0]])

        if kwargs['N_res']==1:
            L = np.array([(x[1]-kwargs['L_min'])+kwargs['L_min']])
            N = np.expand_dims(np.floor(kwargs['OAR']*A_s/(np.pi*a**2)),axis = 0)
        else:
            t = np.linspace(0,1, kwargs['N_res'])
            L_dist =(list(x[2:4])+[1])@np.array(((1-t)**2,2*t*(1-t),t**2))
            L = (x[1]-kwargs['L_min'])*L_dist+kwargs['L_min']

            N_dist = x[4:9]@np.array(((1-t)**4,4*t*(1-t)**3,6*t**2*(1-t)**2,4*t**3*(1-t),t**4))
            N = np.floor((kwargs['OAR']*np.expand_dims(A_s,axis = -1)/(np.pi*a**2))/np.sum(N_dist)*N_dist).T

        # import matplotlib.pyplot as plt
        # r_ind = int(len(N)*.75)
        # fig,ax = plt.subplots(2,1, figsize = (6.5,4.5))
        # plt.subplots_adjust(bottom = .15, left = 0.15,hspace = .25)
        # ax[0].plot(t,L_dist)
        # ax[0].set_ylabel('$B_L$')
        # ax[0].grid()
        # ax[1].plot(L)
        # ax[1].set_ylabel('$L \ [m]$')
        # ax[1].grid()

        # fig,ax = plt.subplots(2,1, figsize = (6.5,4.5))
        # plt.subplots_adjust(bottom = .15, left = 0.15,hspace = .25)
        # ax[0].plot(t,N_dist)
        # ax[0].set_ylabel('$B_N$')
        # ax[0].grid()
        # ax[1].plot(N[:,r_ind])
        # ax[1].set_ylabel('$N$')
        # ax[1].grid()

        return N,a,L

    def get_res_dist(saved_params,res_params):

        mult = 8

        t = np.arange(res_params['N_r']*mult)/(res_params['N_r']*mult-1)
        f_max = (2*np.diff(t[:2])*mult)**-1
        f_min = (2*res_params['N_r']*mult*np.diff(t[:2]))**-1

        x = (10-10**(1-np.asarray(res_params['dist'])))/10
        # x =np.zeros(5)

        phi = 2*np.pi*(x@np.asarray((1/5*(t-1)**5,
                                    t**2*(2-4*t+3*t**2-4/5*t**3),
                                   t**3*(2-3*t+6/5*t**2),
                                   t**4*(1-4/5*t),
                                   1/5*t**5))*(f_max-f_min)+f_min*t)
        # x =[1,1,1]
        # phi = 2*np.pi*(x@np.array((t*(1/3*t**2-t+1),
        #                             t**2*(1-2/3*t),
        #                             1/3*t**3))*(f_max-f_min)+f_min*t)
        filt_ind = np.invert(np.sin(phi)[int(mult/2)::mult] < 0)

        return filt_ind

    
    out = {}
    for i in range(kwargs['N_patches']):
        if kwargs['N_res'] >1:
            x = [kwargs['x'][0]]+kwargs['x'][(i*8+1):(((i+1)*8+1))]
        else:
            x = [kwargs['x'][0]]+kwargs['x'][(i+1):(i+2)]
        N,a,L = get_res_geom(x)
        out.update({f'patch_{i}':{'N':N,'a':a,'L':L}})

    if kwargs['staggered']:
        filt_ind = get_res_dist(kwargs['x'][-5:])

    out.update({'filt_ind':filt_ind})

    return out

            
def init_res(f,a_n,L_n,a_c,L_c):
    '''
    This function creates and returns a resonator object with its complex impedance evaluated
    '''
    res_temp = res.resonator(a_n = a_n,L_n =L_n,a_c = a_c, L_c = L_c)
    res_temp.set_Z(f,model = 'K',rad = False,interior = False,loss = True,table = False)
    return res_temp

def init_porous_res(f,**kwargs):
    '''
    This function creates and returns a resonator object with its complex impedance evaluated
    '''
    res_temp = res.resonator(q=kwargs['q'],s_b =kwargs['s_b'],t =kwargs['t'],sigma = kwargs['sigma'],phi = kwargs['phi'])
    res_temp.set_Z(f)
    return res_temp

def init_fs(f,t_fs,r_fs,phi_fs,SPL,M,Z_cav):
    '''
    This function creates and returns a resonator object with its complex impedance evaluated
    '''
    fs_temp = res.fs(t = t_fs,r = r_fs,phi = phi_fs)
    fs_temp.set_Z(f,M = M,SPL = SPL,model = '2P',Z_cav = Z_cav)
    return fs_temp


def smeared_Z(f,A_s,**kwargs):
    '''
    This function computes the smeared impedance and absorption for a sample that consists of multiple resonator cavities that have different geometries. 
    
    Parameters:
    f: frequency array [Hz]
    res_params: a nested array whose elements corresponding to each resonator in the sample. Each of the nested arrays must be of size (5,) and specifies
    [[# of this resonator in the sample, a_n,L_n,a_c,L_c]]

    Return:
    Z_tot: total non-dimensionalized (rho*c0) complex impedance of the sample (inverse of the total admittance)
    alpha: total absorption of the sample
    '''
    # res_params = np.array([[1.14238749e+02, 1.22225632e-03, 2.55143525e-01],[1.49012815e+02, 9.76120148e-04 ,4.66389696e-02],[3.12709586e+01, 1.38153591e-03, 1.25022033e-01],[5.91861319e+01, 1.05103046e-03, 1.85281711e-01]])

    # gets complex acoustic impedance of each resonator in a single sample/blade element
    # if N_res==1:
    #     Z = init_res(f,a_n = a,L_n = L/2,a_c = a,L_c = L/2).Z
    #     Z_tot = np.array([(N[i]*np.pi*a**2/A_s[i]*Z**-1)**-1 for i in range(elements)])

    # else:
    # if N_res ==1:
    #     Z = np.array(init_res(f,a_n = a,L_n = L/2,a_c = a,L_c = L/2).Z)
    #     Z_tot = ((np.expand_dims(N*np.pi*a**2/A_s,axis = -1)*Z**-1)**-1).T
    # else:
    Z = np.array([init_res(f,a_n = kwargs['a'],L_n = kwargs['L'][i]/2,a_c = kwargs['a'],L_c = kwargs['L'][i]/2).Z for i in range(len(kwargs['L']))])
    # if len(kwargs['L'])>1:
    Z_tot = (((kwargs['N']*np.pi*kwargs['a']**2)/A_s).T@Z**-1).T**-1
    # else:
    # Z_tot =(np.expand_dims(kwargs['N']*np.pi*kwargs['a']**2/kwargs['A_s'],axis = -1)@Z**-1).T**-1
    # if facesheet:

    #     # if linear:
    #     #     M = 0
    #     #     SPL = 60

    #     # else:
    #     #     SPL = 20*np.log10(np.sqrt(np.mean((z_loading_store[:,0]/A_s)**2,axis = 0))/20e-6)
    #     #     M = np.array(r)*R*omega/sos

    #     if N_res==1:
            
    #         r_fs = np.sqrt(phi_fs*np.pi*a**2/(N_fs*np.pi))
            
    #         if linear:
    #             Z_fs = init_fs(f,t_fs = t_fs,r_fs = r_fs,phi_fs = phi_fs,M = M,SPL = SPL,Z_cav = Z).Z
    #             Z_tot = (N*np.pi*a**2/A_s*np.expand_dims(Z+Z_fs,axis = -1)**-1).T**-1
    #         else:
    #             Z_fs = np.array([init_fs(f,t_fs = t_fs,r_fs = r_fs,phi_fs = phi_fs,M = M[i],SPL = SPL[i],Z_cav = Z).Z for i in range(elements)])
    #             Z_tot = (N*np.pi*a**2/A_s*(Z+Z_fs).T**-1).T**-1

    #     else:
    #         r_fs = np.min(np.sqrt(phi_fs*np.pi*a**2/(N_fs*np.pi)))
    #         N_fs = np.floor(phi_fs*np.pi*a**2/(np.pi*r_fs**2))
    #         phi_fs = N_fs*np.pi*r_fs**2/(np.pi*a**2)

    #         # r_fs = np.sqrt((np.min(np.pi*a**2)*phi)/np.pi)
    #         # N_fs = np.floor(phi*a**2/r_fs**2)
    #         # phi = N_fs*np.pi*r_fs**2/(np.pi*a**2)
    #         if linear:
    #             Z_fs = np.array([init_fs(f,t_fs = t_fs,r_fs = r_fs,phi_fs = phi_fs[i],M = M,SPL = SPL,Z_cav = Z[i]).Z for i in range(N_res)]).T
    #         else:
    #             Z_fs = np.array([[init_fs(f,t_fs = t_fs,r_fs = r_fs,phi_fs = phi_fs[i],M = M[ii],SPL = SPL[ii],Z_cav = Z[i]).Z for i in range(N_res)] for ii in range(elements)]).transpose(0,-1,1)


    # alpha = 1 - abs((Z_tot-1)/(Z_tot+1))**2


    return Z_tot

def get_filt_resp(Z_tot):

    N_elements = Z_tot.shape[-1]
    N_pnt = int(len(Z_tot))
     # reflection coefficient
    R = (Z_tot-1)/(Z_tot+1)
    
    if not N_pnt%2:
        filt_resp = np.ones((N_pnt*2+1,N_elements),dtype = complex)
        filt_resp[1:N_pnt+1] = R
        filt_resp[N_pnt+1:] = np.conj(R)[::-1]

    else:
        # complex admittance of the resonators
        filt_resp = np.ones((N_pnt*2,N_elements),dtype = complex)
        filt_resp[1:N_pnt] = R[:-1]
        filt_resp[N_pnt+1:] = np.conj(R[:-1])[::-1]

    return filt_resp

def get_filt_resp_2(Z_tot):

    N_elements = Z_tot.shape[-1]
    N_pnt = int(len(Z_tot))
     # reflection coefficient
    
    if not N_pnt%2:
        filt_resp = np.ones((N_pnt*2+1,N_elements),dtype = complex)
        filt_resp[1:N_pnt+1] = Z_tot
        filt_resp[N_pnt+1:] = np.conj(Z_tot)[::-1]

    else:
        # complex admittance of the resonators
        filt_resp = np.ones((N_pnt*2,N_elements),dtype = complex)
        filt_resp[1:N_pnt] = Z_tot[:-1]
        filt_resp[N_pnt+1:] = np.conj(Z_tot[:-1])[::-1]

    return filt_resp


def apply_filt(loads,filt_resp):

    Xm = fft(loads,axis = 0)
    # Xm_filt = np.zeros(Xm.shape,dtype=complex)
    # Xm_filt = (Xm.T*filt_resp).T
    Xm_filt = (Xm.T*filt_resp[None]).T
    # performs an ifft to convert filtered loads back to the time domain
    xn_filt = np.real(ifft(Xm_filt,axis =0))
    return xn_filt

def get_opt_bounds(**kwargs):

    lb,ub= [kwargs['r_min'],kwargs['L_min']], [kwargs['r_max'],kwargs['L_max']]
    if kwargs['N_res']>1:
        lb = lb+[0]*7
        ub = ub+[1]*7
    lb[1:] = np.tile(lb[1:],kwargs['N_patches'])
    ub[1:] = np.tile(ub[1:],kwargs['N_patches'])

    if kwargs['staggered']:
        lb = lb+[0]*5
        ub = ub+[1]*5

    bounds = Bounds(lb = lb, ub = ub,keep_feasible=True)
    return bounds
 
 