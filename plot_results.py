#!/usr/bin/env python3
import post.plot_styles
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import griddata
from scipy.signal import welch
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
from matplotlib.cm import ScalarMappable


default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyle = ['-',':','--','-.',':']
title = [r"$\phi=23^\circ$",r"$\phi=0^\circ$",r"$\phi=-23^\circ$"]

def plot_ptseries(saved_params,res_params):

    figsize = (.6*6.5*.95,0.6*6.5*.95)
    fig, ax = plt.subplots(len(saved_params['observers']),1,figsize = figsize)
    plt.subplots_adjust(left=0.175,right = 0.95,top = 0.95,bottom=0.18,hspace = 0.4)
    for mic_itr in range(len(saved_params['observers'])):
        ax[mic_itr].plot(saved_params['baseline_function_values'][saved_params['observers']][mic_itr,0,:,0]/saved_params['baseline_function_values'][saved_params['observers']][mic_itr,0,-1,0],np.roll(saved_params['baseline_function_values'][saved_params['observers']][mic_itr,0,:,-1],-128), linestyle=linestyle[0])
        ax[mic_itr].plot(saved_params['function_values'][saved_params['observers']][mic_itr,0,:,0]/saved_params['function_values'][saved_params['observers']][mic_itr,0,-1,0],np.roll(saved_params['function_values'][saved_params['observers']][mic_itr,0,:,-1],-128), linestyle=linestyle[1])
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        t = ax[mic_itr].text(0.5, .15, rf'$\Delta$ OASPL ={np.round(10*np.log10(np.mean(saved_params['baseline_function_values'][saved_params['observers']][mic_itr,0,:,-1]**2)/20e-6**2)-10*np.log10(np.mean(saved_params['function_values'][saved_params['observers']][mic_itr,0,:,-1]**2)/20e-6**2),1)} dB', ha="center",va="center",transform=ax[mic_itr].transAxes)
        t.set_bbox(dict(facecolor='white' ,edgecolor = 'gray' ,alpha=.6,boxstyle='round'))
        ax[mic_itr].set(xlim = [0,1],ylim = [-10,5],title = title[mic_itr])
        if mic_itr !=len(saved_params['observers'])-1:
            ax[mic_itr].set_xticklabels([])
        ax[mic_itr].grid()
    ax[-1].set(xlabel = 'Rev. Fraction')
    ax[int(len(saved_params['observers'])/2)].set(ylabel = 'p [Pa]')
    fig.legend(['Untreated','Treated'],ncol = 2,loc='lower center',bbox_to_anchor=(.5, -0.02))
    plt.savefig(os.path.join(saved_params['case_dir'],f'p_tseries_{os.path.splitext(saved_params['resonator_fname'])[0]}.pdf'),format = 'pdf')
    plt.close()

def plot_psd_ptseries(saved_params,res_params):

    nperseg = saved_params['function_values'].shape[-2]
    dt = np.diff(saved_params['function_values'][0,0,:2,0])[0]
    df = (nperseg*dt)**-1
    f,pxx_baseline = welch(saved_params['baseline_function_values'][saved_params['observers']], fs=dt**-1, window='boxcar', nperseg=nperseg, noverlap=0, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-2, average='mean')
    f,pxx = welch(saved_params['function_values'][saved_params['observers']], fs=dt**-1, window='boxcar', nperseg=nperseg, noverlap=0, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-2, average='mean')
    
    figsize = (6.5*.95,6.5*.95)
    fig, ax = plt.subplots(len(saved_params['observers']),2,figsize = figsize)
    plt.subplots_adjust(left=0.1,right = 0.95,top = 0.95,bottom=0.125,hspace = 0.3,wspace = 0.3)
    for mic_itr in range(len(saved_params['observers'])):
        ax[mic_itr,0].plot(saved_params['baseline_function_values'][saved_params['observers']][mic_itr,0,:,0]/saved_params['baseline_function_values'][saved_params['observers']][mic_itr,0,-1,0],np.roll(saved_params['baseline_function_values'][saved_params['observers']][mic_itr,0,:,-1],-128), linestyle=linestyle[0])
        ax[mic_itr,0].plot(saved_params['function_values'][saved_params['observers']][mic_itr,0,:,0]/saved_params['function_values'][saved_params['observers']][mic_itr,0,-1,0],np.roll(saved_params['function_values'][saved_params['observers']][mic_itr,0,:,-1],-128), linestyle=linestyle[1])
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        t = ax[mic_itr,0].text(0.5, .15, rf'$\Delta$ OASPL={np.round(10*np.log10(np.mean(saved_params['baseline_function_values'][saved_params['observers']][mic_itr,0,:,-1]**2)/20e-6**2)-10*np.log10(np.mean(saved_params['function_values'][saved_params['observers']][mic_itr,0,:,-1]**2)/20e-6**2),1)} dB', ha="center",va="center",transform=ax[mic_itr,0].transAxes,fontsize=9)
        t.set_bbox(dict(facecolor='white' ,edgecolor = 'gray' ,alpha=.6,boxstyle='round'))
        ax[mic_itr,0].set(xlim = [0,1],ylim = [-10,5],title =title[mic_itr])
        if mic_itr !=len(saved_params['observers'])-1:
            ax[mic_itr,0].set_xticklabels([])
        ax[mic_itr,0].grid()
    ax[-1,0].set(xlabel = 'Rev. Fraction')
    ax[int(len(saved_params['observers'])/2),0].set(ylabel = 'p [Pa]')
    for mic_itr in range(len(saved_params['observers'])):
        markerline, stemlines ,baseline= ax[mic_itr,1].stem(f,10*np.log10(pxx_baseline[mic_itr,0,:,-1]*np.diff(f[:2])[0]/20e-6**2))
        stemlines.set(color = default_colors[0])
        markerline.set(color = default_colors[0])
        markerline, stemlines,baseline = ax[mic_itr,1].stem(f,10*np.log10(pxx[mic_itr,0,:,-1]*np.diff(f[:2])[0]/20e-6**2))
        stemlines.set(color = default_colors[1])
        markerline.set(color = default_colors[1])
        ax[mic_itr,1].set(ylim = [0,90],xscale = 'linear',xlim = [100,5e3],title =title[mic_itr])
        if mic_itr !=len(saved_params['observers'])-1:
            ax[mic_itr,1].set_xticklabels([])
    ax[mic_itr,1].grid()
    ax[-1,1].set_xlabel('Frequency [Hz]')
    ax[int(len(saved_params['observers'])/2),1].set_ylabel(r'SPL, dB (re: 20$\mathrm{\mu}$Pa)')
    fig.legend(['Untreated','Treated'],ncol = 2,loc='lower center',bbox_to_anchor=(.5, -0.01))
    plt.savefig(os.path.join(saved_params['case_dir'],f'psd_p_tseries_{os.path.splitext(saved_params['resonator_fname'])[0]}.pdf'),format = 'pdf')
    plt.close()

def plot_geom(saved_params,res_params):

    # patch_vals = np.asarray(saved_params['patch_types'])
    
    patch_vals = np.unique(saved_params['patch_types'][saved_params['patch_filt_ind']])

    bounds = np.zeros(len(patch_vals) + 1)
    bounds[1:-1] = 0.5 * (patch_vals[:-1] + patch_vals[1:])
    bounds[0]  = patch_vals[0] - (bounds[1] - patch_vals[0])
    bounds[-1] = patch_vals[-1] + (patch_vals[-1] - bounds[-2])

    cmap = cm.get_cmap('inferno', len(patch_vals))
    norm = BoundaryNorm(bounds, cmap.N)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for older matplotlib

    fig = plt.figure(figsize=(4.5*.95,2/3*4.5*.95))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0.1, right=.8, bottom=0.1, top=1)


    for i, ind in enumerate(patch_vals[np.asarray(res_params['l'])[patch_vals].squeeze().argsort()]):
        filt_ind = (saved_params['point_filt_ind'][(saved_params['patch_types'] == ind) & saved_params['patch_filt_ind']]).flatten()

        if len(filt_ind) != 0:
            ax.scatter(
                np.abs(saved_params['geometry'].zones[0].nodes[0][filt_ind, 0])*1e3,
                saved_params['geometry'].zones[0].nodes[0][filt_ind, 1]*1e3,
                saved_params['geometry'].zones[0].nodes[0][filt_ind, 2]*1e3,
                c = 'r'
                # c=np.full(len(filt_ind), patch_vals[i]),
                # cmap=cmap,
                # norm=norm
            )


    ax.scatter(np.abs(saved_params['geometry'].zones[0].nodes[0][:,0])*1e3,saved_params['geometry'].zones[0].nodes[0][:,1]*1e3,saved_params['geometry'].zones[0].nodes[0][:,2]*1e3,alpha=0.2,c = 'gray')
    if len(res_params['l'])>1:
        cax = fig.add_axes([0.75, 0.1, 0.03, 0.8])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cax, ticks= 0.5 * (bounds[:-1] + bounds[1:]),location = 'right',orientation = 'vertical',pad = .1,label = rf'$l \ [mm]$',drawedges=True,boundaries=bounds)
        cbar.set_ticklabels(np.sort(np.round(np.asarray(res_params['l'])[patch_vals].squeeze()*1e3,1)))
    ax.set_xlabel(r'$x \ [mm]$', labelpad=20)
    ax.set_ylabel(r'$y \ [mm]$')
    ax.invert_zaxis()
    ax.set_box_aspect(np.abs(np.asarray((np.diff(ax.get_xlim()),np.diff(ax.get_ylim()),np.diff(ax.get_zlim())))).squeeze())
    ax.view_init(elev=-90,azim = 90)
    ax.set_zticks([])
    ax.set_zlabel('')
    ax.invert_xaxis()
    ax.zaxis.line.set_visible(False)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.grid(False)
    plt.savefig(os.path.join(saved_params['case_dir'],f'geom_{os.path.splitext(saved_params['resonator_fname'])[0]}.png'),format = 'png',dpi=400)

    plt.savefig(os.path.join(saved_params['case_dir'],f'geom_{os.path.splitext(saved_params['resonator_fname'])[0]}.pdf'),format = 'pdf',pad_inches=.05)
    plt.close()


    fig = plt.figure(figsize=(4.5*.95,2/3*4.5*.95))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0.1, right=.8, bottom=0.1, top=1)
    for i, ind in enumerate(np.asarray(res_params['l']).squeeze().argsort()):
        filt_ind = (saved_params['point_filt_ind'][(saved_params['patch_types'] == ind) & saved_params['patch_filt_ind']]).flatten()
        y = np.unique(saved_params['geometry'].zones[0].nodes[0][filt_ind, 0])
        
        if len(filt_ind) != 0:
            ax.scatter(
                np.abs(y)*1e3,
                np.zeros(len(y)),
                np.zeros(len(y)),
                c=np.full(len(y), patch_vals[i]),
                cmap=cmap,
                norm=norm
            )
    ax.scatter(np.abs(saved_params['geometry'].zones[0].nodes[0][:,0])*1e3,saved_params['geometry'].zones[0].nodes[0][:,1]*1e3,saved_params['geometry'].zones[0].nodes[0][:,2]*1e3,alpha=0.2,c = 'gray')
    if len(res_params['l'])>1:
        cax = fig.add_axes([0.75, 0.1, 0.03, 0.8])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cax, ticks=saved_params['patch_types'],location = 'right',orientation = 'vertical',pad = .1,label = rf'$l \ [mm]$',drawedges=True,boundaries=bounds)
        cbar.set_ticklabels(np.sort(np.round(np.asarray(res_params['l']).squeeze()*1e3,1)))
    else:
        val = saved_params['patch_types'][0]
        bounds = [val - 0.5, val + 0.5]
        norm = BoundaryNorm(bounds, cmap.N)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.75, 0.5, 0.03, 0.15])
        cbar = fig.colorbar(sm,cax=cax,ticks=[val],boundaries=bounds,drawedges=True,label=rf'$l \ [mm]$' )
        cbar.set_ticklabels([np.round(np.asarray(res_params['l']).squeeze()*1e3, 1) ])


    ax.set_xlabel(r'$x \ [mm]$', labelpad=20)
    ax.set_ylabel(r'$y \ [mm]$')
    ax.invert_zaxis()
    ax.set_box_aspect(np.abs(np.asarray((np.diff(ax.get_xlim()),np.diff(ax.get_ylim()),np.diff(ax.get_zlim())))).squeeze())
    ax.view_init(elev=-90,azim = 90)
    ax.set_zticks([])
    ax.set_zlabel('')
    ax.invert_xaxis()
    ax.zaxis.line.set_visible(False)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.grid(False)
    plt.savefig(os.path.join(saved_params['case_dir'],f'geom_{os.path.splitext(saved_params['resonator_fname'])[0]}_compact.pdf'),format = 'pdf',pad_inches=.05)
    plt.close()



        # leg_labs = [rf'$l={i}~mm$' for i in np.round(np.asarray(res_params['l']).squeeze()[plot_ind],3)*1e3]
    # bounds = np.zeros(len(saved_params['patch_types']) + 1)
    # bounds[0]  = saved_params['patch_types'][0] - (bounds[1] - saved_params['patch_types'][0])
    # bounds[-1] = saved_params['patch_types'][-1] + (saved_params['patch_types'][-1] - bounds[-2])

    # cmap = cm.get_cmap('inferno')
    # get_c = lambda patch_types: cmap((patch_types-bounds[0])/(bounds[-1]-bounds[0]))

    # fig = plt.figure(figsize=(4.5*.95,2/3*4.5*.95))
    # ax = fig.add_subplot(111, projection='3d')
    # fig.subplots_adjust(left=0, right=.9, bottom=0, top=1)
    # for i,ind in enumerate(np.asarray(res_params['l']).squeeze().argsort()):
    #     filt_ind = (saved_params['point_filt_ind'][(saved_params['patch_types']==ind) & saved_params['patch_filt_ind']]).flatten()
    #     if len(filt_ind) !=0:
    #         ax.scatter(np.abs(saved_params['geometry'].zones[0].nodes[0][filt_ind,0]),saved_params['geometry'].zones[0].nodes[0][filt_ind,1],saved_params['geometry'].zones[0].nodes[0][filt_ind,2],c = get_c(i),label = rf'$l={np.round(np.asarray(res_params['l']).squeeze()[ind]*1e3,1)}~mm$')
    # ax.scatter(np.abs(saved_params['geometry'].zones[0].nodes[0][:,0]),saved_params['geometry'].zones[0].nodes[0][:,1],saved_params['geometry'].zones[0].nodes[0][:,2],alpha=0.2,c = 'gray')
    # ax.set_xlabel(r'x [m]', labelpad=20)
    # ax.set_ylabel(r'y [m]')
    # ax.invert_zaxis()
    # ax.set_box_aspect(np.abs(np.asarray((np.diff(ax.get_xlim()),np.diff(ax.get_ylim()),np.diff(ax.get_zlim())))).squeeze())
    # ax.legend(ncol = np.ceil(saved_params['N_patches']/2),loc = 'lower center',alignment='center',fontsize=9,borderaxespad=0.25,columnspacing=0.3)
    # ax.view_init(elev=-90, azim=90,roll = 180)
    # # ax.invert_xaxis()
    # # ax.invert_yaxis()
    # ax.set_zticks([])
    # ax.set_zlabel('')
    # ax.zaxis.line.set_visible(False)
    # ax.grid(False)

