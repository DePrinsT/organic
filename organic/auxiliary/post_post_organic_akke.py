from matplotlib import pyplot
import numpy as np
import matplotlib
import datetime
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.transforms import offset_copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
import os
import matplotlib.pylab as pylab
import copy
import organic.auxiliary.ReadOIFITS as oi
import scipy as sc
from astropy import units as u
# from yellowbrick.cluster import SilhouetteVisualizer
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
from shutil import copyfile

class statistics_organic:

    def __init__(self, imgdir, savedir, savename='', epoch = ''):
        self.imgdir = imgdir
        self.mas2rad = 1/3600 * np.pi / 180. / 1000
        self.savedir = savedir
        self.epoch = epoch
        self.savename = savename

    def general(self, datadir, datafiles, specific_model, obj='', dirsave_fig_vis='/', dirsave_fig_img='/',
                savefig=True, residuals=True, savestats=True):
        """
        datadir and datafiles; specify the interferometric data to read
        specific model: assumes a dictionary of models and the fits file(s) it should take. The dictionary should have an entry 'folders', listing all the specified model dirs,
        and a 'files', specifying all fits files to be taken into account.
        The general dir to the images is already specified by self.imgdir, so only specify here the subdirectories to the .fits files.
        """

        self.read_data(datadir, datafiles)

        if len(specific_model['folders']) == 1:
            subfolders = self.imgdir + specific_model['folders'][0]
            fits_files = specific_model['files']
            #print(f"FITS FILES: {fits_files}")
            for i in np.arange(len(fits_files)):
                # DIRTY TRY EXCEPT CLAUSE FOR IF YOU'RE RUNNING A SINGLE MODEL
                try:
                    modelshort = specific_model['folders'][0].split('/')[-2] + '__' + fits_files[i].split('.fits')[0]
                    self.modelshort = modelshort
                except:
                    modelshort = "test_"
                    self.modelshort="test_"
                self.calc_imgobs(subfolders, fits_files[i])
                self.getstats(subfolders, fits_files[i], savestats)
                self.plot_model_vs_data(obj, dirsave_fig_vis, savefig,residuals)
                self.plot_image(obj, dirsave_fig_img, savefig)
        elif len(specific_model['folders']) > 1:
            fits_files = specific_model['files']
            for j in np.arange(len(specific_model['folders'])):
                subfolders = self.imgdir + specific_model['folders'][j]
                for i in np.arange(len(fits_files)):
                    modelshort = specific_model['folders'][j].split('/')[-2] + '__' + fits_files[i].split('.fits')[0]
                    self.modelshort = modelshort
                    self.calc_imgobs(subfolders, fits_files[i])
                    self.getstats(subfolders, fits_files[i], savestats)
                    self.plot_model_vs_data(obj, dirsave_fig_vis, savefig,
                                            residuals)
                    self.plot_image(obj, dirsave_fig_img, savefig)
        else:
            print('Please check your input of specific_model; it should be a dictionary ... ')

    def read_data(self, datadir, datafiles):
        data = oi.read(datadir, datafiles)
        data_wave, data_V2, data_V2err, data_u, data_v, data_base = (np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
        # "a" for append
        #Visibilities
        for i in np.arange(len(data.vis2)):
            data_wave_a = data.vis2[i].effwave
            data_V2_a, data_V2err_a = data.vis2[i].vis2data, data.vis2[i].vis2err
            data_u_a = data.vis2[i].uf
            data_v_a = data.vis2[i].vf
            data_base_a = np.sqrt(data_u_a ** 2 + data_v_a ** 2)
            data_wave = np.append(data_wave_a, data_wave)
            data_V2 = np.append(data_V2_a, data_V2)
            data_V2err = np.append(data_V2err_a, data_V2err)
            data_u = np.append(data_u_a, data_u)
            data_v = np.append(data_v_a, data_v)
            data_base = np.append(data_base_a, data_base)
        #Closure phases
        data_Bmax, data_CP, data_CPerr, data_u1, data_u2, data_u3, data_v1, data_v2, data_v3, data_wavecp = (
            np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])  )
        for i in np.arange(len(data.t3)):
            data_CP_a, data_CPerr_a = data.t3[i].t3phi, data.t3[i].t3phierr
            data_u1_a = data.t3[i].uf1
            data_u2_a = data.t3[i].uf2
            data_v1_a = data.t3[i].vf1
            data_v2_a = data.t3[i].vf2
            data_u3_a = data_u1_a + data_u2_a
            data_v3_a = data_v1_a + data_v2_a #+ or -, I seemed to have a - in December
            data_wavecp_a = data.t3[i].effwave
            B1 = np.sqrt(data_u1_a ** 2 + data_v1_a ** 2)
            B2 = np.sqrt(data_u2_a ** 2 + data_v2_a ** 2)
            B3 = np.sqrt(data_u3_a ** 2 + data_v3_a ** 2)
            Bmax_a = np.maximum(B1, B2, B3)
            data_u1 = np.append(data_u1_a, data_u1)
            data_v1 = np.append(data_v1_a, data_v1)
            data_u2 = np.append(data_u2_a, data_u2)
            data_v2 = np.append(data_v2_a, data_v2)
            data_u3 = np.append(data_u3_a, data_u3)
            data_v3 = np.append(data_v3_a, data_v3)
            data_wavecp = np.append(data_wavecp_a, data_wavecp)
            data_Bmax = np.append(data_Bmax, Bmax_a)
            data_CP = np.append(data_CP, data_CP_a)
            data_CPerr = np.append(data_CPerr, data_CPerr_a)
        self.data_wave = data_wave
        self.data_V2 = data_V2
        self.data_V2err = data_V2err
        self.data_u = data_u
        self.data_v = data_v
        self.data_u1 = data_u1
        self.data_v1 = data_v1
        self.data_u2 = data_u2
        self.data_v2 = data_v2
        self.data_u3 = data_u3
        self.data_v3 = data_v3
        self.data_wavecp = data_wavecp
        self.data_base = data_base
        self.data_CP = data_CP
        self.data_CPerr = data_CPerr
        self.data_Bmax = data_Bmax
        print('=checkpoint')

    def calc_imgobs(self, subfolders, fits_files):
        interp = self.read_image(subfolders, fits_files)
        vis2 = self.comp_obser(interp, self.data_u, self.data_v, self.data_wave) #complex visibility
        vcomp1_abs = self.comp_obser(interp, self.data_u1, self.data_v1, self.data_wavecp)
        vcomp2_abs = self.comp_obser(interp, self.data_u2, self.data_v2, self.data_wavecp)
        vcomp3_abs = self.comp_obser(interp, self.data_u3, self.data_v3, self.data_wavecp)
        # plt.figure()
        # t = vcomp1_abs * vcomp2_abs * np.conjugate(vcomp3_abs)
        # plt.scatter(t.real, t.imag)
        # plt.savefig('/home/acorpora/PIONIER/IRAS08_time-series/Image_reconstructions/testing' + '.jpg')
        # print(t)
        # print(np.angle(t, deg = True))
        # print(t[-1])
        # print(np.angle(t[-1], deg = True))
        self.model_CP =   (np.angle(vcomp1_abs * vcomp2_abs * np.conjugate(vcomp3_abs), deg=True)) #TO CHECK (- sign or not), which direction are the phases
        self.model_V2 = np.abs(vis2) ** 2

    def read_image(self, subfolders, fits_files):
        self.read_header_image_geo(subfolders, fits_files)
        img_fft, uf, vf = self.doFFT(subfolders, fits_files)
        interp = self.image_fft_comp_vis_interpolator([img_fft], uf, vf)
        return interp

    def comp_obser(self, fft_interp, u, v, wave):
        Vimg = np.array([])
        for ui, vi in zip(u, v):
            imgFT = fft_interp((vi, ui))
            Vimg = np.append(Vimg, imgFT)

        # add sparco on top
        f1 = self.fs * np.power(wave / self.wave0, -4)  # flux fraction of primary
        f2 = self.fsec  * np.power(wave / self.wave0, self.dsec)  # flux fraction of secondary
        fimg = (1 - self.fs - self.fsec) * np.power(wave / self.wave0, self.denv)  # flux fraction of the environment
        # visibility of the primary
        V1 = 2 * sc.special.jv(1, np.pi * self.UD * self.mas2rad * np.sqrt(u** 2 + v** 2)) / (
                    np.pi * self.UD * self.mas2rad * np.sqrt(u ** 2 + v** 2))
        alpha = self.x2 * self.mas2rad #rad
        delta = self.y2 * self.mas2rad #rad
        # visibility of the secondary
        V2 = np.exp(-2j * np.pi * (u * alpha + v * delta))  # To CHECK the direction!!!
        # total visibility
        Vtot = (fimg * Vimg + f1 * V1 + f2 * V2  )/ (fimg+ f1+ f2)
        return Vtot

    def read_header_image_geo(self, subfolders, fits_files):
        hdul = fits.open(subfolders + '/' + fits_files)
        self.n = hdul[0].header['NAXIS2']
        self.ps = hdul[0].header['CDELT2']
        print("PS!!!", self.ps)
        self.x1 = hdul[0].header['CRVAL1']
        self.y1 = hdul[0].header['CRVAL2']
        self.cenpixx = hdul[0].header['CRPIX1']
        self.cenpixy = hdul[0].header['CRPIX2']
        self.x2 = hdul[0].header['SDEX2']
        self.y2 = hdul[0].header['SDEY2']
        self.fs = hdul[0].header['SFLU1']
        self.fsec = hdul[0].header['SFLU2']
        self.UD = hdul[0].header['SUD1']
        self.denv = hdul[0].header['SIND0']
        self.dprim = hdul[0].header['SIND1']
        self.dsec = hdul[0].header['SIND2']
        self.img_wave0 = hdul[0].header['SWAVE0']
        self.totflux = hdul[0].header['FTOT']

    def doFFT(self, subfolders, fits_files):
        # for i in fits_files:
        hdul = fits.open(subfolders + '/' + fits_files)
        self.image = hdul[0].data
        img = hdul[0].data
        #img = np.flip(img, axis=0)

        #img_flipped = hdul[0].data
        #img_flipped = np.flip(img_flipped, axis=0)

        #print("PLOTTING IMG READ IN BY AKKE")
        #fig, ax = pyplot.subplots()
        #ax.imshow(img)
        #plt.show()

        #print("PLOTTING IMG FLIPPED")
        #fig, ax = pyplot.subplots()
        #ax.imshow(img_flipped)
        #plt.show()

        self.wave0 = 1.65e-6
        # img = self.add_stars(subfolders, fits_files)

        img /= np.sum(img)
        img_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img)))  # perform complex FFT
        w_x = - np.fft.fftshift(np.fft.fftfreq(img_fft.shape[1]))  # also use fftshift so the 0 frequency
        w_y = - np.fft.fftshift(np.fft.fftfreq(img_fft.shape[0]))  # lies in the middle of the returned array

        uf = w_x / (self.ps * self.mas2rad)  # 1/rad  # spatial frequencies in units of 1/radian
        vf = w_y / (self.ps * self.mas2rad)
        # img_fft = self.add_pointsource(img_fft, uf, vf, 0, 0, self.fs)
        # img_fft = self.add_pointsource(img_fft, uf, vf, self.x2, self.y2, self.fsec)
        # img_fft = self.add_back(img_fft)
        return img_fft, uf, vf

    def image_fft_comp_vis_interpolator(self, img_ffts, uf, vf):
        if len(img_ffts) == 1:  # single image -> monochromatic emission model
            img = img_ffts[0]
            interp_method = 'cubic'
            interpolator = sc.interpolate.RegularGridInterpolator((vf, uf), img, method=interp_method,
                                                                  bounds_error=False,
                                                                  fill_value=None)  # make interpol absolute FFT
        return interpolator


    def getstats(self, subfolders, fits_files, savestats):
        try:
            if self.data_V2[0]:
                chi2v2 = ((self.model_V2 - self.data_V2) / (self.data_V2err)) ** 2
                chi2v2 = np.sum(chi2v2) / len(self.data_V2)
            else:
                raise IndexError
        except IndexError:
            print('No OIVIS2 table detected')

        try:
            if self.data_CP[0]:
                chi2cp = (self.model_CP - self.data_CP) ** 2 / (self.data_CPerr) ** 2
                chi2cp = np.sum(chi2cp) / len(self.data_CP)
            else:
                raise IndexError
        except IndexError:
            print('No OIT3 table detected')

        try:
            if self.data_CP[0] and self.data_V2[0]:
                chi2v2_2 = ((self.model_V2 - self.data_V2) / (self.data_V2err)) ** 2
                chi2cp_2 = (self.model_CP - self.data_CP) ** 2 / (self.data_CPerr) ** 2
                chi2tot = np.sum(chi2v2_2)+ np.sum(chi2cp_2)
                chi2tot = chi2tot / (len(self.data_CP)+len(self.data_V2))
            else:
                raise IndexError
        except IndexError:
            print('No OIT3 or OIVIS2 table detected. Please check')


        print('chi2_V2', chi2v2)
        print('chi2_CP', chi2cp)
        print('chi2_total', chi2tot)
        if savestats == True:
            self.savestats(subfolders, fits_files, chi2v2, chi2cp)

    def savestats(self, subfolders, fits_files, chi2v2, chi2cp):
        modelname = [x for x in subfolders.split('/') if "img_rec" in x][0]
        with open(self.savedir+'all_images_stats'+self.epoch+'.csv', 'a') as f:
            f.write(modelname + '   ' + fits_files + '   ' +  str(chi2v2) + '   ' + str(chi2cp) + '\n')

    def plot_model_vs_data(self, obj, dirsave_fig_vis, savefig, residuals):
        fig, ((axV2, axT3), (axRESV2, axREST3)) = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(9, 4.7),
                                                               gridspec_kw={'height_ratios': [5, 1]})
        resV2 = self.model_V2 - self.data_V2
        resV2 /= self.data_V2err
        resCP = self.model_CP - self.data_CP
        resCP /= self.data_CPerr
        if residuals == True:
            fig.subplots_adjust(hspace=0, wspace=0.28)
            axRESV2.set_xlabel(r'B (M$\lambda$)')
            axRESV2.set_ylim(-9, 9)
            axRESV2.axhline(0, linestyle='--', linewidth=0.9, color='darkgrey', zorder=0)
            axREST3.set_xlabel(r'B (M$\lambda$)')
            axREST3.axhline(0, linestyle='--', linewidth=0.9, color='darkgrey', zorder=0)
            axREST3.set_ylim(-30, 30)
            axRESV2.set_ylabel(r'res. $\sigma_\mathrm{V^2}$')
            axREST3.set_ylabel(r'res. $\sigma_\mathrm{CP}$')
            axRESV2.scatter(self.data_base * 1e-6,
                            resV2, color='darkgoldenrod', label='model', s=3, zorder=2)
            axREST3.scatter(self.data_Bmax * 1e-6,
                            resCP, color='darkgoldenrod', label='model', s=3, zorder=2)

        #Plot visibility
        axV2.errorbar(self.data_base * 1e-6,
                      self.data_V2,
                      yerr=self.data_V2err,
                      linestyle='none', marker='.', capsize=1,
                      label='data', color='grey', zorder=0)
        axV2.scatter(self.data_base * 1e-6 , self.model_V2, color = 'darkgoldenrod', s = 10, zorder =1)

        #Plot Phase
        axT3.errorbar(self.data_Bmax * 1e-6,
                      self.data_CP,
                      yerr=self.data_CPerr,
                      linestyle='none', marker='.', capsize=1,
                      label='data', color='grey', zorder=0)
        axT3.scatter(self.data_Bmax * 1e-6, self.model_CP, color='darkgoldenrod', s=10, zorder=1)

        #Other settings
        axV2.set_ylabel(r'V$^2$')
        axT3.set_ylabel(r'CP (deg)')
        axV2.set_ylim(0.01,1)
        axT3.set_ylim(-24,24)

        if savefig == True:
            dpi = 300
            fig.savefig(dirsave_fig_vis + self.epoch + self.savename +'_'+ self.modelshort +obj + '_observables.jpg', bbox_inches='tight', dpi=dpi)

    def plot_image(self, obj, dirsave_fig_img, savefig, axes = 'off'):
        xr = np.arange(self.n)+1
        x = self.x1 -(xr - self.cenpixx) * self.ps
        y = self.y1- (xr - self.cenpixy) * self.ps
        d = self.n * self.ps / 2.
        unit_abr = 'mas'
        dpi = 300

        ## Plot the image
        fig, ax = plt.subplots()
        cs = ax.imshow(self.image/np.max(self.image), extent=[d, -d, d, -d], cmap='inferno', vmin=0.0) #inferno
        ax.scatter(-self.ps/2, self.ps/2, color = 'teal', s =75) #primary
        if self.x2!=0 and self.y2!=0:
            ax.scatter(self.x2, self.y2, color = 'lightyellow', s =6) #secondary
        ax.axis([d, -d, -d, d])
        ax.set_ylim(-17,17)
        ax.set_xlim(17,-17)

        # ax.set_ylim(-9,9)
        # ax.set_xlim(9,-9)
        ax.set_xlabel(r'$\Delta \alpha$ ({})'.format(unit_abr))
        ax.set_ylabel(r'$\Delta \delta$ ({})'.format(unit_abr))
        divider = make_axes_locatable(ax)
        if savefig == True:
            if axes == 'off':
                ax.set_axis_off()
                # plt.savefig(dir_save+obj+addition+'_noaxis.pdf')
                fig.savefig(dirsave_fig_img + self.epoch  + self.savename + '_' + self.modelshort + obj +'_img_no_axis' + '.jpg', bbox_inches='tight', dpi=dpi)
            else:
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(cs, cax=cax)
                ax.set_xticks([15, 10, 5, 0, -10, -15])
               # plt.savefig(dir_save+obj+addition+'_best.pdf')
                fig.savefig(dirsave_fig_vis_img + self.epoch + self.savename + '_' + self.modelshort + obj + '_img_withaxes' + '.jpg', bbox_inches='tight', dpi=dpi)

    def plot_contours(self):
        n = 2 #plot contour levels of 2 sigma
        contour_levels = n * self.stds





# 3 def add_pointsource(self, img, uf, vf, x, y, f):
#         # define numpy arrays with coordinates for pixel centres in milli-arcsecond
#         coords_pix_x = (
#                 np.linspace(self.n / 2 - 0.5, -self.n / 2 + 0.5, self.n)
#                 * self.ps
#                 * 1/self.mas2rad
#         )
#         coords_pix_y = coords_pix_x
#         coords_pix_mesh_x, coords_pix_mesh_y = np.meshgrid(coords_pix_x, coords_pix_y)  # create a meshgrid
#         distances = np.sqrt((coords_pix_mesh_x -x) ** 2 + (coords_pix_mesh_y - y) ** 2)  # calc dist pixels to point
#         min_dist_indices = np.where(distances == np.min(distances))  # retrieve indices of where distance is minimum
#         min_ind_x, min_ind_y = min_dist_indices[1][0], min_dist_indices[0][0]
#         img[min_ind_y][min_ind_x] += f # add point source flux to the nearest pixel
#         # define meshgrid for spatial frequencies in units of 1/milli-arcsecond
#         freq_mesh_u, freq_mesh_v = np.meshgrid(uf * 1/self.mas2rad, vf * 1/self.mas2rad)
#         # add the complex contribution of the secondary to the stored FFT
#         img += f * np.exp(-2j * np.pi * (freq_mesh_u * x + freq_mesh_v * y))
#         return img
#
#
#     def add_back(self, img):
#         wave = self.data_wave
#         # define numpy arrays with coordinates for pixel centres in milli-arcsecond
#         coords_pix_x = (
#                 np.linspace(self.n / 2 - 0.5, -self.n / 2 + 0.5, self.n)
#                 * self.ps
#                 * 1/self.mas2rad
#         )
#         coords_pix_y = coords_pix_x
#
#         coords_pix_mesh_x, coords_pix_mesh_y = np.meshgrid(coords_pix_x, coords_pix_y)  # create a meshgrid
#         f = 0.18 * np.power (wave / self.wave0, -1.88)
#         print(np.shape(img))
#         for i in np.arange(self.n):
#             for j in np.arange(self.n):
#                 img[i][j] += f/(self.n**2) # add point source flux to the nearest pixel
#
#         # define meshgrid for spatial frequencies in units of 1/milli-arcsecond
#         #freq_mesh_u, freq_mesh_v = np.meshgrid(uf * 1/self.mas2rad, vf * 1/self.mas2rad)
#         # add the complex contribution of the secondary to the stored FFT
#         #img += f * np.exp(-2j * np.pi * (freq_mesh_u * x + freq_mesh_v * y))
#         return img
