from matplotlib import pyplot
import numpy as np
import matplotlib
import datetime
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
from matplotlib.transforms import offset_copy
from astropy.io import fits
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pylab as pylab
import cmasher as cmr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import copy
import ReadOIFITS as oi
import scipy as sc
from astropy import units as u
import post_organic_dev as post_organic
import post_post_organic as PPO
# from yellowbrick.cluster import SilhouetteVisualizer
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
from shutil import copyfile

class post_process():
    def __init__(self, imgdir, savedir, savename='', epoch = '', obj = '', dirsave_fig_img = './', dirsave_fig_vis ='./',  minfracPCA = 0.7):
        self.imgdir = imgdir
        self.mas2rad = 1/3600 * np.pi / 180. / 1000
        self.savedir = savedir
        self.epoch = epoch
        self.savename = savename
        self.obj = obj
        self.dirsave_fig_img = dirsave_fig_img
        self.dirsave_fig_vis = dirsave_fig_vis
        self.minfracPCA = minfracPCA


    def call(self, specific_model, test_nPCA = True, write = True, test_nk_means = False, ktest_technique = 'silhouette', plot_images = True):
        for j in np.arange(len(specific_model['folders'])):
            subfolders = self.imgdir + specific_model['folders'][j]
            self.read_cube(subfolders)
            self.doPCA(test_nPCA)
            self.doKmeans()
            if test_nk_means == True:
                if ktest_technique == 'elbow':
                    self.check_k_elbow()
                elif ktest_technique == 'silhouette':
                    self.check_k()
            self.giveKmeansmetrics()
            self.group_images()
            self.give_best()
            if write == True:
                self.write_sets(writedir = subfolders)

    def read_cube(self, subfolders):
        hdul = fits.open(subfolders + 'cube.fits')
        self.cube = hdul[0].data
        self.header = hdul[0].header
        metrics = hdul[1].data
        self.fdata = metrics['fdata']
        self.ftot = metrics['ftot']
        self.frgl = metrics['frgl']

    def testPCA(self, cube):
        test_components = 20
        pca = PCA(n_components=test_components)
        cube_pca = pca.fit(cube).transform(cube)
        exp_var_ratio = pca.explained_variance_ratio_
        print(exp_var_ratio)
        cumdist = np.cumsum(exp_var_ratio)
        ind_minfrac = np.where(cumdist > self.minfracPCA)[0]
        print(ind_minfrac)
        min_nPCA = ind_minfrac[0] + 1  # +1 to account for list starting at ind. 0
        print("At least {} PCA components are needed".format(min_nPCA))

    def doPCA(self, test_nPCA):
        cube = copy.deepcopy(self.cube)
        shape = cube.shape
        cube = np.reshape(cube, [shape[0], shape[1] * shape[2]])
        if test_nPCA == True:
            self.testPCA(cube)
        pca = PCA(n_components=self.minfracPCA, random_state=1)
        cube_pca = pca.fit(cube).transform(cube)
        print(f"Explained variance ratio over principal components: {pca.explained_variance_ratio_}")
        print(f"Total explained variance ratio over chosen #components: {sum(pca.explained_variance_ratio_)}")
        self.pca = cube_pca

    def doKmeans(self):
        self.nkmeans = 3
        kmeans = KMeans(n_clusters=self.nkmeans, init = 'k-means++', n_init=500, random_state=1)
        # method for initialisation - k-menas ++ --> selects initial cluster centroids using campling based on empirical PD of the points' contribution to the overall inertia.
        #n_init gives the number of times the k-means algorithm is run with different centroid seeds

        clusters = kmeans.fit_predict(self.pca) # return cluster labels in PCA space
        centers = kmeans.cluster_centers_  # retrieve the centres of the kmeans clusters in PCA space
        self.kmeans = clusters  # assign to appropriate object attributes
        self.centers = centers
        print(centers)  # this shows the pca in the horizontal direction (rows) and the kmeans in the vertical direction (columns)
        print(kmeans.labels_)  # same as clusters

    def iterate_kmeans(self):
        "Do kmeans multiple times, save distances (i.e. stds), and check convergence"
        self.doKmeans()

    def giveKmeansmetrics(self):
        clusters = self.kmeans
        nk = 3
        self.nk = nk
        fdata = self.fdata
        frgl = self.frgl
        fdatacluster, frglcluster = [], []
        for i in np.arange(nk):
            fdatak = fdata[clusters == i]
            frglk = frgl[clusters == i]
            fdatacluster.append(np.median(fdatak))
            frglcluster.append(np.median(frglk))
        self.kmeanfdata = fdatacluster
        self.kmeanfrgl = frglcluster
        # print('median fdata per cluster', fdatacluster)

    def group_images(self):
        uu, idx, counts = np.unique(self.kmeans, return_inverse=True, return_counts = True) # select the unique sets, provide unique array (idx), and the counts of each set
        cube = self.cube
        medians, stds = np.array([]), np.array([])
        for k in uu:
            cub = []
            for image, i in zip(cube, idx):
                if i == k:
                    cub.append(image)
            cub = np.array(cub)
            median_img = np.median(cub, axis=0)
            median = median_img / np.sum(median_img)
            medians = np.append(medians, median)
            stds = np.append(stds, np.std(cub, axis=0))
            #self.nsets = counts[k]
        self.medians = medians  # median images per cluster group
        self.stds = stds  # standard deviation image per cluster group
        self.counts = counts  # amount of images in each group
        # self.groups = groups  # a cube of images per cluster group


    def scee_plot_k(self):
        "Fucntion to plot a scee plot and determine k visually from the elbow"
        self.check_k()
        fig, ax = plt.subplots(1, 2)
        ax.plot(range(1, self.nk_test+1), self.sse, linestyle='--', marker='o')
        ax.set_xticks(range(1, self.nk_test+1))
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("SSE")
        fig.savefig(self.savedir +'test_nk_kmeans_n_init=1, max_iter=500_rand_state=1'+'.jpg', bbox_inches='tight')

    def check_k_elbow(self):
        "Function to run kmeans for a set few times to see whether it reaches the elbow"
        self.nk_test = 5
        iterations = self.nk_test

        sse = []
        fig, ax = plt.subplots(3, 2, figsize=(15, 8))
        for k in range(1, iterations+1):
            kmeans = KMeans(n_clusters=k, init = 'k-means++', n_init=500, random_state=1)
            kmeans.fit(self.pca)
            # sse = visualizer.fit()
        # fig.savefig(self.savedir + 'test_nk_kmeans_silhouette' + '.jpg', bbox_inches='tight')
        fig.savefig(self.savedir + 'test_nk_kmeans_silhouette' + '.jpg', bbox_inches='tight')

        self.sse = sse

    def plot_silhouette(self, nk):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        y_lower = 0
        for i in range(self.nk_test):
            ith_cluster_silhouette_values = self.silhouette_vals[self.cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            # color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                alpha=0.7,
            )
            y_lower = y_upper + 10
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")
            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            ax1.axvline(x=self.silhouette_avg, color="red", linestyle="--")
        ax2.scatter(self.pca[:, 0], self.pca[:, 1], marker=".", s=30, lw=0, alpha=0.7,  edgecolor="k")
        centers = self.kmeans.cluster_centers_
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        plt.suptitle(f' Silhouette analysis using k = {nk}, score = {self.silhouette_avg}', fontsize=16, fontweight='semibold')
        fig.savefig(self.savedir + f'test_nk_kmeans_silhouette{nk}.jpg', bbox_inches='tight')

    def check_k(self):
        "Function to run kmeans for a set few times to see whether it reaches the elbow"
        self.nk_test = 10
        iterations = self.nk_test
        for k in range(2, iterations+1):
            self.kmeans = KMeans(n_clusters=k, init = 'k-means++', n_init=500, random_state=10)
            self.cluster_labels = self.kmeans.fit_predict(self.pca)
            silhouette_avg = silhouette_score(self.pca, self.cluster_labels)
            sample_silhouette_values = silhouette_samples(self.pca, self.cluster_labels) #compute the silhouette scores for each sample
            self.silhouette_vals = sample_silhouette_values
            self.silhouette_avg =  silhouette_avg
            self.plot_silhouette(k)



    def group_images2(self):
        idx = list(self.kmeans)  # get cluster indices for images in cube
        cube = self.cube
        medians, stds = [], []
        counts, groups = [], []
        for k in np.arange(self.nkmeans):
            counts.append(idx.count(k))
            cub = []
            for image, i in zip(cube, idx):
                if i == k:
                    cub.append(image)
            cub = np.array(cub)
            groups.append(cub)
            median_img = np.median(cub, axis=0)
            median_img /= np.sum(median_img)
            medians.append(median_img)
            stds.append(np.std(cub, axis=0))
        self.medians = medians  # median images per cluster group
        self.stds = stds  # standard deviation image per cluster group
        self.counts = counts  # amount of images in each group
        self.groups = groups  # a cube of images per cluster group

    def give_best(self):
        fdata = self.fdata
        ftot = self.ftot
        frgl = self.frgl
        cube = self.cube
        nbest = self.nk  # number of best images to single out of the cube
        print(nbest)

        idx = ftot.argsort()  # get indices which would sort the ftot loss array

        zebest, zemetrics = [], []  # arrays to store in the best
        for i in np.arange(nbest):
            idx_nbest = np.where(idx == i)[0]  # index for the nth best image
            zebest.extend(cube[idx_nbest, :, :])  # get nth best image
            zemetrics.append(np.array([ftot[idx_nbest][0], fdata[idx_nbest][0], frgl[idx_nbest][0]]))  # get its metrics

        self.bestimages = np.array(zebest)  # store best images and metrics in the appropriate attributes
        self.bestmetrics = np.array(zemetrics)
        print(zemetrics)
        # self.plot_best()  # plot the n best images'

    def write_sets(self, writedir):
        best = self.bestimages  # get best image
        print(best)
        metrics = self.bestmetrics  # get their metrics
        ftot, fdata, frgl = metrics[:,0], metrics[:,1], metrics[:,2]  # get lists of the metrics terms
        for i in np.arange(self.nk):
            hdr = self.header  # take the header from the primary cube header
            hdr['NBEST'] = i+1
            hdr['CDELT1'] = hdr['CDELT1']
            hdr['FTOT'] = ftot[i]
            hdr['FDATA'] = fdata[i]
            hdr['FRGL'] = frgl[i]
            img = best[i,:,:]
            hdu = fits.PrimaryHDU(img, header = hdr)
            hdul = fits.HDUList([hdu])
            hdul.writeto(os.path.join(writedir, f'best_image{i+1}.fits'), overwrite=True)

    def read_header_image_geo(self, subfolders, fits_files):
        hdul = fits.open(subfolders + '/' + fits_files)
        self.n = hdul[0].header['NAXIS2']
        self.ps = hdul[0].header['CDELT2']
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

    # def plot_image(self, savefig, axes = 'off'):
    #     xr = np.arange(self.n) + 1
    #     x = self.x1 - (xr - self.cenpixx) * self.ps
    #     y = self.y1 - (xr - self.cenpixy) * self.ps
    #     d = self.n * self.ps / 2.
    #     unit_abr = 'mas'
    #     dpi = 300
    #     for i in np.arange(self.nk):
    #     ## Plot the image
    #         fig, ax = plt.subplots()
    #         cs = ax.imshow(self.image/np.max(self.image), extent=[d, -d, d, -d], cmap='inferno', vmin=0.0) #inferno
    #         ax.scatter(-self.ps/2, self.ps/2, color = 'teal', s =75) #primary
    #         if self.x2!=0 and self.y2!=0:
    #             ax.scatter(self.x2, self.y2, color = 'lightyellow', s =6) #secondary
    #         ax.axis([d, -d, -d, d])
    #         ax.set_ylim(-17,17)
    #         ax.set_xlim(17,-17)
    #
    #         # ax.set_ylim(-9,9)
    #         # ax.set_xlim(9,-9)
    #         ax.set_xlabel(r'$\Delta \alpha$ ({})'.format(unit_abr))
    #         ax.set_ylabel(r'$\Delta \delta$ ({})'.format(unit_abr))
    #         divider = make_axes_locatable(ax)
    #         if savefig == True:
    #             if axes == 'off':
    #                 ax.set_axis_off()
    #                 fig.savefig(self.dirsave_fig_img + self.epoch  + self.savename + '_' + modelshort + self.obj +'_no_axis' + '.jpg', bbox_inches='tight', dpi=dpi)
    #             else:
    #                 cax = divider.append_axes("right", size="5%", pad=0.05)
    #                 plt.colorbar(cs, cax=cax)
    #                 ax.set_xticks([15, 10, 5, 0, -10, -15])
    #                 # plt.savefig(dir_save+obj+addition+'_best.pdf')
    #                 fig.savefig(dirsave_fig_vis_img + self.epoch + self.savename + '_' + modelshort + obj + '_withaxes' + '.jpg', bbox_inches='tight', dpi=dpi)
    #
    # def plot_model_vs_data(self, modelshort, obj, dirsave_fig_vis, savefig, residuals):
    #     fig, ((axV2, axT3), (axRESV2, axREST3)) = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(9, 4.7),
    #                                                            gridspec_kw={'height_ratios': [5, 1]})
    #     resV2 = self.model_V2 - self.data_V2
    #     resV2 /= self.data_V2err
    #     resCP = self.model_CP - self.data_CP
    #     resCP /= self.data_CPerr
    #     if residuals == True:
    #         fig.subplots_adjust(hspace=0, wspace=0.28)
    #         axRESV2.set_xlabel(r'B (M$\lambda$)')
    #         axRESV2.set_ylim(-9, 9)
    #         axRESV2.axhline(0, linestyle='--', linewidth=0.9, color='darkgrey', zorder=0)
    #         axREST3.set_xlabel(r'B (M$\lambda$)')
    #         axREST3.axhline(0, linestyle='--', linewidth=0.9, color='darkgrey', zorder=0)
    #         axREST3.set_ylim(-30, 30)
    #         axRESV2.set_ylabel(r'res. $\sigma_\mathrm{V^2}$')
    #         axREST3.set_ylabel(r'res. $\sigma_\mathrm{CP}$')
    #         axRESV2.scatter(self.data_base * 1e-6,
    #                         resV2, color='darkgoldenrod', label='model', s=3, zorder=2)
    #         axREST3.scatter(self.data_Bmax * 1e-6,
    #                         resCP, color='darkgoldenrod', label='model', s=3, zorder=2)
    #
    #     #Plot visibility
    #     axV2.errorbar(self.data_base * 1e-6,
    #                   self.data_V2,
    #                   yerr=self.data_V2err,
    #                   linestyle='none', marker='.', capsize=1,
    #                   label='data', color='grey', zorder=0)
    #     axV2.scatter(self.data_base * 1e-6 , self.model_V2, color = 'darkgoldenrod', s = 10, zorder =1)
    #
    #     #Plot Phase
    #     axT3.errorbar(self.data_Bmax * 1e-6,
    #                   self.data_CP,
    #                   yerr=self.data_CPerr,
    #                   linestyle='none', marker='.', capsize=1,
    #                   label='data', color='grey', zorder=0)
    #     axT3.scatter(self.data_Bmax * 1e-6, self.model_CP, color='darkgoldenrod', s=10, zorder=1)
    #
    #     #Other settings
    #     axV2.set_ylabel(r'V$^2$')
    #     axT3.set_ylabel(r'CP (deg)')
    #     axV2.set_ylim(0.01,1)
    #     axT3.set_ylim(-24,24)
    #
    #     if savefig == True:
    #         dpi = 300
    #         fig.savefig(dirsave_fig_vis + self.epoch + self.savename +'_'+ modelshort +obj + '.jpg', bbox_inches='tight', dpi=dpi)


    def takefits(self, ignore=None, savefits=True):
        subfolders = self.find_dir()[1]
        fits_files = [x for x in os.listdir(subfolders) if x.endswith(".fits") and x.startswith('Images_set')]
        if ignore != None:
            fits_files.remove(ignore)
        n = len(fits_files)
        if n > 0:
            hdulist = fits.open(subfolders + '/' + fits_files[0])
            hdr = hdulist[0].header
            data = hdulist[0].data
            header_copy = hdr.copy()
        for i in range(1, n):
            hdulist = fits.open(subfolders + '/' + fits_files[i])
            data += hdulist[0].data
            hdulist.close()
        medianImage = data / n
        if savefits == True:
            fits.writeto(subfolders + '/newImage.fits', medianImage, header_copy, overwrite=True)
        # medianImage /= np.sum(medianImage)
        return medianImage
