from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from astropy.io import fits
import sys
import matplotlib.pyplot as plt
import organic.auxiliary.ReadOIFITS as oi
import numpy as np
import scipy as sc
import os


class Data:
    def __init__(self, dir, file):
        self.dir = dir
        self.file = file
        self.read_data()

    def read_data(self):
        data = oi.read(self.dir, self.file)
        dataObj = data.givedataJK()

        V2observed, V2err = dataObj['v2']
        nV2 = len(V2err)

        CPobserved, CPerr = dataObj['cp']
        nCP = len(CPerr)

        u, u1, u2, u3 = dataObj['u']
        v, v1, v2, v3 = dataObj['v']

        waveV2 = dataObj['wave'][0]
        waveCP = dataObj['wave'][1]

        V2 = V2observed#conversion to tensor
        V2err = V2err#conversion to tensor
        CP = CPobserved*np.pi/180 #conversion to radian & cast to tensor
        CPerr = CPerr*np.pi/180 #conversion to radian & cast to tensor
        waveV2 = waveV2 #conversion to tensor
        waveCP = waveCP #conversion to tensor

        self.nV2 = nV2
        self.nCP = nCP
        self.V2 = V2
        self.V2err = V2err
        self.CP = CP
        self.CPerr = CPerr
        self.waveV2 = waveV2
        self.waveCP = waveCP
        self.u = u
        self.u1 = u1
        self.u2 = u2
        self.u3 = u3
        self.v = v
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3


    def get_data(self):
        return self.V2, self.V2err, self.CP, self.CPerr, self.waveV2, self.waveCP,self.u, self.u1, self.u2, self.u3, self.v, self.v1, self.v2, self.v3


    def get_bootstrap(self):
        V2selection = np.random.randint(0,self.nV2,self.nV2)
        newV2, newV2err = self.V2[V2selection], self.V2err[V2selection]
        CPselection = np.random.randint(0,self.nCP,self.nCP)
        newCP, newCPerr = self.CP[CPselection], self.CPerr[CPselection]
        newu, newu1, newu2, newu3 = self.u[V2selection], self.u1[CPselection], self.u2[CPselection], self.u3[CPselection]
        newv, newv1, newv2, newv3 = self.v[V2selection], self.v1[CPselection], self.v2[CPselection], self.v3[CPselection]
        newwavelV2 = self.waveV2[V2selection]
        newwavelCP = self.waveCP[CPselection]
        return newV2, newV2err, newCP, newCPerr, newwaveV2, newwaveCP, newu, newu1, newu2, newu3, newv, newv1, newv2, newv3



class cube:
    def __init__(self, dir, file = 'Cube.fits', nPCA=3, nKmeans=3, nbest=6):
        self.dir = dir
        self.file = file
        self.nKmeans = nKmeans
        self.nPCA = nPCA
        self.nbest = nbest
        self.read()
        self.doPCA()
        self.doKmeans()
        #self.plotPCA()
        self.plotKmeans()
        self.plotImages()

    def read(self):
        path = os.path.join(self.dir, self.file)
        hdul = fits.open(path)
        cube = hdul[0].data
        self.hdr = hdul[0].header
        self.ps = hdul[0].header['CDELT2']
        self.n = hdul[0].header['NAXIS2']
        try:
            self.x2 = hdul[0].header['SDEX2']
            self.y2 = hdul[0].header['SDEY2']
            self.fs = hdul[0].header['SFLU1']
            self.f2 = hdul[0].header['SFLU2']
            self.UD = hdul[0].header['SUD1']
            self.denv = hdul[0].header['SIND0']
            self.dsec = hdul[0].header['SIND2']
            self.cube = cube
            # self.x2 = hdul[0].header['X']
            # self.y2 = hdul[0].header['Y']
            # self.fs = hdul[0].header['UDF']
            # self.f2 = hdul[0].header['PF']
            # self.UD = hdul[0].header['UDD']
            # self.denv = hdul[0].header['DENV']
            # self.dsec = hdul[0].header['DSEC']
            # self.cube = cube
        except:
            self.x2 = hdul[0].header['XBVAL']
            self.y2 = hdul[0].header['YBVAL']
            self.fs = hdul[0].header['FSVAL']*100
            self.f2 = hdul[0].header['FBVAL']*100
            self.UD = hdul[0].header['UDVAL']
            self.denv = hdul[0].header['DEVAL']
            self.dsec = hdul[0].header['DBVAL']
            self.cube = cube

        # Read the metrics associated to each image.
        metrics = hdul[1].data
        self.fdata = metrics['fdata']
        self.ftot = metrics['ftot']
        self.frgl = metrics['fdiscriminator']


        self.set_coord()

    def doPCA(self):
        cube = self.cube
        shape = cube.shape
        cube = np.reshape(cube, [shape[0], shape[1]*shape[2]])
        pca = PCA(n_components = self.nPCA)
        cube_pca = pca.fit(cube).transform(cube)
        self.pca = cube_pca

    def doKmeans(self):
        pca = self.pca
        kmeans = KMeans(n_clusters=self.nKmeans, n_init=500)
        clusters = kmeans.fit_predict(pca)
        centers = kmeans.cluster_centers_
        self.kmeans = clusters
        self.centers = centers
        kmeansmetrics = self.giveKmeansmetrics()


    def giveBest(self):
        fdata = self.fdata
        ftot = self.ftot
        frgl = self.frgl
        cube = self.cube
        nbest = self.nbest

        idx = ftot.argsort()

        zebest, zemetrics = [], []
        for i in np.arange(nbest):
            id = np.where(idx == i)[0]
            zebest.extend(cube[id, :, :])
            zemetrics.append(np.array([ftot[id][0], fdata[id][0], frgl[id][0]]))

        self.bestimages = np.array(zebest)
        self.bestmetrics = np.array(zemetrics)

        self.plotBest()
        self.writeBest()

    def writeBest(self):
        best = self.bestimages
        metrics = self.bestmetrics
        ftot, fdata, frgl = metrics[:,0], metrics[:,1], metrics[:,2]
        nbest = self.nbest
        for i in np.arange(self.nbest):
            hdr = self.hdr
            hdr['NBEST'] = i+1
            hdr['CDELT1'] = hdr['CDELT1']
            hdr['FTOT'] = ftot[i]
            hdr['FDATA'] = fdata[i]
            hdr['FRGL'] = frgl[i]
            img = best[i,::-1,:]
            hdu = fits.PrimaryHDU(img, header = hdr)
            hdul = fits.HDUList([hdu])
            hdul.writeto(os.path.join(self.dir, f'Images_Best{i+1}.fits'), overwrite=True)


    def plotBest(self):
        best = self.bestimages
        metrics = self.bestmetrics
        ftot, fdata, frgl = metrics[:,0], metrics[:,1], metrics[:,2]
        nbest = self.nbest
        ncols = int(np.sqrt(nbest))
        nrows = int(np.sqrt(nbest))
        if nrows*ncols != nbest:
            ncols += 1
        indices = np.indices((nrows, ncols))
        indrow = indices[0].ravel()
        indcols = indices[1].ravel()
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharey=True, sharex=True)
        d = self.d
        for i in np.arange(nbest):
            ax = axs[indrow[i], indcols[i]]
            image = np.array(best[i,::-1,:])
            ax.imshow(image, extent = (d, -d, -d, d),  cmap='inferno')
            ax.set_xlim(d, -d)
            ax.text(0.8*d, 0.8*d, f"chi2={fdata[i]:2.2f}", color='white', size=9)
            ax.set_title(f"Best model {i+1}")
            if indcols[i] == 0:
                ax.set_xlabel('$\Delta\alpha$ (mas)')
            if indrow[i] == indrow[-1]:
                ax.set_ylabel('$\Delta\delta$ (mas)')
        plt.tight_layout
        plt.savefig(os.path.join(self.dir,f'Image_best{nbest}.pdf'))
        plt.show()
        plt.close()

    def giveKmeansmetrics(self):
        clusters = self.kmeans
        nk = self.nKmeans
        fdata = self.fdata
        frgl = self.frgl

        fdatacluster, frglcluster = [], []
        for i in np.arange(nk):
            fdatak = fdata[clusters==i]
            frglk = frgl[clusters==i]
            fdatacluster.append(np.median(fdatak))
            frglcluster.append(np.median(frglk))
        self.kmeanfdata = fdatacluster
        self.keamnfrgl = frglcluster

    def plotKmeans(self):
        fig, ax = plt.subplots()
        plt.scatter(self.pca[:,0], self.pca[:,1], c=self.kmeans)
        centroids = self.centers
        for centre, k in zip(centroids,np.arange(self.nKmeans)):
            plt.text(centre[0], centre[1], s=f"{k}", size='large', backgroundcolor='lightgray', alpha=0.6)
        plt.ylabel('PCA2')
        plt.xlabel('PCA1')
        plt.tight_layout
        plt.savefig(os.path.join(self.dir,'PCAsets.pdf'))
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        plt.scatter(self.fdata, self.frgl, c=self.kmeans)
        for fd, fr, k in zip(self.kmeanfdata, self.keamnfrgl, np.arange(self.nKmeans)):
            plt.text(fd, fr, s=f"{k}", size='large', backgroundcolor='lightgray', alpha=0.6)
        plt.ylabel('frgl')
        plt.xlabel('fdata')
        plt.tight_layout
        plt.savefig(os.path.join(self.dir,'PCAmetrics.pdf'))
        plt.show()
        plt.close()

    def plotPCA(self):
        fig, ax = plt.subplots()
        plt.scatter(self.pca[:,0], self.pca[:,1])
        plt.show()
        plt.close()

    def groupImages(self):
        idx = list(self.kmeans)
        cube = self.cube
        medians, stds = [], []
        counts, groups = [], []
        for k in np.arange(self.nKmeans):
            counts.append(idx.count(k))
            cub = []
            for image, i in zip(cube, idx):
                if i == k:
                    cub.append(image)
            cub = np.array(cub)
            groups.append(cub)
            medians.append(np.median(cub, axis=0))
            stds.append(np.std(cub, axis=0))
        self.medians = medians
        self.stds = stds
        self.counts = counts
        self.groups = groups

    def plotImages(self):
        self.groupImages()
        images = np.array(self.medians)
        fig, axs = plt.subplots(ncols=self.nKmeans, sharey=True)
        d = self.d
        for ax,i in zip(axs,np.arange(self.nKmeans)):
            ax.imshow(images[i,::-1,:], extent = (d, -d, -d, d),  cmap='inferno')
            ax.set_xlim(d, -d)
            #ax.text(20, 20, f"{i}: {self.counts[i]} images", color='white', size=7)
            ax.set_title(f"{i}: {self.counts[i]} images")
            ax.set_xlabel('$\Delta\alpha$ (mas)')
            ax.set_ylabel('$\Delta\delta$ (mas)')
        plt.tight_layout
        plt.savefig(os.path.join(self.dir,'Image_sets.pdf'))
        plt.show()
        plt.close()

    def writeImages(self, add_stars=False):
        if add_stars:
            self.add_stars()
        images = np.array(self.medians)
        for i in np.arange(len(self.counts)):
            hdr = self.hdr
            hdr['SET'] = i
            hdr['CDELT1'] = hdr['CDELT1']
            img = images[i,:,:]
            hdu = fits.PrimaryHDU(img, header = hdr)
            hdul = fits.HDUList([hdu])
            hdul.writeto(os.path.join(self.dir, f'Images_set{i}.fits'), overwrite=True)

    @staticmethod
    def add_Gaussian(image, x, y, f):
        nx, ny = image.shape
        xg = np.linspace(-nx/2.-x, nx/2.-x, nx)
        yg = np.linspace(-ny/2.-y, ny/2.-y, ny)
        xg2, yg2 = np.meshgrid(xg, yg)
        R = np.sqrt(xg2**2 + yg2**2)
        c = 0.2
        Gauss = np.exp(-(R)**2/(2*c**2))
        #fig, ax = plt.subplots()
        #plt.imshow(Gauss)
        #plt.scatter(nx/2.,nx/2.)
        #plt.show()
        #plt.close()

        Gauss /= np.sum(Gauss)
        image += f*Gauss
        if np.sum(image) != 0:
            image /= np.sum(image)
        return image


    def add_stars(self):
        images = np.array(self.medians)
        xb = self.x2
        yb = self.y2
        fs = self.fs/100
        fb = self.f2/100
        newimages = []
        for i in np.arange(len(self.counts)):
            img = images[i,:,:]
            img /= np.sum(img)
            Gs = self.add_Gaussian(np.zeros_like(img), 0.0, 0.0, fs)
            xbp = xb/self.ps
            ybp = yb/self.ps
            Gb = self.add_Gaussian(np.zeros_like(img), xbp, ybp, fb)

            final = (1-fs-fb) * img + fs * Gs + fb * Gb
            newimages.append(final)
        self.medians = newimages


    def getChi2(self, dir, file):
        data = Data(dir, file)
        V2, V2e, CP, CPe, waveV2, waveCP, u, u1, u2, u3, v, v1, v2, v3 = data.get_data()
        self.data = data

        # setting the ft coordinates
        ftps = 1000*3600*180/np.pi/self.d
        ftd = 1000*3600*180/np.pi/self.ps
        ftx0 = np.linspace(-ftd, ftd, self.n)  #- +
        fty0 = np.linspace(-ftd, ftd, self.n)  #- +
        ftx, fty = np.meshgrid(ftx0, fty0)
        self.wave0 = 2.2e-6

        imgs = np.array(self.medians)
        for i in np.arange(self.nKmeans):
            image = imgs[i,:,:]
            image /= np.sum(image)
            ftimg = np.fft.ifftshift(image, axes=(0,1) )
            ftimg = np.fft.fft2(ftimg)
            ftimg = np.fft.fftshift(ftimg, axes=(0,1) )


            #fig, ax = plt.subplots()
            #plt.imshow(np.abs(ftimg), extent=( -ftd, ftd, -ftd, ftd) )
            #plt.scatter(u, v, marker='+')
            #plt.scatter(-u, -v, marker='+')
            #plt.show()
            #plt.close()

            FourierRe = sc.interpolate.interp2d(ftx0, fty0, ftimg.real, bounds_error=True, kind='cubic')
            FourierIm = sc.interpolate.interp2d(ftx0, fty0, ftimg.imag, bounds_error=True, kind='cubic')

            #FourierRe = sc.interpolate.RectBivariateSpline(ftx0, fty0, ftimg.real)
            #FourierRe = sc.interpolate.RectBivariateSpline(ftx0, fty0, ftimg.real)

            #fig,ax = plt.subplots()
            #plt.imshow(np.abs(ftimg))
            #plt.show()
            #plt.close()
#            print( f'u shape is {u.shape} and v shape is {v.shape}' )
            Vis, Phi = self.giveVis(u, v, waveV2, FourierRe, FourierIm)

            chi2v2 = ((np.power(Vis,2) - V2)/ V2e )**2
            chi2v2 = np.sum(chi2v2) / len(V2)

            #fig, ax = plt.subplots()
            #plt.scatter(np.sqrt(u**2, v**2), np.power(Vis,2) )
            #plt.scatter(np.sqrt(u**2, v**2), V2)
            #plt.show()
            #plt.close()


    def giveVis(self, u, v, wave, FTre, FTim):
        # interpolate in the Fourier plane
        #Vimg = FTre(u, v) + 1j*FTim(u, v)
        #Vimg = [ FTre(ui, vi) + 1j*FTim(ui, vi) for ui, vi in zip(u, v)]
        Vimg = np.zeros_like(u).astype(complex)
        for i, (ui, vi) in enumerate(zip(u, v)):

            Vimg[i] = FTre(ui, vi) + 1j*FTim(ui, vi)
        #Vimg = np.array(Vimg)

        # add sparco on top
        f1 = self.fs/100 * np.power(wave/self.wave0, -4)
        f2 = self.f2/100 * np.power(wave/self.wave0, self.dsec)
        fimg = (1 - self.fs/100 - self.f2/100) * np.power(wave/self.wave0, self.denv)


        V1 = 0j + 2 * sc.special.jv(1, np.pi * self.UD/1000/3600/180*np.pi * np.sqrt(u**2, v**2)) / (np.pi * self.UD/1000/3600/180*np.pi * np.sqrt(u**2, v**2) )

        alpha = self.x2 /1000 /3600 *np.pi /180
        delta = self.y2 /1000 /3600 *np.pi /180
        V2 = np.exp(-2*np.pi*1j * (u*alpha + v*delta) )


        #Vtot = np.divide(np.multiply(fimg , Vimg) + np.multiply(f1, V1) + np.multiply(f2, V2 ), fimg + f1 + f2)

        Vtot = (fimg * Vimg + f1 * V1 + f2 * V2 )/ (fimg + f1 + f2)

#        print(Vtot.shape)

        return np.abs(Vtot), np.arctan2(Vtot.imag, Vtot.real)


    def plotSet(self, set):
        images = self.groups[set]
        number = self.counts[set]

        fig, axs = plt.subplots(ncols=5,nrows=3)
        i = -1
        for axy in axs:
            for ax in axy:
                i += 1
                if i < number-1:
                    ax.imshow(images[i,::-1,:], cmap='inferno')
        plt.tight_layout
        plt.show()
        plt.close()

    def radial_profiles(self, R=20, set=-1):
        if set == -1:
            cube = self.cube
            image = np.median(cube, axis=0)
            std = np.std(cube, axis=0)
        else:
            image = np.array(self.medians)[set,:,:]
            std = np.array(self.stds)[set,:,:]


        self.compute_profile(R, image, std)
        self.plot_profile()
        #self.plot_std_profile(R, image, std)

    def set_coord(self):
        n = self.n
        xr = np.arange(n)+1
        x =  - (xr-self.n/2)*self.ps
        y =  - (xr-self.n/2)*self.ps
        d = n*self.ps/2.
        x = np.linspace(d, -d, n)
        y = np.linspace(-d, d, n)
        x2, y2 = np.meshgrid(x, y)
        R = np.sqrt(x2**2+y2**2)
        angle = np.arctan2(y2, x2)

        # fig,ax = plt.subplots()
        # plt.imshow(y2)
        # plt.show()
        # plt.close()
        self.d = d
        self.x = x2
        self.y = -y2
        self.R = R
        self.angle = angle

    def compute_profile(self, R, image, noise):
        #noise /= np.max(image)
        #image /= np.max(image)
        dr = (self.d - self.ps) / R
        radii_m = np.linspace(self.ps, self.d-dr, R)
        radii_p = np.linspace(self.ps+dr, self.d, R)
        self.radii = (radii_m + radii_p)/2.
        profile, noises, rms = [], [], []
        for rad_m, rad_p in zip(radii_m, radii_p):
            mask = (self.R >= rad_m) * (self.R < rad_p)
            reduced_image = image[mask!=0]
            reduced_noise = noise[mask!=0]
            fluxi = np.average(reduced_image)
            noisei = np.average(reduced_noise)
            rmsi = np.std(reduced_image ) #/len(reduced_image)
            profile.append(fluxi)
            noises.append(noisei)
            rms.append(rmsi)

        self.profile = np.array(profile)
        self.noises = np.array(noises)
        self.rms = np.array(rms)

    def plot_profile(self):

        fig, ax = plt.subplots()
        ax.plot(self.radii, self.profile, label = 'profile', color='black')
        ax.plot(self.radii, self.noises, label = 'noise', color='blue')
        ax.plot(self.radii, self.rms, label = 'azim. std.', color='red')
        plt.yscale('log')
        plt.legend()
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(self.radii, self.profile/self.noises, label = 'SNR-noise', color='blue')
        ax.plot(self.radii, self.profile/self.rms, label = 'SNR-std.', color='red')
        #plt.ylim(0,5)
        plt.legend()
        plt.show()
        plt.close()

def main():
    dirdata = '/Users/jacques/Work/Organic/diagnostics/ImageRec_mu=0.1/'
    file = 'c_imaging_contest2.fits'

    arg = sys.argv[1:]
    Cube0 = cube(arg[0], nPCA = 10, nKmeans=3)
    Cube0.giveBest()
    #Cube0.getChi2(dirdata, file)
    Cube0.writeImages(add_stars=False)
    #Cube0.radial_profiles(R=100)
    #Cube0.radial_profiles(R=100,set=0)
    #Cube0.radial_profiles(R=100,set=1)
    #Cube0.radial_profiles(R=100,set=2)
    #Cube0.plotSet(0)

if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv)
