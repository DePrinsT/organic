"""
A module to reconstruct images from optical interferometry data using Generative Adversarial Networks. This is the
core module of ORGANIC.
"""

# Organic FunctionLibrary
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.optimizers as optimizers  # not needed?
from tensorflow.keras.utils import plot_model
import os
from astropy.io import fits
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import itertools
import organic.auxiliary.ReadOIFITS as oi
import tensorflow.keras.backend as kbackend
import scipy.special as sp
import matplotlib.colors as colors
import sys
import glob


# some colors for printing warnings, etc
class Bcolors:
    """
    A class to represent ANSI escape sequences for terminal text formatting.

    Attributes
    ----------
    HEADER : str
        ANSI escape sequence for magenta colored text.
    OKBLUE : str
        ANSI escape sequence for blue colored text.
    OKGREEN : str
        ANSI escape sequence for green colored text.
    WARNING : str
        ANSI escape sequence for yellow colored text.
    FAIL : str
        ANSI escape sequence for red colored text.
    ENDC : str
        ANSI escape sequence to reset text formatting.
    BOLD : str
        ANSI escape sequence for bold text.
    UNDERLINE : str
        ANSI escape sequence for underlined text.
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# Some fancy message writing functions
def header(msg):
    """
    Prints a message in a header format with specific colors.

    Parameters
    ----------
    msg : str
        The message to be printed.
    """
    print(Bcolors.HEADER + msg + Bcolors.ENDC)


def bold(msg):
    """
    Print a message in bold text.

    Parameters
    ----------
    msg : str
        The message to be printed in bold.
    """
    print(Bcolors.BOLD + msg + Bcolors.ENDC)


def underline(msg):
    """
    Print a message with an underline.

    Parameters
    ----------
    msg : str
        The message to be printed with an underline.
    """
    print(Bcolors.UNDERLINE + msg + Bcolors.ENDC)


def inform(msg):
    """
    Prints a message in blue color.

    Parameters
    ----------
    msg : str
        The message to be printed.
    """
    print(Bcolors.OKBLUE + msg + Bcolors.ENDC)


def inform2(msg):
    """
    Prints a message in green color.

    Parameters
    ----------
    msg : str
        The message to be printed.
    """
    print(Bcolors.OKGREEN + msg + Bcolors.ENDC)


def warn(msg):
    """
    Prints a warning message in a specific format.

    Parameters
    ----------
    msg : str
        The warning message to be printed.
    """
    print(Bcolors.WARNING + msg + Bcolors.ENDC)


def fail(msg):
    """
    Print a failure message in a specific format.

    Parameters
    ----------
    msg : str
        The failure message to be printed.
    """
    print(Bcolors.FAIL + msg + Bcolors.ENDC)


def log(msg, dir):
    """
    Append a message to a log file.

    Parameters
    ----------
    msg : str
        The message to be logged.
    dir : str
        The directory where the log file is located. The log file will be named "log.txt".

    Returns
    -------
    None
    """
    f = open(dir + "log.txt", "a")
    f.write(msg + "\n")
    f.close()


# return the binary cross-entropy function
# NOTE: if y_true = 1 and y_pred is the discriminator output D -> ouputs -log(D)
# this is how this function is used as a loss function during the image reconstruction
# phase to calculate the regularization loss term
def cross_entropy(y_true, y_pred):
    """
    Computes the binary cross-entropy loss between true labels and predicted labels.

    Parameters
    ----------
    y_true : tensor
        True binary labels.
    y_pred : tensor
        Predicted probabilities.

    Returns
    -------
    tensor
        Binary cross-entropy loss.

    Notes
    -----
    This function uses the Keras backend to compute the binary cross-entropy loss.
    The `from_logits` parameter is set to False, indicating that `y_pred` is expected
    to be probabilities (i.e., values between 0 and 1).
    """
    return kbackend.binary_crossentropy(y_true, y_pred, from_logits=False)


## definition of the GAN class
# gen = path to the generator model file (h5 file)
# dis = path to the discriminator model file (h5 file)
# npix = number of pixels (used a lot -> will require extensive rewrites before extensions are made),
# not used if the generator and discriminator are read in from a file, the it is just read in from the generator output
# train_disc = whether or not to train the discriminator (is passed to the create_gan method), i.e. no training
# of the discriminator when doing image reconstruction
# noiselength = length of the noise vector at the start of the generator
# adam_lr = learning rate parameter for the ADAM optimization algorithm
# adam_beta1 = The exponential decay rate for the 1st moment estimates in ADAM (no parameter for 2nd moment estimates
# yet)
# reset_opt = whether or not to reset the optimizer when performing restart iterations while doing image reconstruction
# set to true by default (NOTE: False gives bullshit results!!)
# amsgrad = whether or not to use the amsgrad version of the Adam optimizer (this is generally more efficient and
# provides better convergence).
# tf_verbose = whether or not to let tensorflow be verbose in its debug warnings
# gen_init = copy of the initial generator state (only created if reading in a GAN from file)
# also contains self.params -> to store image recosntruction parameters, including both settings like pixelscale and
# the SPARCO parameters.
class GAN:
    """
    The GAN class to train and use it
    The GAN is made from a generator and a discriminator

    Parameters
    ----------
    gen : str, optional
        Path to the generator model file (h5 file).
    dis : str, optional
        Path to the discriminator model file (h5 file).
    npix : int, optional
        Number of pixels.
    train_disc : bool, optional
        Whether or not to train the discriminator.
    noiselength : int, optional
        Length of the noise vector at the start of the generator.
    adam_lr : float, optional
        Learning rate parameter for the ADAM optimization algorithm.
    adam_beta1 : float, optional
        The exponential decay rate for the 1st moment estimates in ADAM.
    reset_opt : bool, optional
        Whether or not to reset the optimizer when performing restart iterations.
    amsgrad : bool, optional
        Whether or not to use the amsgrad version of the Adam optimizer.
    tf_verbose : bool, optional
        Whether or not to let tensorflow be verbose in its debug warnings.
    """

    def __init__(
        self,
        gen="",
        dis="",
        npix=128,
        train_disc=False,
        noiselength=100,
        adam_lr=0.0001,
        adam_beta1=0.91,
        reset_opt=True,
        amsgrad=True,
        tf_verbose=False,
    ):
        # suppress tensorflow warnings if needed
        if not tf_verbose:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        self.reset_opt = reset_opt
        self.adam_lr = adam_lr
        self.adam_beta1 = adam_beta1
        self.amsgrad = amsgrad
        # function call to get Adam optimizer
        self.opt = self.get_optimizer(self.adam_lr, self.adam_beta1, amsgrad=self.amsgrad)
        self.train_disc = train_disc
        self.noiselength = noiselength
        # if the paths to the generator and discriminator are not empty -> read them in
        if gen != "" and dis != "":
            self.dispath = dis
            self.genpath = gen
            self.read_models()
        # otherwise fall back on default generator/discriminator creation
        else:
            self.npix = npix
            self.gen = self.create_generator()
            self.dis = self.create_discriminator()
        # after the generator and discriminator have been read in -> create the full GAN
        self.gan = self.create_gan(train_disc=self.train_disc)

    # static method means no self argument has to be passed for the function to work ->
    # i.e. it is only part of the class because of organizational reasons. Calling on this method works from
    # both calling on an instance of the class as well as on the class itself
    # beta2 and epsilon have default values here
    @staticmethod
    def get_optimizer(lr, beta1, beta2=0.999, epsilon=1e-7, amsgrad=False):
        """
        Get the Adam optimizer.

        Parameters
        ----------
        lr : float
            Learning rate.
        beta1 : float
            The exponential decay rate for the 1st moment estimates.
        beta2 : float, optional
            The exponential decay rate for the 2nd moment estimates.
        epsilon : float, optional
            A small constant for numerical stability.
        amsgrad : bool, optional
            Whether to use the AMSGrad variant of Adam.

        Returns
        -------
        Adam
            The Adam optimizer.
        """
        return Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2,
            epsilon=epsilon,
            amsgrad=amsgrad,
        )

    def read_models(self):
        """
        Loading the dictionary from the generator and discriminator paths.
        """
        inform(f"Loading the generator from {self.genpath}")
        gen = load_model(self.genpath)
        gen.summary()  # call Keras Model.summary function to print out a summary
        inform(f"Loading the discriminator from {self.dispath}")
        dis = load_model(self.dispath)
        dis.summary()  # call summary
        self.gen = gen  # set the generator model
        self.dis = dis  # set the discriminator model
        self.npix = gen.output.shape[1]  # read the number of pixels
        # create copy of generator architecture at read-in to reset during image reconstruction
        gen_copy = tf.keras.models.clone_model(gen)  # doesn't set the weights yet
        gen_copy.set_weights(gen.get_weights())  # call to get and set the weights
        self.gen_init = gen_copy  # sets the initial generator to be preserved

    def create_generator(self, relu=0.25):
        """
        Create a compiled GAN generator if paths to a generator and discriminator were not specified.

        Parameters
        ----------
        relu : float, optional
            Leaky ReLU activation parameter.

        Returns
        -------
        generator : keras.Sequential
            Compiled generator model.
        """
        inform("Creating the generator")
        npix = self.npix  # use GAN-specified number of pixels
        # create the gemerator architecture
        # create a Sequential (special case of a Model), where the model is a stack of single-input,
        # single-ouput layers. I.e., it's just one layer after the other. I.e. something like an inception module
        # cannot be implemented this way.
        # NOTE: THE SIZE IN THE ORIGINAL ORGANIC PAPER SCHEMATIC IS WRONG
        generator = keras.Sequential(
            [
                # first dense layer with noise vector input
                # several more arguments that aren't used here, inlcuding a lower-rank approximation
                layers.Dense(
                    int((npix / 8) * (npix / 8) * 256),  # units = dimensionality of output (number of nodes)
                    use_bias=False,  # whether or not to add a vector of bias parameters
                    # initializer for the kernel weights matrix (he_normal = truncated normal centered on 0, scaled
                    # by number of input weights, show to work well for ReLU activation).
                    kernel_initializer="he_normal",
                    input_shape=(self.noiselength,),  # input shape
                ),
                layers.LeakyReLU(alpha=relu),  # apply leaky ReLU activation
                layers.Reshape((int(npix / 8), int(npix / 8), 256)),  # reshape inputs into cube for convolutions
                # also has additional options, e.g. for input data format (i.e. channels first or channels last,
                # the default, in shape)
                # output shape = 16 x 16 x 256 (last number = number of filters)
                layers.Conv2DTranspose(  # apply first transposed convolution
                    npix,  # number of filters to be considered
                    (4, 4),  # shape of the filter
                    strides=(2, 2),  # stride of the filters movement
                    # padding option, either 'same' which provides 0-padding (so with 1,1 stride the ouput shape is the
                    # same as the input), or 'valid' = no padding
                    padding="same",
                    use_bias=False,  # whether to add bias
                    kernel_initializer="he_normal",  # initialization of kernel weigths
                ),
                # output shape = 32 x 32 x 128
                # Batch normalization layer; has a bunch of options including momentum options for during inference.
                # NOTE: it behaves very differently during training and inference
                # during inference it namely uses a moving mean/variance calculated over all the batches it has
                # seen during training
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=relu),
                layers.Conv2DTranspose(
                    64,  # NOTE: this is no longer dependent on npix -> this will only work properly for npix = 128
                    (4, 4),
                    strides=(2, 2),
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_normal",
                ),
                # output shape = 64 x 64 x 64
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=relu),
                layers.Conv2DTranspose(
                    32,
                    (4, 4),
                    strides=(2, 2),
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_normal",
                ),
                # output shape = 128 x 128 x 32
                layers.BatchNormalization(),
                layers.LeakyReLU(alpha=relu),
                layers.Conv2D(
                    1,
                    (2, 2),
                    strides=(1, 1),
                    padding="same",
                    use_bias=False,
                    activation="tanh",
                    # last convolution instead uses glorot normal, gaussian initializations caled by number of inputs
                    # and outputs. Shown to work well for tanh activation functions.
                    kernel_initializer="glorot_normal",
                ),
                # output shape = 128 x 128 x 1 -> final image to be fed to the discriminator
            ],
            name="generator",
        )
        generator.summary()  # print summary of the generator architecture

        # compile model for training with binary cross-entropy loss function
        generator.compile(
            loss="binary_crossentropy",
            optimizer=self.get_optimizer(self.adam_lr, self.adam_beta1),  # call self-defined get_optimizer to set Adam
        )
        return generator

    def create_discriminator(self, relu=0.25, dropout=0.5):
        """
        Create a compiled discriminator generator if paths to a generator and discriminator were not specified.

        Parameters
        ----------
        relu : float, optional
            Leaky ReLU activation parameter.
        dropout : float, optional
            Dropout fraction.

        Returns
        -------
        discriminator : keras.Sequential
            Compiled discriminator model.
        """
        inform("Creating the discriminator")
        npix = self.npix
        discriminator = keras.Sequential(
            [
                layers.Conv2D(
                    npix / 4,  # to match the 32 filters in the preceding generator layer
                    (3, 3),
                    strides=(2, 2),
                    padding="same",
                    input_shape=[npix, npix, 1],
                    kernel_initializer="he_normal",
                ),
                # output size 64 x 64 x 32
                layers.LeakyReLU(relu),
                layers.SpatialDropout2D(dropout),
                layers.Conv2D(
                    npix / 2,
                    (3, 3),
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer="he_normal",
                ),
                # output size 32 x 32 x 64
                layers.LeakyReLU(relu),
                layers.SpatialDropout2D(dropout),
                layers.Conv2D(
                    npix,
                    (3, 3),
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer="he_normal",
                ),
                # output size 16 x 16 x 128
                layers.LeakyReLU(relu),
                layers.SpatialDropout2D(dropout),
                layers.Flatten(),
                # gets fed using a dense connection into a single node with a sigmoid activation
                layers.Dense(
                    1,
                    activation="sigmoid",
                    use_bias=False,
                    kernel_initializer="glorot_normal",
                ),
            ],
            name="discriminator",
        )
        discriminator.summary()  # print discriminator summary
        # compile into a working model for the discriminator
        discriminator.compile(
            loss="binary_crossentropy",
            optimizer=self.get_optimizer(self.adam_lr, self.adam_beta1),
            metrics=["accuracy"],
        )

        return discriminator

    # Once the generator and discriminator have been set and compiled -> set the GAN. This will be the object that's
    # used for actually training the generator (while the )
    def create_gan(self, train_disc=False, train_gen=True, reinit=True):
        """
        Create the GAN model.

        Parameters
        ----------
        train_disc : bool, optional
            Whether or not to train the discriminator.
        train_gen : bool, optional
            Whether or not to train the generator.
        reinit : bool, optional
            Whether or not to re-initialize generator state.

        Returns
        -------
        gan : keras.Model
            Compiled GAN model.
        """
        if reinit:
            # code below makes a deep copy of the model!!!
            gen_init_copy = tf.keras.models.clone_model(self.gen_init)  # doesn't set the weights yet
            gen_init_copy.set_weights(self.gen_init.get_weights())  # call to get and set the weights
            gen = gen_init_copy
        else:
            gen = self.gen  # otherwise, just use the generator from the previous iteration itself
        self.dis.trainable = train_disc  # set trainability of the discriminator (false when reconstructing of course)
        gen.trainable = train_gen  # set whether
        gan_input = layers.Input(shape=(self.noiselength,))  # instanciate the input noise vector layer
        # NOTE: creating a GAN model like below keeps track of the underlying layers/sub-models and the associated
        # weight. I.e. setting gen.trainable also affects the trainability of those weights in the gan object.
        gen_output = gen(gan_input)  # pass the input to the generator and calculate its output
        gan_output = self.dis(
            gen_output
        )  # pass generator output as input to the discriminator to calculate the GAN output
        # create the final model by specifying inputs and outputs
        gan = Model(inputs=gan_input, outputs=[gan_output, gen_output])
        # NOTE: models do need to be recompiled if the trainability of inner layers has been changed!
        gan.compile(
            loss="binary_crossentropy", optimizer=self.opt, metrics=["accuracy"]
        )  # compile the GAN using binary crossentropy as a loss again
        self.gen = gen
        return gan

    # function to train the GAN network
    def train(
        self,
        input_images,  # InputImages object containing the training images
        save_dir="./saved_models/",  # where to save trained networks
        nepochs=2000,  # number of epochs to run over the training set
        nbatch=50,  # mini-batch size
        overtrain_discr=1,
        plot_epochs=25,  # epoch interval after which to plot examples of generated images
        use_1sided_label_smooth=False,  # set the value of 'true' images to 0.9 instead of 1 to prevent overconfidence
        save_epochs=[],  # additional list of epochs at which to save the model
    ):
        """
        Train the GAN network.

        Parameters
        ----------
        input_images : InputImages
            InputImages object containing the training images.
        save_dir : str, optional
            Directory where the trained networks will be stored.
        nepochs : int, optional
            Number of epochs to run over the training set.
        nbatch : int, optional
            Mini-batch size.
        overtrain_discr : int, optional
            Number of times to train the discriminator more than the generator.
        plot_epochs : int, optional
            Epoch interval after which to plot examples of generated images.
        use_1sided_label_smooth : bool, optional
            Whether or not to use one-sided label smoothing.
        save_epochs : list, optional
            Additional list of epochs at which to save the model.
        """
        self.save_dir = save_dir  # set save directory

        self.nbatch = nbatch  # set batch size
        self.nepochs = nepochs  # set number of epochs
        # this is the GAN
        generator = self.gen
        discriminator = self.dis
        gan = self.gan
        # these are the images
        x_train = input_images.images  # retrieve 4D training input images numpy tensor
        datagen = input_images.data_gen  # retrieve the image data generator
        batch_count = int(np.ceil(x_train.shape[0] / nbatch))  # defining the total number batches

        # define the labels
        y_real = 1
        batches = datagen.flow(x_train, y=None, batch_size=nbatch)  # creates an iterator to give training batches
        if use_1sided_label_smooth:
            y_real = 0.9  # re-assign to 0.9 in order to avoid over-confidence
        y_false = np.zeros(nbatch)
        y_true = np.ones(nbatch) * y_real

        # create lists for holding the loss and metric values over the different epochs
        dis_fake_loss, dis_real_loss, dis_fake_accuracy, gen_accuracy, gen_loss = (
            [],
            [],
            [],
            [],
            [],
        )

        inform("Starting GAN training")
        # iterate over the training epochs
        for epoch in np.arange(nepochs):
            inform2(f"Epoch {epoch+1} of {nepochs}")
            # set loss terms and accuracy terms for this epoch
            # set the loss terms and accuracy for the discriminator
            dis_fake_loss_ep, dis_real_loss_ep, dis_fake_accuracy_ep = 0, 0, 0
            # set the loss term and accuracy for the generator.
            gen_loss_ep, gen_accuracy_ep = 0, 0
            for _ in range(nbatch):  # batch_size in version from jacques
                # generate  random noise as an input  to  initialize the  generator
                noise = np.random.normal(0, 1, [nbatch, self.noiselength])  # note the output shape: nbatch x input
                # Generate ORGANIC images from noised input
                generated_images = generator.predict(noise)  # generate 'fake' generator images from noise input
                # train the discriminator (more than the generator if requested)
                for i in range(overtrain_discr):
                    # Get a random set of  real images
                    training_image_batch = batches.next()
                    # if the batch created by the generator is too small, resample (TODO: what does this even do? Is it
                    # not just redundant?)
                    # TODO: Why is the amount of 'real' training and 'fake' generator images the same in one batch?
                    if training_image_batch.shape[0] != nbatch:
                        batches = datagen.flow(x_train, y=None, batch_size=nbatch)
                        training_image_batch = batches.next()
                    # reshape the input image batch to match the shape of the generator's 'fake' images
                    training_image_batch = training_image_batch.reshape(nbatch, self.npix, self.npix, 1)
                    # Construct different batches of real and fake data
                    x_batch = np.concatenate([training_image_batch, generated_images])
                    # Labels for generated and real data
                    y_pred = np.concatenate([y_true, y_false])
                    # Pre train discriminator on fake and real data before starting the gan.
                    # NOTE: need to reset to trainable after the generator training step below has set to non-trainable
                    discriminator.trainable = (
                        True  # TODO: don't models need to recompile if their trainability is changed?
                    )
                    discriminator.train_on_batch(x_batch, y_pred)  # Run a single gradient descent step for the batch
                    # NOTE: the evaluation of the metrics done after a gradient descent step instead of before?
                    # While it doesn't make that much difference probably it doesn't make sense.
                # evaluate the discriminator loss and metrics in test mode for the 'real' test images
                dis_real_eval = discriminator.evaluate(training_image_batch, y_pred[:nbatch], verbose=0)
                # evaluate the discriminator loss and metrics in test mode for the 'fake' generator images
                dis_fake_eval = discriminator.evaluate(generated_images, y_pred[nbatch:], verbose=0)
                # evaluations for the cost evolution of the discriminator
                # TODO: why the normalisation over batch count, I'd suspect this to be done automatically?
                dis_fake_loss_ep += dis_fake_eval[0] / batch_count
                dis_real_loss_ep += dis_real_eval[0] / batch_count
                # How many 'fake' generator images were correctly flagged as such (i.e. sensitivity to fakes)
                dis_fake_accuracy_ep += dis_fake_eval[1] / batch_count

                # Treating the noised input of the generator as real data (i.e. the generator will now adapt in order
                # to attempt to force the discriminator to recognize it's output as real images).
                # With this formulation the minimization for the gradient descent in this step
                # uses eq. 3.25 of Rik Claes' masters thesis.
                noise = np.random.normal(0, 1, [nbatch, self.noiselength])
                y_gen = np.ones(nbatch)

                # During the training of gan,
                # the weights of discriminator should be fixed.
                # We can enforce that by setting the trainable flag
                discriminator.trainable = False  # set discriminator to untrainable for the generator training step

                # training the GAN by alternating the training of the Discriminator (i.e. this switches to training
                # the generator)
                gan.train_on_batch(noise, y_gen)
                # evaluation of generator
                gen_eval = gan.evaluate(noise, y_gen, verbose=0)
                gen_loss_ep += gen_eval[0] / batch_count
                gen_accuracy_ep += gen_eval[1] / batch_count

            # Saving all the metrics per epoch
            gen_accuracy.append(gen_accuracy_ep)
            gen_loss.append(gen_loss_ep)
            dis_fake_loss.append(dis_fake_loss_ep)
            dis_real_loss.append(dis_real_loss_ep)
            dis_fake_accuracy.append(dis_fake_accuracy_ep)
            # save current state of networks
            self.gan = gan
            self.dis = discriminator
            self.gen = generator

            # plot examples of generated images
            if epoch == 1 or epoch % plot_epochs == 0:
                self.plot_generated_images(epoch)
            if epoch in save_epochs:
                self.save_model(str(epoch) + "thEpoch.h5")

        self.save_model("finalModel.h5")
        self.plot_gan_evolution(dis_fake_loss, dis_real_loss, gen_loss, dis_fake_accuracy, gen_accuracy)

        inform(f"Training succesfully finished.\nResults saved at {self.save_dir}")

    def get_image(self, noise):
        """
        Get a random image from the generator.

        Parameters
        ----------
        noise : array-like
            Noise vector.

        Returns
        -------
        img : numpy.ndarray
            Generated image.
        """
        # random input
        noise_input = np.array(noise)
        # NOTE: this works because the generator image output is one of the outputs defined when creating the gan
        # object
        img = self.gan.predict(noise_input)[1]  # select second GAN model output, i.e. the generator image
        # select batch position and channel (both 0, since only 1 channel
        # and only one noise vector passed through).
        img = (np.array(img)[0, :, :, 0],)  # notice we also make it a numpy array
        img = img[0]  # extract from tuple with one value

        return img

    def plot_image(self, img, name="image.png", chi2_label=""):
        """
        Plot an image.

        Parameters
        ----------
        img : numpy.ndarray
            Image to plot.
        name : str, optional
            Name of the saved image file.
        chi2_label : str, optional
            Chi2 label string to be added to the image.
        """
        binary = False
        star_present = False
        d = self.params["ps"] * self.npix / 2.0
        if self.params["fsec"] > 0:
            binary = True
            xb = self.params["xsec"]
            yb = self.params["ysec"]
        if self.params["fstar"] > 0:
            star_present = True

        fig, ax = plt.subplots()
        plt.imshow(img[::-1, :], extent=(d, -d, -d, d), cmap="inferno")  # note that the x axis is flipped here
        if star_present:
            plt.plot(0, 0, color="gold", marker="*")
        if binary:
            plt.plot(xb, yb, color="white", marker="x")
        plt.text(0.9 * d, 0.9 * d, chi2_label, c="white")
        plt.xlabel(r"$\Delta \alpha$ (mas)")
        plt.ylabel(r"$\Delta \delta$ (mas)")
        plt.tight_layout()
        plt.savefig(name, dpi=250)
        plt.close()

        return img

    def save_image_from_noise(self, noise, name="image.png"):
        """
        Create and save an image from an input noise vector.

        Parameters
        ----------
        noise : array-like
            Noise vector.
        name : str, optional
            Name of the saved image file.
        """
        img = self.get_image(noise)  # get an image from the noise vector (why do this if it was already done)
        # NOTE: ::-1 reverses the direction of the axis
        self.plot_image(img[:, ::-1], name=name)

    # NOTE: the size of the input is hardcoded here
    def get_random_image(self):
        """
        Create a generator image from a completely random input noise vector.

        Returns
        -------
        img : numpy.ndarray
            Generated image.
        """
        # random input
        input = np.array([np.random.normal(0, 1, 100)])
        img = self.gan.predict(input)[1]
        img = np.array(img)[0, :, :, 0]

        return img

    def save_random_image(self, name="random_image.png"):
        """
        Create and save a generator image from a completely random input noise vector.

        Parameters
        ----------
        name : str, optional
            Name of the saved image file.
        """
        img = self.get_random_image()
        fig, ax = plt.subplots()
        plt.imshow(img)
        plt.savefig(name, dpi=250)
        plt.close()

    def plot_gan_evolution(self, dis_fake_loss, dis_real_loss, gen_loss, dis_fake_accuracy, gen_accuracy):
        """
        Plot the GAN's evolution during training.

        Parameters
        ----------
        dis_fake_loss : list
            Discriminator fake data loss values.
        dis_real_loss : list
            Discriminator real data loss values.
        gen_loss : list
            Generator loss values.
        dis_fake_accuracy : list
            Discriminator fake data accuracy values.
        gen_accuracy : list
            Generator accuracy values.
        """
        save_dir = self.save_dir

        fig, ax = plt.subplots()

        color = iter(plt.cm.rainbow(np.linspace(0, 1, 5)))

        c = next(color)
        plt.plot(dis_fake_loss, label="discriminator fake data loss", c=c)
        c = next(color)
        plt.plot(dis_real_loss, label="discriminator real data loss", c=c)
        c = next(color)
        plt.plot(gen_loss, label="generator loss", c=c)
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(save_dir + "loss_evolution.png", dpi=250)
        plt.close()

        fig, ax = plt.subplots()
        plt.plot(dis_fake_accuracy, label="discriminator data accuracy", c=c)
        c = next(color)
        plt.plot(gen_accuracy, label="generator data accuracy", c=c)
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.savefig(save_dir + "accuracy_evolution.png", dpi=250)
        plt.close()

    def save_model(self, model_name):
        """
        Save a trained GAN model to HDF5 files.

        Parameters
        ----------
        model_name : str
            Name to be used for storing the networks of this run.
        """
        # test if the path exists, if not, creates it
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        model_path_gan = os.path.join(self.save_dir, "GANfull" + model_name)
        self.gan.save(model_path_gan)
        plot_model(self.gan, to_file="full.png", show_shapes=True)

        model_path_generator = os.path.join(self.save_dir, "generator" + model_name)
        self.gen.save(model_path_generator)
        plot_model(self.gen, to_file="generator.png", show_shapes=True)

        model_path_discriminator = os.path.join(self.save_dir, "discriminator" + model_name)
        self.dis.save(model_path_discriminator)
        plot_model(self.dis, to_file="discriminator.png", show_shapes=True)
        print(f"Saved trained model at {model_path_gan}")

    def plot_generated_images(self, epoch, examples=36, dim=(6, 6), figsize=(15, 9)):
        """
        Plot examples of generated images.

        Parameters
        ----------
        epoch : int
            The epoch at which the plots are made, used for naming the image.
        examples : int, optional
            The number of examples to be displayed in the plot.
        dim : tuple, optional
            Dimensions of the plot grid.
        figsize : tuple, optional
            Size of the figure.
        """
        generator = self.gen
        noise = np.random.normal(loc=0, scale=1, size=[examples, self.noiselength])
        generated_images = generator.predict(noise)
        # TODO: for some reason a reshape? Which is not done in the get_image function -> have to be really careful
        # about sign conventions here. Might be because the imshow origin is chosen to be 'lower' here
        generated_images = generated_images.reshape(examples, self.npix, self.npix)
        fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize, sharex=True, sharey=True)
        i = -1
        for axv in axs:
            for ax in axv:
                i += 1
                ax.imshow(
                    generated_images[i],
                    origin="lower",  # changes the origin from the standard convention
                    interpolation=None,
                    cmap="inferno",
                    vmin=-1,
                    vmax=1,
                )
                # ax.invert_xaxis()
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"cgan_generated_image_ep{epoch}.png", dpi=250)
        plt.close()

    def image_reconstruction(
        self,
        data_files,
        sparco,
        data_dir="./",
        mu=1,
        epochs=50,
        nrestart=50,
        reinit_gen=False,  # whether to re-initialize generator state each iteration
        boot=False,
        nboot=100,
        ps=0.6,
        use_low_cp_approx=False,
        grid=False,
        diagnostics=False,
        name="",
    ):
        """
        Master image reconstruction function.

        Parameters
        ----------
        data_files : list
            List of data files.
        sparco : SPARCO
            SPARCO object containing geometrical model information.
        data_dir : str, optional
            Directory containing the data files.
        mu : float, optional
            Regularization weight.
        epochs : int, optional
            Number of epochs during a single reconstruction.
        nrestart : int, optional
            Number of generator restarts to perform.
        reinit_gen : bool, optional
            Whether to re-initialize generator state each iteration.
        boot : bool, optional
            Whether to bootstrap.
        nboot : int, optional
            Number of bootstrap samples.
        ps : float, optional
            Pixel scale.
        use_low_cp_approx : bool, optional
            Whether to use low closure phase approximation when calculating chi2.
        grid : bool, optional
            Whether to do a grid.
        diagnostics : bool, optional
            Whether to print out diagnostic plots for every epoch.
        name : str, optional
            Directory name under which to store everything.
        """
        self.mu = mu  # regularization weight
        self.epochs = epochs  # number of epochs (single gradient descent steps) during a single reconsturction
        self.ps = ps  # pixelscale
        self.nboot = nboot  # number of bootstrap samples
        self.boot = boot  # whether to bootstrap in the first place
        self.nrestart = nrestart  # number of generator restarts to perform
        self.use_low_cp_approx = (
            use_low_cp_approx  # whether to use low closure phase approximation when calculating chi2
        )
        self.sparco = sparco  # SPARCO object containing geometrical model information
        self.grid = grid  # whether to do a grid, but this is just checked from whether the input contains lists below
        self.data = Data(data_dir, data_files)  # create a Data object to contain the OI data
        self.diagnostics = diagnostics  # whether to print out diagnostic plots for every epoch
        self.dir0 = name  # directory name under which to store everything
        self.dirorig = os.getcwd()

        # create directory to store results
        if self.dir0 != "":
            try:
                os.makedirs(self.dir0)
            except FileExistsError:
                underline("Working in an already existing folder:")
                print(os.path.join(os.getcwd(), self.dir0))
        else:
            warn("Will put all the outputs in {os.getcwd()}")
            warn("It may overwrite files if they already exist!")

        # Creating dictionary with image recsontruction parameters
        self.params = {
            "mu": self.mu,
            "ps": self.ps,
            "epochs": self.epochs,
            "nrestart": self.nrestart,
            "use_low_cp_approx": self.use_low_cp_approx,
            "fstar": sparco.fstar,
            "dstar": sparco.dstar,
            "denv": sparco.denv,
            "udstar": sparco.udstar,
            "fsec": sparco.fsec,
            "dsec": sparco.dsec,
            "xsec": sparco.xsec,
            "ysec": sparco.ysec,
            "wave0": sparco.wave0,
        }

        # checking if grid of reconstructions needed
        ngrid, niters = 0, 1  # ngrid is number of gridded parameters, niters = total iterations
        gridpars, gridvals = [], []
        # __dict__ returns a dictionary of the instance attributes
        # .items() returns these as key-value tupel pairs in a dictionary view you can run a for loop over
        for x, v in self.__dict__.items():
            if isinstance(v, list):  # check if attributes are lists
                self.grid = True  # if so set grid to True
                ngrid += 1
                gridpars.append(x)  # adapt list of grid parameters and values
                gridvals.append(v)
                niters *= len(v)  # adapt the total number of iterations accordingly (it's a grid so you just multiply)
        # same for SPARCO parameters
        for x, v in sparco.__dict__.items():
            if isinstance(v, list):
                self.grid = True
                ngrid += 1
                gridpars.append(x)
                gridvals.append(v)

        # Run a single image reconstruction or a grid
        if self.grid:
            self.niters = niters
            self.ngrid = ngrid
            self.gridpars = gridpars
            self.gridvals = gridvals
            # print(gridvals)
            self.iterable = itertools.product(*self.gridvals)  # make cartesian product by unpacking lists in gridvals
            inform(
                f"Making an image reconstruction grid ({niters} reconstructions) on {ngrid} parameter(s): {gridpars}"
            )
            self.run_grid(reinit_gen=reinit_gen)  # run a grid
        else:
            inform("Running a single image reconstruction")
            self.dir = "img_rec"  # set the output directory of the GAN object for this image reconstruction
            self.single_img_rec(reinit_gen=reinit_gen)

    def run_grid(self, reinit_gen=False):
        """
        Run a grid of image reconstructions.

        Parameters
        ----------
        reinit_gen : bool, optional
            Whether to re-initialize generator state each iteration.
        """
        for i, k in zip(self.iterable, np.arange(self.niters)):
            state = ""
            dir_name = "img_rec"
            # iterate over the different points in the grid
            for pars, p in zip(self.gridpars, np.arange(self.ngrid)):
                # get the value for parameter number p in the current considered grid point i
                # and store those in the self.params dict
                self.params[f"{pars}"] = i[p]
                state += f"{pars}={i[p]}"
                dir_name += f"_{pars}={i[p]}"  # create the appropriate folder name for the current grid point

            self.dir = dir_name
            try:
                os.makedirs(os.path.join(self.dir0, self.dir))
            except FileExistsError:
                fail("The following folder already exists: {self.dir0, self.dir)}")
                fail("Please define another folder by changing the name keyword")
                fail("in the imag_reconstruction command")
                sys.exit(1)

            inform2(f"Image reconstruction with {state}")

            self.img_rec(reinit_gen=reinit_gen)  # do an image reconstruction  for the current grid point

    def single_img_rec(self, reinit_gen=False):
        """
        Run a single image reconstruction.

        Parameters
        ----------
        reinit_gen : bool, optional
            Whether to re-initialize generator state each iteration.
        """
        inform2("Single image reconstruction started")
        self.dir = "img_rec"
        try:
            os.makedirs(os.path.join(self.dir0, self.dir))
        except FileExistsError:
            fail("The following folder already exists: {self.dir0, self.dir)}")
            fail("Please define another folder by changing the name keyword")
            fail("in the image_reconstruction command")
            sys.exit(1)

        self.img_rec(reinit_gen=reinit_gen)

    def img_rec(self, reinit_gen=False):
        """
        Perform a single image reconstruction.

        Parameters
        ----------
        reinit_gen : bool, optional
            Whether to re-initialize generator state each iteration.
        """
        # get parameters
        params = self.params
        mu = params["mu"]
        # create data loss
        data_loss_function = self.set_dataloss()  # the dataloss function is what does the whole V2 T3PHI calculation
        # lists to store loss terms across restarts
        # NOTE: DIS_LOSS_RESTARTS WILL HOLD the -log(D) TERM (i.e. discriminator loss not yet multiplied by the
        # appropriate regularization weight)
        chi2_restarts, dis_loss_restarts = [], []
        imgs = []  # to store different images accross epochs
        vects = []  # to store different noise vectors
        num_iterations = range(params["nrestart"])  # number of iterations
        if self.diagnostics:
            print("#restart\tftot\tfdata\tfrgl")
        for r in num_iterations:
            # Re init GAN
            # 'reinit=True' only re-initializes the generator state. The optimizer in the returned gan is then just
            # the self.opt optimizer (i.e. the one shared by the entire GAN class instance), and thus shared
            self.gan = self.create_gan(reinit=reinit_gen)

            # by default, the Adam optimizer state is reset in a new optimizer object, this can be changed
            # by setting reset_opt = False in the argument list of the GAN class __init__
            if self.reset_opt:
                opt = "optimizers." + self.opt.name  # python command get the optimizer
                opt = eval(opt)

                # TODO: this seems like a pretty weird way of doing it, since here we call from_config specifically from
                # the Adam subclass of the Optimizers class using this stupid eval -> can't we just call from_config
                # from the parent class Optimizer instead?
                opt = opt.from_config(self.opt.get_config())

                # TODO: using this structure self.gan gets compiled twice; once in the self.create_gan() call above,
                # and once in this statement, maybe rewrite so we don't need to recompile so often
                # note this also assigns the weights to the losses
                # NOTE: The cross entropy function is passed here to describe the regularization term -log(D), with
                # D the discriminator output. This is achieved by setting y_true = 1 (in which case the binary
                # cross entropy term just reduces to -log(D)).
                self.gan.compile(loss=[cross_entropy, data_loss_function], optimizer=opt, loss_weights=[mu, 1])
                # cross_entropy loss is applied to the 1st gan Keras model ouptut -> e.g. the discriminator's value
                # the data loss one is applied to the second output of the model, which is the generator image ->
                # this is then just passed onto the set_dataloss function which compares to the actual OIdata

            # generating the noise vector for this restart
            noisevector = np.array([np.random.normal(0, 1, 100)])
            # target values, for the discriminator output it makes sense (causes the binary cross entropy term to
            # reduce to -log(D)), the data_loss one is just a dummy one since the set_dataloss function doesn't make
            # use of it really
            y_target = [np.ones(1), np.ones(1)]
            # the loop on epochs with one noise vector
            if self.diagnostics:  # keep track of diagnostics accross epochs if needed
                dis_loss_epochs = []  # regularization loss term
                chi2_epochs = []  # chi2 data loss term
            # iterate over epochs
            epochs = range(1, params["epochs"] + 1)
            for e in range(1, params["epochs"] + 1):
                ### TEST CODE TO SEE WHICH FORM OF REGULARIZATION LOSS HIST HOLDS (this is manual calculation)
                # if e == params["epochs"]:
                #    # RESULTS FROM MANUAL CALCULATION
                #    print("\n")
                #    print(f"EPOCH {e} =====================")
                #    disc_prediction = self.gan.predict(noisevector)[0]
                #    img_prediction = self.gen.predict(noisevector)
                #    print(f"y_pred from discriminator (calling through self.gan): {disc_prediction}")
                #    print(
                #        f"output from calling predict directly on the discriminator: {self.dis.predict(img_prediction)[0, 0]}"
                #    )
                #    print(f"-log(y_pred): {-np.log(disc_prediction)}")
                #    print(f"-mu log(y_pred): {-mu * np.log(disc_prediction)}")

                # perform a training step with noisvector input and y_gen true labels
                hist = self.gan.train_on_batch(noisevector, y_target)  # hist gives us the loss terms
                # NOTE: HIST REGULATORY LOSS TERM IS NOT YET MULTIPLIED WITH THE APPROPRIATE REGULARIZATION WEIGHT
                # hist[0] thus contains the total loss (regularization weight is taken into account here),
                # hist[1] contains -log(D) (with D the discriminator output) and hist[2] the reduced chi2

                if self.diagnostics:  # add losses to diagnostics if needed
                    dis_loss_epochs.append(hist[1])  # note the losses are outputs
                    chi2_epochs.append(hist[2])

                ### TEST CODE TO SEE WHICH FORM OF REGULARIZATION LOSS HIST HOLDS (this is straight from hist)
                # if e == params["epochs"]:
                #
                #    # RESULTS FROM HIST
                #    print(f"Full return from train_on_batch: {hist}")
                #    print(f"Discriminator weight: {mu}")
                #    print(f"Discriminator loss term returned by train_on_batch: {hist[1]}")
                #    print("===============================")
                #    print("\n")

            img = self.get_image(noisevector)  # retrieve the image
            img = (img + 1) / 2  # remap the image to fall in the 0 to 1 range
            # normalize the image by the total image flux (should be more stable for variance
            # estimation).
            img = img / np.sum(img)
            if self.diagnostics:
                self.give_imgrec_diagnostics(hist, chi2_epochs, dis_loss_epochs, r, epochs, mu)
                self.save_image_from_noise(
                    noisevector, name=os.path.join(self.dir0, self.dir, f"image_restart{r}.png")
                )  # get an image from the noise vector and save at the correct spot
            chi2_restarts.append(hist[2])  # data loss (chi2)
            dis_loss_restarts.append(hist[1])  # discriminator loss (-log(D) NOT multiplied by the weight)
            imgs.append(img[:, ::-1])  # append the image to the list of images
            vects.append(noisevector)  # append used noise vector to a list

        self.save_cube(
            imgs, [chi2_restarts, dis_loss_restarts], name="cube.fits"
        )  # save images accross restarts to fits cube
        # save the median image and best fits images, both into plots and into fits files
        self.save_images(imgs, [chi2_restarts, dis_loss_restarts])
        self.plot_loss_evol(chi2_restarts, dis_loss_restarts)  # plot and save the loss evolution accross restarts

    def save_images(self, image, losses):
        """
        Save median and best image fits.

        Parameters
        ----------
        image : list
            List of images.
        losses : list
            List of loss values.
        """
        # NOTE: `frgl` is defined as -log(D) (with D the discriminator outpu)

        # TODO: rewrite the median function to be more robust against
        # first find the median
        median_img = np.median(image, axis=0)  # get the median image (NOTE: the images are normalized before)
        median_img /= np.sum(median_img)  # normalize the median image by its own total intensity
        # some reshaping going on, ::-1 means the order is reversed along the axis
        # NOTE: note this also remaps the median image back into the -1 to 1 interval (which the discriminator and
        # data_loss() function expect).
        median_remap = np.reshape(median_img[:, ::-1] * 2 - 1, (1, self.npix, self.npix, 1))
        # get the associated losses
        fdata_med = self.data_loss(1, median_remap).numpy()  # get data loss the 1 is just a dummy value, it's not used
        frgl_med = -np.log(self.dis.predict(median_remap)[0, 0])  # get regulator loss (-log(D), D = disc output)
        ftot_med = fdata_med + self.params["mu"] * frgl_med  # calc total loss
        # plot median image and save it to location
        self.plot_image(
            median_img,
            name=os.path.join(self.dir0, self.dir, "median_image.png"),
            chi2_label=f"chi2={fdata_med:.2f} ; frgl={frgl_med:.2f} ; ftot={ftot_med:.2f}",
        )
        # save median image in a fits file
        self.image_to_fits(
            median_img,
            ftot_med,
            fdata_med,
            frgl_med,
            name=os.path.join(self.dir0, self.dir, "median_image.fits"),
        )

        # Same but for best image (best meaning lowest total loss) out of all restarts
        fdata = np.array(losses[0])
        frgl = np.array(losses[1])
        ftot = fdata + self.params["mu"] * frgl

        idx_best = np.argmin(ftot)  # best image index in the cube
        img_best = image[idx_best]
        ftot_best = ftot[idx_best]
        fdata_best = fdata[idx_best]
        frgl_best = frgl[idx_best]
        # plot best image
        self.plot_image(
            img_best,
            name=os.path.join(self.dir0, self.dir, "best_image.png"),
            chi2_label=f"chi2={fdata_best:.2f} ; frgl={frgl_best:.2f} ; ftot={ftot_best:.2f}",
        )
        # save best image to fits file
        self.image_to_fits(
            img_best,
            ftot_best,
            fdata_best,
            frgl_best,
            name=os.path.join(self.dir0, self.dir, "best_image.fits"),
        )

    def image_to_fits(self, image, ftot, fdata, frgl, name="image.fits"):
        """
        Save an image to a FITS file.

        Parameters
        ----------
        image : numpy.ndarray
            Image to save.
        ftot : float
            Total loss.
        fdata : float
            Data loss.
        frgl : float
            Regularization term -log(D).
        name : str, optional
            Name of the saved FITS file.
        """
        params = self.params
        mu = params["mu"]  # regularization weight
        npix = self.npix

        header = fits.Header()

        header["SIMPLE"] = "T"  # whether conforms to FITS standard
        header["BITPIX"] = -64  # how many bits per pixel value in primary

        header["NAXIS"] = 2  # how many axes in primary
        header["NAXIS1"] = npix  # number of values per dimension
        header["NAXIS2"] = npix
        # note that if you pass a tuple the second element is interpreted as the comment
        header["EXTEND"] = "T"  # whether extensions (extra extensions on top of `PRIMARY` are allowed)
        header["CRVAL1"] = (0.0, "Coordinate system value at reference pixel")  # coordinate calue at reference pixel
        header["CRVAL2"] = (0.0, "Coordinate system value at reference pixel")
        header["CRPIX1"] = npix / 2  # reference pixel
        header["CRPIX2"] = npix / 2
        header["CTYPE1"] = ("milliarcsecond", "RA in mas")
        header["CTYPE2"] = ("milliarcsecond", "DEC in mas")
        # increment per pixel along the axis direction (note negative here because the positive x-axis points towards
        # the left in interferometry)
        header["CDELT1"] = -1 * params["ps"]
        header["CDELT2"] = params["ps"]

        # additional fields with prefix `S` for the SPARCO parameters
        header["SWAVE0"] = (params["wave0"], "SPARCO central wavelength in (m)")  # central wavelength
        header["SPEC0"] = "pow"  # type of spectrum used for the environment (reconstructed image), i.e. power law
        header["SIND0"] = (params["denv"], "spectral index of the image")  # spectral index for the image

        header["SNMODS"] = (2, "number of SPARCO parameteric models")  # number of components in SPARCO model

        header["SMOD1"] = ("UD", "model for the primary")  # model for the primary, `UD` stands for a uniform disk model
        header["SFLU1"] = (params["fstar"], "SPARCO flux ratio of primary")  # flux ratio contribution of primary
        header["SPEC1"] = "pow"  # spectrum type used for primary, i.e. power law
        header["SDEX1"] = (0, "dRA Position of primary")  # X position of primary
        header["SDEY1"] = (0, "dDEC position of primary")  # Y position of primary
        header["SIND1"] = (params["dstar"], "Spectral index of primary")  # spectral index of primary
        header["SUD1"] = (params["udstar"], "UD diameter of primary")  # uniform diameter of primary

        header["SMOD2"] = "star"  # model for secondary, `star` just a point source
        header["SFLU2"] = (params["fsec"], "SPARCO flux ratio of secondary")  # flux ratio of secondary
        header["SPEC2"] = "pow"  # type of spectrum for secondary, i.e. power law
        header["SDEX2"] = (params["xsec"], "dRA Position of secondary")  # position of secondary
        header["SDEY2"] = (params["ysec"], "dDEC position of secondary")
        header["SIND2"] = (params["dsec"], "Spectral index of secondary")  # spectral index of secondary

        header["NEPOCHS"] = params["epochs"]  # number of epochs per restart
        header["NRSTARTS"] = params["nrestart"]  # number of restarts

        header["FTOT"] = ftot  # total loss (i.e. data loss + weigth-multiplied regularization)
        header["FDATA"] = fdata  # data loss (i.e. chi2)
        header["FRGL"] = frgl  # regularization term (not multiplied by the regularization weight)
        header["MU"] = mu  # regularization weight

        # Make the headers
        prim_hdu = fits.PrimaryHDU(image, header=header)  # primary HDU
        hdul = fits.HDUList([prim_hdu])  # primary HDU is the only one to be added
        hdul.writeto(os.path.join(self.dir0, self.dir, name), overwrite=True)  # write to file

    def plot_loss_evol(self, chi2, dis_loss):
        """
        Plot the loss evolution across restarts.

        Parameters
        ----------
        chi2 : list
            List of chi2 values.
        dis_loss : list
            List of discriminator loss values.
        """
        mu = self.params["mu"]  # regularization weight

        fig, ax = plt.subplots()
        plt.plot(chi2, label="f_data")
        plt.plot(mu * np.array(dis_loss), label="mu * f_discriminator")
        plt.plot(np.array(chi2) + mu * np.array(dis_loss), label="f_tot")
        plt.legend()
        plt.xlabel("#restart")
        plt.ylabel("Losses")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir0, self.dir, "lossevol.png"), dpi=250)
        plt.close()

    def save_cube(self, cube, losses, name="cube.fits"):
        """
        Save a cube of images and their diagnostics across restarts in a .fits file.

        Parameters
        ----------
        cube : numpy.ndarray
            Cube of images.
        losses : list
            List of loss values.
        name : str, optional
            Name of the saved FITS file.
        """
        params = self.params
        mu = params["mu"]
        npix = self.npix

        header = fits.Header()  # header for the primary

        header["SIMPLE"] = "T"
        header["BITPIX"] = -64

        header["NAXIS"] = 3
        header["NAXIS1"] = npix
        header["NAXIS2"] = npix
        header["NAXIS3"] = params["nrestart"]

        header["EXTEND"] = "T"
        header["CRVAL1"] = (0.0, "Coordinate system value at reference pixel")
        header["CRVAL2"] = (0.0, "Coordinate system value at reference pixel")
        header["CRPIX1"] = npix / 2
        header["CRPIX2"] = npix / 2
        header["CTYPE1"] = ("milliarcsecond", "RA in mas")
        header["CTYPE2"] = ("milliarcsecond", "DEC in mas")
        header["CDELT1"] = -1 * params["ps"]  # reversed because of East-West axis direction convention
        header["CDELT2"] = params["ps"]

        header["CDELT3"] = 1.0
        header["CTYPE3"] = ("Nrestart", "Number of generator restarts")

        header["SWAVE0"] = (params["wave0"], "SPARCO central wavelength in (m)")
        header["SPEC0"] = "pow"
        header["SIND0"] = (params["denv"], "spectral index of the image")

        header["SNMODS"] = (2, "number of SPARCO parameteric models")

        header["SMOD1"] = ("UD", "model for the primary")
        header["SFLU1"] = (params["fstar"], "SPARCO flux ratio of primary")
        header["SPEC1"] = "pow"
        header["SDEX1"] = (0, "dRA Position of primary")
        header["SDEY1"] = (0, "dDEC position of primary")
        header["SIND1"] = (params["dstar"], "Spectral index of primary")
        header["SUD1"] = (params["udstar"], "UD diameter of primary")

        header["SMOD2"] = "star"
        header["SFLU2"] = (params["fsec"], "SPARCO flux ratio of secondary")
        header["SPEC2"] = "pow"
        header["SDEX2"] = (params["xsec"], "dRA Position of secondary")
        header["SDEY2"] = (params["ysec"], "dDEC position of secondary")
        header["SIND2"] = (params["dsec"], "Spectral index of secondary")

        header["NEPOCHS"] = params["epochs"]
        header["NRSTARTS"] = params["nrestart"]
        header["MU"] = mu

        # define columns for the losses
        fdata = np.array(losses[0])
        frgl = np.array(losses[1])  # contains -log(D) (with D the discriminator output)
        ftot = fdata + mu * frgl
        colftot = fits.Column(name="ftot", array=ftot, format="E")
        colfdata = fits.Column(name="fdata", array=fdata, format="E")
        colfrgl = fits.Column(name="frgl", array=frgl, format="E")
        cols = fits.ColDefs([colftot, colfdata, colfrgl])

        headermetrics = fits.Header()  # create header for the metrics extension
        headermetrics["TTYPE1"] = "FTOT"  # total loss (i.e. data loss + weigth-multiplied regularization)
        headermetrics["TTYPE2"] = "FDATA"  # data loss (i.e. chi2)
        headermetrics["TTYPE3"] = "FRGL"  # regularization term (not multiplied by the weight)
        headermetrics["MU"] = mu  # regularization weight

        # Make the primary and secondary HDU's
        prim_hdu = fits.PrimaryHDU(cube, header=header)
        sec_hdu = fits.BinTableHDU.from_columns(cols, header=headermetrics, name="METRICS")

        hdul = fits.HDUList([prim_hdu, sec_hdu])  # make list of HDUs
        hdul.writeto(os.path.join(self.dir0, self.dir, name), overwrite=True)  # write to file

    def filter_pca_cluster(self, cube, ncluster=1):
        """
        Method to filter the worst (in terms of total loss) `ncluster` clusters in PCA-projected space.

        Parameters
        ----------
        cube : Cube
            The post_organic Cube object containing the generated images.
        ncluster : int, optional
            Number of clusters to filter.
        """
        # load in required data
        img_cube = cube.cube  # cube of all images
        kmeanftot = cube.kmeanftot  # median ftot over cluster groups
        cluster_ids = cube.kmeans  # ID numbers of kmeans clusters

        # identify `ncluster` worst clusters based on median ftot
        idx_ftot_sorted = np.argsort(kmeanftot)  # indices to sort by ftot value
        bad_cluster_ids = idx_ftot_sorted[-ncluster:]  # cluster ID numbers of bad clusters

        # filter bad clusters from the cube
        img_cube_filtered = []  # array to contain images with chosen cluster groups filtered out
        ftot_filtered = []  # arrays to contain loss terms for filtered images
        fdata_filtered = []
        frgl_filtered = []
        for i in range(np.shape(img_cube)[0]):
            if cluster_ids[i] not in bad_cluster_ids:
                img_cube_filtered.append(img_cube[i, :, :])  # add to filtered images
                ftot_filtered.append(cube.ftot[i])
                fdata_filtered.append(cube.fdata[i])
                frgl_filtered.append(cube.frgl[i])
            else:
                print(f"Bad image; ID: {i}")
        img_cube_filtered = np.array(img_cube_filtered)  # cast to numpy array
        ftot_filtered = np.array(ftot_filtered)
        fdata_filtered = np.array(fdata_filtered)
        frgl_filtered = np.array(frgl_filtered)

        # calculate median image after filtering
        median_img_filtered = np.median(img_cube_filtered, axis=0)
        median_img_filtered = median_img_filtered / np.sum(median_img_filtered)

        # get the associated loss terms
        # NOTE: note this also remaps the median image back into the -1 to 1 interval (which the discriminator and
        # data_loss() function expect).
        median_filtered_remap = np.reshape(median_img_filtered[:, ::-1] * 2 - 1, (1, self.npix, self.npix, 1))
        fdata_median_filtered = self.data_loss(
            1, median_filtered_remap
        ).numpy()  # get data loss the 1 is just a dummy value, it's not used
        # get regulator loss, note this is not multiplied by the weight yet
        frgl_median_filtered = -np.log(self.dis.predict(median_filtered_remap)[0, 0])
        ftot_median_filtered = fdata_median_filtered + self.params["mu"] * frgl_median_filtered  # calc total loss

        # plot median image and save it to location
        self.plot_image(
            median_img_filtered,
            name=os.path.join(self.dir0, self.dir, "median_image_filtered.png"),
            chi2_label=f"chi2={fdata_median_filtered:.2f} ; frgl={frgl_median_filtered:.2f} "
            + f"; ftot={ftot_median_filtered:.2f}",
        )
        # save median image in a fits file
        self.image_to_fits(
            median_img_filtered,
            ftot_median_filtered,
            fdata_median_filtered,
            frgl_median_filtered,
            name=os.path.join(self.dir0, self.dir, "median_image_filtered.fits"),
        )

        # Save filtered cube of images
        self.save_cube(img_cube_filtered, [fdata_filtered, frgl_filtered], name="cube_filtered.fits")

    def give_imgrec_diagnostics(self, hist, chi2, discloss, r, epochs, mu):
        """
        Print and save the plots of the diagnostics across epochs for every restart.

        Parameters
        ----------
        hist : list
            List of loss values.
        chi2 : list
            List of chi2 values.
        discloss : list
            List of discriminator loss values.
        r : int
            Restart index.
        epochs : list
            List of epochs.
        mu : float
            Regularization weight.
        """
        # print(r, hist[0], hist[2], mu * hist[1], sep="\t")  # print loss terms at the final epoch of the restart
        # print(mu)  # print the weight of the discriminator loss
        # print(type(discloss))
        # print(discloss)  # print the discriminator loss
        fig, ax = plt.subplots()
        plt.plot(epochs, chi2, label="f_data")
        plt.plot(epochs, mu * np.array(discloss), label="mu * f_discriminator")
        plt.plot(epochs, np.array(chi2) + mu * np.array(discloss), label="f_tot")
        plt.legend()
        plt.xlabel("#epochs")
        plt.ylabel("Losses")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir0, self.dir, f"lossevol_restart{r}.png"), dpi=250)
        plt.close()

    # !! Master function to calculate the data loss !!
    # TODO: check correctness
    # TODO: current iteration of bootstrap makes no sense, since it resamples data each time the function is evaluated,
    # i.e. each epoch, while it should be called each restart iteration instead
    # TODO: adapt to always use phasors instead of just the raw closure phase values
    # TODO: really don't like the structure of this function, lots of concerning scope things going on, nested functions
    # the function itself returns the function. However, it works so maybe not touch too much for now
    def set_dataloss(self):
        """
        Master function to calculate the data loss.

        Returns
        -------
        data_loss : function
            Data loss function.
        """
        data = self.data  # retrieve data object

        # if bootstrapping, resample the data
        # TODO: THE IMPLEMENTATION HERE IS WRONG
        if self.boot:
            V2, V2e, CP, CPe, waveV2, waveCP, u, u1, u2, u3, v, v1, v2, v3 = data.get_bootstrap()
        else:
            V2, V2e, CP, CPe, waveV2, waveCP, u, u1, u2, u3, v, v1, v2, v3 = data.get_data()

        params = self.params

        MAS2RAD = np.pi * 0.001 / (3600 * 180)  # conversion of milli-arcseconds to rad

        fstar = params["fstar"]
        fsec = params["fsec"]
        ud = params["udstar"] * MAS2RAD  # put uniform diameter in radian
        dstar = params["dstar"]
        denv = params["denv"]
        dsec = params["dsec"]
        xsec = params["xsec"] * MAS2RAD  # put positions in radian
        ysec = params["ysec"] * MAS2RAD
        ps = params["ps"]
        wave0 = params["wave0"]
        use_low_cp_approx = params["use_low_cp_approx"]
        nV2 = len(V2)  # number of V2 points
        nCP = len(CP)  # number of closure phase points
        npix = self.npix  # number of pixels
        spatial_freq_per_pixel = 1 / (npix * ps * MAS2RAD)  # size per pixel in FFT frequency space

        assert nV2 > 0  # throw an error if there's less than one datapoint

        # function to calculate an offcenter transformation term to be applied to a fourier transform's result
        def off_center_point_ft(x, y, u, v):
            u = tf.constant(u, dtype=tf.complex128)
            v = tf.constant(v, dtype=tf.complex128)
            return tf.math.exp(-2 * np.pi * 1j * (x * u + y * v))

        # performs a binlinear interpolation on grid at continuous pixel coordinates ufunc, vfunc
        # TODO: there has to be an inbuilt function for this no? This manual implementation seems very verbose,
        # but it could be that there's just no good support at the moment.
        def bilinear_interp(grid, ufunc, vfunc):
            ubelow = np.floor(ufunc).astype(int)
            vbelow = np.floor(vfunc).astype(int)
            uabove = ubelow + 1
            vabove = vbelow + 1
            coords = tf.constant([[[0, ubelow[i], vbelow[i]] for i in range(len(ufunc))]])
            # calculate interpolated values
            interp_values = tf.gather_nd(grid, coords) * (uabove - ufunc) * (vabove - vfunc)
            coords1 = tf.constant([[[0, uabove[i], vabove[i]] for i in range(len(ufunc))]])
            interp_values += tf.gather_nd(grid, coords1) * (ufunc - ubelow) * (vfunc - vbelow)
            coords2 = tf.constant([[[0, uabove[i], vbelow[i]] for i in range(len(ufunc))]])
            interp_values += tf.gather_nd(grid, coords2) * (ufunc - ubelow) * (vabove - vfunc)
            coords3 = tf.constant([[[0, ubelow[i], vabove[i]] for i in range(len(ufunc))]])
            interp_values += tf.gather_nd(grid, coords3) * (uabove - ufunc) * (vfunc - vbelow)
            return interp_values

        # plots a comperison between observations and observables of the reconstruction,as well as the uv coverage
        # TODO: currently not used at all
        def plotObservablesComparison(V2generated, V2observed, V2err, CPgenerated, CPobserved, CPerr):
            # v2 with residual comparison, no colors indicating wavelength
            fig, ax = plt.subplots(figsize=(3.5, 6))
            absB = np.sqrt(u**2 + v**2) / (10**6)  # baseline length in megaLambda
            plt.scatter(
                absB,
                V2generated[0],
                marker=".",
                s=40,
                label="image",
                c="b",
                alpha=0.4,
                edgecolors="k",
                linewidth=0.15,
            )
            plt.scatter(
                absB,
                V2observed,
                marker="*",
                s=40,
                label="observed",
                c="r",
                alpha=0.4,
                edgecolors="k",
                linewidth=0.15,
            )
            plt.errorbar(absB, V2observed, V2err, elinewidth=0.2, ls="none", c="r")
            plt.ylim(0, 1)
            plt.ylabel(r"$V^2$")
            plt.legend()

            plt.savefig(os.path.join(os.getcwd(), "v2_comparison_no_color.png"), dpi=250)
            plt.close()

            # plots the uv coverage
            plt.figure()
            plt.scatter(
                u / (10**6),
                v / (10**6),
                marker=".",
                c=np.real(waveV2),
                cmap="rainbow",
                alpha=0.9,
                edgecolors="k",
                linewidth=0.1,
            )
            plt.scatter(
                -u / (10**6),
                -v / (10**6),
                marker=".",
                c=np.real(waveV2),
                cmap="rainbow",
                alpha=0.9,
                edgecolors="k",
                linewidth=0.1,
            )
            plt.xlabel(r"$ u (M\lambda)$")
            plt.ylabel(r"$ v (M\lambda)$")
            plt.gca().set_aspect("equal", adjustable="box")

            plt.savefig(os.path.join(os.getcwd(), "uv_coverage.png"), dpi=250)
            plt.close()

            # cp with residual comparison without color indicating wavelength
            fig, ax = plt.subplots(figsize=(3.5, 6))
            maxB = np.maximum(
                np.maximum(np.sqrt(u1**2 + v1**2), np.sqrt(u2**2 + v2**2)),
                np.sqrt(u3**2 + v3**2),
            ) / (10**6)
            plt.scatter(
                maxB,
                CPgenerated[0].numpy(),
                s=30,
                marker=".",
                c="b",
                label="image",
                cmap="rainbow",
                alpha=0.4,
                edgecolors=colors.to_rgba("k", 0.1),
                linewidth=0.3,
            )
            plt.scatter(
                maxB,
                CPobserved,
                s=30,
                marker="*",
                label="observed",
                c="r",
                alpha=0.4,
                edgecolors=colors.to_rgba("k", 0.1),
                linewidth=0.3,
            )
            plt.errorbar(maxB, CPobserved, CPerr, ls="none", elinewidth=0.2, c="r")
            plt.legend()
            plt.ylabel(r"closure phase(radian)", fontsize=12)
            plt.xlabel(r"max($\mid B\mid)(M\lambda)$", fontsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(os.getcwd(), "cp_comparison_no_color.png"), dpi=250)

            plt.close()

        # function to calculate full visibility if passed an FT image
        def compTotalCompVis(ftImages, ufunc, vfunc, wavelfunc):
            # to compute the radial coordinate in the uv plane so compute the ft of the primary,
            # which is a uniform disk, so the ft is....
            radii = np.pi * ud * np.sqrt(ufunc**2 + vfunc**2)
            ftPrimary = tf.constant(2 * sp.jv(1, radii) / (radii), dtype=tf.complex128)
            # see extra function
            ftSecondary = off_center_point_ft(xsec, ysec, ufunc, vfunc)
            # get the total visibility amplitudes using bilinear interpolation
            VcomplDisk = bilinear_interp(
                ftImages,
                (vfunc / spatial_freq_per_pixel) + int(npix / 2),
                (ufunc / spatial_freq_per_pixel) + int(npix / 2),
            )
            # equation 4 in sparco paper:
            VcomplTotal = fstar * ftPrimary * kbackend.pow(wavelfunc / wave0, dstar)
            VcomplTotal += fsec * ftSecondary * kbackend.pow(wavelfunc / wave0, dsec)
            VcomplTotal += (1 - fstar - fsec) * VcomplDisk * kbackend.pow(wavelfunc / wave0, denv)
            VcomplTotal = VcomplTotal / (
                (fstar * kbackend.pow(wavelfunc / wave0, dstar))
                + (fsec * kbackend.pow(wavelfunc / wave0, dsec))
                + ((1 - fstar - fsec) * kbackend.pow(wavelfunc / wave0, denv))
            )
            return VcomplTotal

        # core function to actually compute the data loss function (i.e. by comparing to the OI data)
        # notice that y_true is not used, it's just a dummy value that Keras expects you to have
        # note that everything must be done using tensorflow functions in this case, otherwise you lose the
        # speed and parallelization advantage
        def data_loss(y_true, y_pred, training=True):
            # img = y_pred.numpy()[0,:,:,0]
            y_pred = tf.squeeze(y_pred, axis=3)  # remove batch dimension (it's of size 1 anyway) using squeeze
            y_pred = (y_pred + 1) / 2  # NOTE: remaps the image into the 0 to 1 interval to get positive pixel fluxes
            y_pred = tf.cast((y_pred), tf.complex128)
            y_pred = tf.signal.ifftshift(y_pred, axes=(1, 2))
            ftImages = tf.signal.fft2d(y_pred)  # is complex!!
            ftImages = tf.signal.fftshift(ftImages, axes=(1, 2))

            coordsMax = [[[[0, int(npix / 2), int(npix / 2)]]]]
            ftImages = ftImages / tf.cast(tf.math.abs(tf.gather_nd(ftImages, coordsMax)), tf.complex128)
            VcomplForV2 = compTotalCompVis(ftImages, u, v, waveV2)
            V2image = tf.math.abs(VcomplForV2) ** 2  # computes squared vis for the generated images

            V2Chi2Terms = kbackend.pow(V2 - V2image, 2) / (
                kbackend.pow(V2e, 2) * nV2
            )  # individual terms of chi**2 for V**2
            # V2Chi2Terms = V2Chi2Terms
            V2loss = kbackend.sum(V2Chi2Terms, axis=1)

            CPimage = tf.math.angle(compTotalCompVis(ftImages, u1, v1, waveCP))
            CPimage += tf.math.angle(compTotalCompVis(ftImages, u2, v2, waveCP))
            # note it's minus for the third baseline's complex angle since it's defined as the line AC instead of CA
            # in an ABC triangle (often used convention in interferometry)
            CPimage -= tf.math.angle(compTotalCompVis(ftImages, u3, v3, waveCP))
            CPchi2Terms = 2 * (1 - tf.math.cos(CP - CPimage)) / (kbackend.pow(CPe, 2) * nCP)
            if use_low_cp_approx:
                CPchi2Terms = kbackend.pow(CP - CPimage, 2) / (kbackend.pow(CPe, 2) * nCP)

            CPloss = kbackend.sum(CPchi2Terms, axis=1)

            lossValue = (kbackend.mean(V2loss) * nV2 + kbackend.mean(CPloss) * nCP) / (nV2 + nCP)

            if training:
                # plotObservablesComparison(V2image, V2, V2e, CPimage, CP, CPe)
                return tf.cast(lossValue, tf.float32)  # cast to

            else:
                # plotObservablesComparison(V2image, V2, V2e, CPimage, CP, CPe)
                return lossValue, V2loss, CPloss

        self.data_loss = data_loss
        return data_loss


# TODO: note that this class has no filtering capability at all, this of course needs to be adapted a little
# for e.g. MATISSE data -> probably have to port the OIData implemenation of DISTROI at some point, which will
# support data filtering and just a universal handling of OIFITS data
class Data:
    """
    Class to read in and store squared visibility and closure phase information.

    Parameters
    ----------
    dir : str
        Directory containing the data files.
    file : str
        Data file.
    """

    def __init__(self, dir, file):
        self.dir = dir
        self.file = file
        self.read_data()

    def read_data(self):
        """
        Read the data from the file.
        """
        data = oi.read(self.dir, self.file)
        dataObj = data.givedataJK()

        V2observed, V2err = dataObj["v2"]
        nV2 = len(V2err)

        CPobserved, CPerr = dataObj["cp"]
        nCP = len(CPerr)

        u, u1, u2, u3 = dataObj["u"]
        v, v1, v2, v3 = dataObj["v"]

        waveV2 = dataObj["wave"][0]
        waveCP = dataObj["wave"][1]

        V2 = tf.constant(V2observed)  # conversion to tensor
        V2err = tf.constant(V2err)  # conversion to tensor
        CP = tf.constant(CPobserved) * np.pi / 180  # conversion to radian & cast to tensor
        CPerr = tf.constant(CPerr) * np.pi / 180  # conversion to radian & cast to tensor
        waveV2 = tf.constant(waveV2, dtype=tf.complex128)  # conversion to tensor
        waveCP = tf.constant(waveCP, dtype=tf.complex128)  # conversion to tensor

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
        self.target = data.target[0].target[0]

    def get_data(self):
        """
        Get the squared visibility and closure phase data.

        Returns
        -------
        tuple
            Tuple containing V2, V2err, CP, CPerr, waveV2, waveCP, u, u1, u2, u3, v, v1, v2, v3.
        """
        return (
            self.V2,
            self.V2err,
            self.CP,
            self.CPerr,
            self.waveV2,
            self.waveCP,
            self.u,
            self.u1,
            self.u2,
            self.u3,
            self.v,
            self.v1,
            self.v2,
            self.v3,
        )

    def get_bootstrap(self):
        """
        Get a bootstrap sample from the data.

        Returns
        -------
        tuple
            Tuple containing V2, V2err, CP, CPerr, waveV2, waveCP, u, u1, u2, u3, v, v1, v2, v3.
        """
        V2selection = np.random.randint(0, self.nV2, self.nV2)  # sample random indices for V2
        newV2, newV2err = self.V2[V2selection], self.V2err[V2selection]  # get bootstrap sample
        CPselection = np.random.randint(0, self.nCP, self.nCP)  # sample random indices for closure phase
        newCP, newCPerr = self.CP[CPselection], self.CPerr[CPselection]  # get bootstrap sample
        newu, newu1, newu2, newu3 = (
            self.u[V2selection],
            self.u1[CPselection],
            self.u2[CPselection],
            self.u3[CPselection],
        )
        newv, newv1, newv2, newv3 = (
            self.v[V2selection],
            self.v1[CPselection],
            self.v2[CPselection],
            self.v3[CPselection],
        )
        newwavelV2 = self.waveV2[V2selection]
        newwavelCP = self.waveCP[CPselection]
        return (
            newV2,
            newV2err,
            newCP,
            newCPerr,
            newwavelV2,
            newwavelCP,
            newu,
            newu1,
            newu2,
            newu3,
            newv,
            newv1,
            newv2,
            newv3,
        )


class SPARCO:
    """
    Class defining geometric components in the SPARCO paradigm.

    Parameters
    ----------
    wave0 : float, optional
        Central wavelength at which flux ratios are defined.
    fstar : float, optional
        Primary flux contribution.
    dstar : float, optional
        Spectral index of primary.
    denv : float, optional
        Spectral index of reconstructed environment.
    udstar : float, optional
        Uniform diameter of primary in mas.
    fsec : float, optional
        Flux contribution of secondary.
    dsec : float, optional
        Spectral index of secondary.
    xsec : float, optional
        Position of the secondary in mas.
    ysec : float, optional
        Position of the secondary in mas.

    Notes
    -----
    Currently only geared to describing binary stars, with a uniform diameter
    primary and a point-source secondary.
    """

    def __init__(
        self,
        wave0=1.65e-6,
        fstar=0.6,
        dstar=-4.0,
        denv=0.0,
        udstar=0.01,
        fsec=0.0,
        dsec=-4,
        xsec=0.0,
        ysec=0.0,
    ):
        self.wave0 = wave0  # central wavelength at which flux ratio's are defined
        self.fstar = fstar  # primary flux contrib
        self.dstar = dstar  # spectral index of primary
        self.denv = denv  # spectral index of reconstructed environment
        self.udstar = udstar  # uniform diameter of primary in mas
        self.fsec = fsec  # flux contrib of secondary
        self.dsec = dsec  # spectral index of secondary
        self.xsec = xsec  # position of the secondary (primary is at 0 always)
        self.ysec = ysec


# NOTE: Question to self: are the training images normalized first? How
# are the images already pre-processed by Jacques and Rik?
# Answer: they're always remapped to the -1 to 1 range (the same range that the generator produces)


class InputImages:
    """
    A class to format the images the way keras can augment them.

    Parameters
    ----------
    dir : str
        Directory which to search.
    file : str
        Filename of cube of files.
    imagesize : int, optional
        Image pixel size.
    load_from_cube : bool, optional
        Load image from the cube.
    featurewise_center : bool, optional
        Set input mean to 0 over the dataset.
    samplewise_center : bool, optional
        Set each sample mean to 0.
    featurewise_std_normalization : bool, optional
        Divide inputs by std of the dataset.
    samplewise_std_normalization : bool, optional
        Divide each input by its std.
    zca_whitening : bool, optional
        Apply ZCA whitening (zero phase component analysis).
    zca_epsilon : float, optional
        Epsilon for ZCA whitening.
    rotation_range : int, optional
        Randomly rotate images in the range (degrees, 0 to 180).
    width_shift_range : float, optional
        Randomly shift images horizontally (fraction of total width).
    height_shift_range : float, optional
        Randomly shift images vertically (fraction of total height).
    shear_range : float, optional
        Set range for random shear.
    zoom_range : list, optional
        Set range for random zoom factor.
    channel_shift_range : float, optional
        Set range for random channel shifts.
    fill_mode : str, optional
        Set mode for filling points outside the input boundaries.
    cval : float, optional
        Value used for fill_mode = "constant".
    horizontal_flip : bool, optional
        Randomly flip images.
    vertical_flip : bool, optional
        Randomly flip images.
    rescale : float, optional
        Set rescaling factor which the images are multiplied by.
    preprocessing_function : function, optional
        Set function that will be applied on each input.
    data_format : str, optional
        Image data format, either "channels_first" or "channels_last".
    validation_split : float, optional
        Fraction of images reserved for validation.
    """

    def __init__(
        self,
        dir,  # direcory which to search
        file,  # filename of cube of files
        imagesize=128,  # image pixel size
        load_from_cube=True,  # load image from the cube
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening (zero phase component analysis)
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0,
        shear_range=0.0,  # set range for random shear
        zoom_range=[0.9, 1.1],  # set range for random zoom factor
        channel_shift_range=0.0,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode="nearest",
        cval=-1.0,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        # set rescaling factor which the images are multiplied by (applied after any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0,
    ):
        self.npix = imagesize
        # expand to full filepath
        # NOTE: naming is confusing w.r.t. class __init__ function
        self.dir = os.path.join(os.path.expandvars(dir), file)
        self.loadfromCube = load_from_cube
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_epsilon = zca_epsilon
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.data_format = data_format
        self.validation_split = validation_split

        self.load()

    # general function to load in the training images
    def load(self):
        """
        Load the training images.
        """
        # initialize ImageDataGenerator with all the properties passed along to the __init__ function
        self.data_gen = ImageDataGenerator(
            featurewise_center=self.featurewise_center,
            samplewise_center=self.samplewise_center,
            featurewise_std_normalization=self.featurewise_std_normalization,
            samplewise_std_normalization=self.samplewise_std_normalization,
            zca_whitening=self.zca_whitening,
            zca_epsilon=self.zca_epsilon,
            rotation_range=self.rotation_range,
            width_shift_range=self.width_shift_range,
            height_shift_range=self.height_shift_range,
            shear_range=self.shear_range,
            zoom_range=self.zoom_range,
            channel_shift_range=self.channel_shift_range,
            fill_mode=self.fill_mode,
            cval=self.cval,
            horizontal_flip=self.horizontal_flip,
            vertical_flip=self.vertical_flip,
            rescale=self.rescale,
            preprocessing_function=self.preprocessing_function,
            data_format=self.data_format,
            validation_split=self.validation_split,
        )
        if self.loadfromCube:  # if loading from a cube call correct function
            self.images = self.load_data_from_cube()  # initialize the images property (used at beginning of train())
        else:
            self.images = self.load_data()  # otherwise call other function to initialize image property
        inform(f"Input images loaded successfully with shape {self.images.shape}")

    def load_data_from_cube(self):
        """
        Load data from a .fits cube of images.

        Returns
        -------
        numpy.ndarray
            Array containing the image information and dimensions.
        """
        cube = fits.getdata(self.dir, ext=0)  # get data from primary header (where the images are stored in this case)

        images = []  # list of images
        for i in np.arange(len(cube)):  # iterate over the different images
            img0 = PIL.Image.fromarray(cube[i])  # put the images in a PIL image object
            # resize the image using bilinear interpolation to the size requirements of the network if needed
            img = img0.resize((self.npix, self.npix), PIL.Image.BILINEAR)
            img /= np.max(img)  # normalize the image by the maximum flux
            images.append(img)  # append to the list of images

        # put the images back into a 4D numpy array (dim 1 = sample from set, dim 2, 3 =
        # image spatial dimensions, dimension 4 = number of channels, in this case 1 so just add an axis)
        # also remap the images to lie between -1 and 1 (the same output range as the generator produces due to its
        # final tanh activation).

        # NOTE: inherently assumes data_format = 'channels_last' -> might be safer to set this by default instead of
        # leaving it to the __init__ function otherwise the user settings in ~/.keras/keras.json might override this
        # (see the documentation of Keras' ImageDataGenerator).
        newcube = np.array(images)
        newcube = (newcube[:, :, :, np.newaxis] - 0.5) * 2

        return newcube

    def load_data(self):
        """
        Load data from individual .fits files.

        Returns
        -------
        numpy.ndarray
            Array containing the image information and dimensions.
        """
        dirs = glob.glob(self.dir)
        images = []
        for i in np.arange(len(dirs)):
            image = fits.getdata(dirs[i], ext=0)
            img = PIL.Image.fromarray(image)
            img = img.resize((self.npix, self.npix), PIL.Image.BILINEAR)
            img /= np.max(img)
            images.append(img)
        newcube = np.array(images)
        newcube = (newcube[:, :, :, np.newaxis] - 0.5) * 2

        return newcube


if "__main__" == __name__:
    test = GAN()
    dir_name = "./"
    file = "Test"
    imgs = InputImages(dir_name, file)
