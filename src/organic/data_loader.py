"""Module for loading GAN training data images."""


# TODO: should support zooming, adding overresolved flux, rotating randomly
# etc. etc. Reading in should be done by reading .npy files. Returned batches
# should be tuples of JAX arrays for usage. Batch size should be free variable.
# Be careful while computing that we don't calculate gradients w.r.t these
# arrays.
class TrainingImageLoader:
    """Class for loading, augmenting and returning batches of GAN training
    images.
    """

    pass
