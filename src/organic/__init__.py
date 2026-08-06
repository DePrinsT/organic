"""A package for image reconstruction in astronomical optical interferometry \
using generative adversarial networks.
"""

# Version string, which is automatically used when building package.
__version__ = "0.0.4"


# Expose submodules to the main package namespace after importing main package.
from . import (
    _consts as _consts,
)
from . import (
    img_rec as img_rec,
)
from . import (
    model_training as model_training,
)
from . import (
    oi_data as oi_data,
)
from . import (
    sparco as sparco,
)
from . import (
    training_data as training_data,
)
from . import (
    utils as utils,
)

# Expose main objects/functions from modules which will see frequent use to the main
# package namespace.
