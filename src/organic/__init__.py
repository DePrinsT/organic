"""A package for image reconstruction in astronomical optical interferometry \
using generative adversarial networks.
"""

# version string, which is automatically used when building package
__version__ = "0.0.4"


# Expose submodules to the user namespace after importing main package
from . import (
    _consts as _consts,
)
from . import (
    data_loader as data_loader,
)
from . import (
    model_training as model_training,
)
from . import (
    oi_data as oi_data,
)
from . import (
    reconstruction as reconstruction,
)
from . import (
    sparco as sparco,
)
from . import (
    utils as utils,
)
