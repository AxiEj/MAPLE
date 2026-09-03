"""Private, non-admitted Torch smooth finite-dielectric PCM."""
from .functional import Q_RAW_FROM_CARTESIAN, block_permutation
from .identity import CAPABILITIES, CONFIGURATION_CONTRACT_ID, MODEL_ID, PROVIDER_ID, SCALAR_ID
from .scalar import SolveDiagnostic, TorchSmoothPCM
__all__=["TorchSmoothPCM","SolveDiagnostic","Q_RAW_FROM_CARTESIAN","block_permutation","MODEL_ID","PROVIDER_ID","SCALAR_ID","CONFIGURATION_CONTRACT_ID","CAPABILITIES"]
