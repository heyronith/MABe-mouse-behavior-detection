from .tcn import DilatedTCN, TCNBlock, MABeModel
from .dual_branch import DualBranchModel, TemporalTransformer, TemporalTransformerLayer, EnhancedTCN, EnhancedTCNBlock

__all__ = [
    'DilatedTCN', 'TCNBlock', 'MABeModel',
    'DualBranchModel', 'TemporalTransformer', 'TemporalTransformerLayer',
    'EnhancedTCN', 'EnhancedTCNBlock'
]

# Export the main model for easy access
MainModel = DualBranchModel  # Phase 2 primary architecture
