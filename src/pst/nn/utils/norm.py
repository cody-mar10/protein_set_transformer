from torch import Tensor
from torch_geometric.nn import GraphNorm


class NormMixin:
    def normalize(self, x: Tensor, layer: GraphNorm, batch: Tensor) -> Tensor:
        """Ensure that graph normalization always has a batch Tensor supplied, which
        normalizes inputs per graph. Without this, the default behavior is to normalize
        as if the input is a single graph.

        Args:
            x (Tensor): node feature tensor of shape (num_nodes, in_channels)
            layer (GraphNorm): graph normalization layer
            batch (Tensor): idx tensor that specifies which graph each node belongs to
                of shape (num_nodes,)

        Returns:
            Tensor: normalized inputs
        """
        return layer(x, batch)
