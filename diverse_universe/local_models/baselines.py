import torch
import torchvision


class MLP(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        activation_layer
    ):

        assert len(hidden_channels) > 0
        super().__init__()
        self.mlp = torchvision.ops.MLP(
            in_channels,
            hidden_channels,
            activation_layer=activation_layer
        )
        self.in_features = in_channels
        self.out_features = hidden_channels[-1]
        if self.out_features == 1:
            self.sigmoid = torch.nn.Sigmoid()
        else:
            self.sigmoid = None

    def forward(self, x):
        logits = self.mlp(x.view(x.shape[0], -1))
        if self.sigmoid is None:
            probs = logits
        else:
            probs = self.sigmoid(logits)
        return probs


def make_mlp(
    in_channels,
    hidden_channels,
    activation_layer=torch.nn.modules.activation.LeakyReLU
):
    return MLP(
        in_channels,
        hidden_channels,
        activation_layer
    )


def is_mlp(model):
    return isinstance(model, MLP)
