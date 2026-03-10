import torch
import torch.nn as nn

class PixelMLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=128, num_hidden_layers=2):
        super(PixelMLP, self).__init__()
        
        layers = []
        # First layer from input to hidden
        layers.append(nn.Linear(in_channels, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Final layer to output
        layers.append(nn.Linear(hidden_dim, out_channels))
        layers.append(nn.Sigmoid()) # bound [0,1]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        b, c, h, w = x.shape
        # Flatten spatial dimensions
        x = x.permute(0, 2, 3, 1).reshape(-1, c)
        # Apply MLP
        x = self.mlp(x)
        # Reshape back to image format
        x = x.view(b, h, w, -1).permute(0, 3, 1, 2)
        return x


if __name__ == "__main__":
    model = PixelMLP(in_channels=3, out_channels=10, hidden_dim=256, num_hidden_layers=4)
    input_tensor = torch.randn(8, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # (8, 10, 32, 32)

