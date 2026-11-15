"""A native model repository without openassembler.toml or a factory function.

    openassembler compile examples/autodiscovered_model --backend modal --verify
    openassembler run examples/autodiscovered_model --backend modal

The compiler reads the model and example shape statically; CPU graph capture
and native code generation run on the selected worker. This example compiles
forward inference, not a host training program.
"""
import torch


class FeatureMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(8, 16)
        self.down = torch.nn.Linear(16, 4)

    def forward(self, x):
        return self.down(torch.relu(self.up(x)))


example_inputs = (torch.randn(17, 8),)
