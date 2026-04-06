"""
PyTorch model definitions for the accident classifier.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

ARCHITECTURE_PRESETS: dict[str, dict[str, tuple[int, ...] | int]] = {
    "baseline": {
        "branch_widths": (4, 4, 4, 1, 8, 2, 12, 12, 12, 12),
        "final_hidden": 8,
        "num_classes": 2,
    },
    "tuned": {
        "branch_widths": (40, 40, 40, 30, 120, 40, 60, 10, 10, 20),
        "final_hidden": 8,
        "num_classes": 2,
    },
}


class _DenseBranch(nn.Module):
    """
    Single-layer dense branch applying ``Linear`` followed by ``ReLU``.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
    ) -> None:
        """
        Args:
            in_features: Number of input features.
            hidden_features: Number of output features after the linear layer.
        """
        super().__init__()
        self.fc = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the linear layer followed by ReLU activation.

        Args:
            x: Input tensor of shape ``(batch_size, in_features)``.

        Returns:
            Activated tensor of shape ``(batch_size, hidden_features)``.
        """
        return self.act(self.fc(x))


class AccidentClassifier(nn.Module):
    """
    Feed-forward network that fuses per-feature branches for edge accident
    classification.

    Each branch projects one feature tensor through a dense layer; the
    activations are concatenated before a final hidden layer and a
    classification head.
    """

    def __init__(
        self,
        branch_input_dims: Sequence[int],
        branch_hidden_dims: Sequence[int],
        final_hidden: int = 8,
        num_classes: int = 2,
        branch_names: Sequence[str] | None = None,
    ) -> None:
        """
        Args:
            branch_input_dims: Input feature size for each branch.
            branch_hidden_dims: Hidden size for each branch; must match the
                length of ``branch_input_dims``.
            final_hidden: Width of the post-concatenation hidden layer.
            num_classes: Number of output classes.
            branch_names: Optional names for each branch; defaults to
                ``branch_0``, ``branch_1``, ...
        """
        super().__init__()

        if not branch_input_dims:
            raise ValueError("At least one branch is required.")
        if len(branch_input_dims) != len(branch_hidden_dims):
            raise ValueError(
                f"Mismatch between branch_input_dims ({len(branch_input_dims)}) "
                f"and branch_hidden_dims ({len(branch_hidden_dims)})."
            )

        if branch_names is None:
            branch_names = tuple(f"branch_{i}" for i in range(len(branch_input_dims)))
        else:
            branch_names = tuple(branch_names)
        if len(branch_names) != len(branch_input_dims):
            raise ValueError(
                "Number of branch names must match number of branch inputs."
            )

        self.branch_names = branch_names
        self.branch_input_dims = tuple(int(v) for v in branch_input_dims)
        self.branch_hidden_dims = tuple(int(v) for v in branch_hidden_dims)
        self.branches = nn.ModuleList(
            [
                _DenseBranch(in_f, h)
                for in_f, h in zip(self.branch_input_dims, self.branch_hidden_dims)
            ]
        )

        self.concatenated_features = sum(self.branch_hidden_dims)
        self.fc_hidden = nn.Linear(self.concatenated_features, int(final_hidden))
        self.act = nn.ReLU()
        self.fc_out = nn.Linear(int(final_hidden), int(num_classes))

    def forward(
        self,
        inputs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """
        Run all branches and combine their outputs into class logits.

        Args:
            inputs: One tensor per branch, in the order matching
                ``branch_input_dims``.

        Returns:
            Logits tensor of shape ``(batch_size, num_classes)``.
        """
        values = list(inputs)
        if len(values) != len(self.branches):
            raise ValueError(
                f"Expected {len(self.branches)} input tensors, got {len(values)}"
            )

        outputs = [branch(t) for branch, t in zip(self.branches, values)]
        x = torch.cat(outputs, dim=1)
        x = self.act(self.fc_hidden(x))
        return self.fc_out(x)


def parse_branch_widths(
    raw: str,
) -> tuple[int, ...]:
    """
    Parse a comma-separated string into a tuple of positive ints.

    Args:
        raw: Comma-separated list such as ``"16,32,32"``.

    Returns:
        Tuple of positive integer branch widths.
    """
    if not raw:
        raise ValueError("branch widths argument is empty")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected at least one branch width.")
    widths = tuple(int(p) for p in parts)
    if any(w <= 0 for w in widths):
        raise ValueError("All branch widths must be positive integers.")
    return widths


def build_branch_dims(
    profile: str = "baseline",
) -> tuple[int, int]:
    """
    Look up the final-hidden and output sizes for a known preset.

    Args:
        profile: Name of the preset in :data:`ARCHITECTURE_PRESETS`.

    Returns:
        Tuple ``(final_hidden, num_classes)`` for the selected preset.
    """
    if profile not in ARCHITECTURE_PRESETS:
        raise ValueError(f"Unknown profile '{profile}'.")
    preset = ARCHITECTURE_PRESETS[profile]
    return int(preset["final_hidden"]), int(preset["num_classes"])
