import torch
from torch import nn


class SlotAttention(nn.Module):
    """Slot Attention module."""

    def __init__(self,
                 num_iterations: int = 3,
                 num_slots: int = 6,
                 inputs_size: int = 64,
                 slot_size=64,
                 mlp_hidden_size=128,
                 epsilon=1e-8):
        """Builds the Slot Attention module.

        Args:
          num_iterations: Number of iterations.
          num_slots: Number of slots.
          slot_size: Dimensionality of slot feature vectors.
          mlp_hidden_size: Hidden layer size of MLP.
          epsilon: Offset for attention coefficients before normalization.
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.inputs_size = inputs_size

        self.norm_inputs = nn.LayerNorm(self.inputs_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)

        # Parameters for Gaussian init (shared by all slots).
        self.slots_mu = nn.Parameter(torch.zeros(1, 1, self.slot_size))
        torch.nn.init.xavier_uniform_(self.slots_mu)

        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, self.slot_size))
        torch.nn.init.xavier_uniform_(self.slots_log_sigma)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.inputs_size, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.inputs_size, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.slot_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.slot_size, self.slot_size)
        )

    def forward(self, inputs):
        batch_size, num_inputs, inputs_size = inputs.shape
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots = self.slots_mu + torch.exp(self.slots_log_sigma) \
                * torch.randn(inputs.shape[0], self.num_slots, self.slot_size).type_as(inputs)

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            q = q * self.slot_size ** -0.5  # Normalization.
            attn_logits = torch.einsum('bid,bsd->bis', k, q)
            attn = attn_logits.softmax(dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].

            # Weigted mean.
            attn = attn + self.epsilon
            attn = attn / attn.sum(axis=-1, keepdims=True)
            updates = torch.einsum('bis,bid->bsd', attn, v)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            u = updates.reshape(-1, self.slot_size)
            p = slots_prev.reshape(-1, self.slot_size)
            slots = self.gru(u, p)
            slots = slots.reshape(batch_size, -1, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


if __name__ == '__main__':
    slot_attention = SlotAttention()
    x = torch.randn((10, 16384, 64))

    out = slot_attention(x)
    print("Done")
