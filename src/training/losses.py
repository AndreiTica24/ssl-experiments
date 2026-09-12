"""NT-Xent loss (Normalized Temperature-scaled Cross Entropy).

The core loss function of SimCLR. Given a batch of N images, each with 2
augmented views, we get 2N projections. For each projection, exactly ONE
other projection in the batch is a "positive" (the other view of the same
image); the remaining 2N-2 are "negatives".

The loss pulls positives close and pushes negatives away, in a normalized
embedding space (unit sphere), with a temperature parameter that controls
how "sharp" the similarity comparisons are.

Reference: Chen et al., "A Simple Framework for Contrastive Learning of
Visual Representations" (SimCLR), ICML 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """NT-Xent (info-NCE style) loss for SimCLR.

    Args:
        temperature: scaling factor for similarity scores. Lower values
            make the loss more sensitive to hard negatives. Typical: 0.1 - 0.5.

    Forward inputs:
        z1: projections of view 1, shape (N, D).
        z2: projections of view 2, shape (N, D).

    Both z1 and z2 come from the same N images; z1[i] and z2[i] are the
    positive pair for image i.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.shape[0]
        device = z1.device

        # 1. L2-normalize projections onto the unit hypersphere.
        #    After this, dot product == cosine similarity.
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # 2. Concatenate both views: rows 0..N-1 = view1, rows N..2N-1 = view2.
        z = torch.cat([z1, z2], dim=0)  # shape (2N, D)

        # 3. Compute all pairwise cosine similarities, scaled by temperature.
        #    similarity_matrix[i, j] = cos(z[i], z[j]) / temperature
        similarity_matrix = torch.mm(z, z.T) / self.temperature  # shape (2N, 2N)

        # 4. Mask out self-similarities (diagonal): a sample is not its own positive.
        #    Set diagonal to -inf so it contributes zero after softmax.
        mask_self = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        similarity_matrix.masked_fill_(mask_self, float("-inf"))

        # 5. Build positive-pair targets.
        #    For row i in [0, N), its positive is at column i + N (the other view).
        #    For row i in [N, 2N), its positive is at column i - N.
        targets = torch.arange(2 * batch_size, device=device)
        targets = (targets + batch_size) % (2 * batch_size)

        # 6. Cross-entropy over the similarity rows:
        #    for each row, we want the softmax mass concentrated on `targets[row]`.
        loss = F.cross_entropy(similarity_matrix, targets)
        return loss