import torch

class L_WSC(torch.nn.Module):
    """
    weighted structure-aware contrastive loss for contrastive learning.
    please find more details at https://doi.org/10.1016/j.media.2025.103553
    """

    def __init__(self, bit, Gweight, device):
        """
        Args:
            bit (int): Length of the hash code.
            config (dict): Configuration dictionary.
            Gweight (tensor): Global weights for each class.
            device (torch.device): Device on which tensors are allocated.
        """
        super(L_WSC, self).__init__()
        self.bit = bit
        self.device = device
        self.global_weights = Gweight.to(device)

    def cal_distance(self, hi, hj):
        """
        Compute distance between hash codes based on cosine similarity transformation.

        Args:
            hi (tensor): Hash code tensor of shape (N, bit).
            hj (tensor): Hash code tensor of shape (M, bit).
            Normally, N==M

        Returns:
            tensor: Distance values in the range (0, 1).
        """
        inner_product = (hi-hi.mean(dim=-1,keepdims=True)) @ (hj-hj.mean(dim=-1,keepdims=True)).t()
        norm = ((hi-hi.mean(dim=-1,keepdims=True)).pow(2).sum(dim=1, keepdim=True)+1e-6).pow(0.5) @ \
                ((hj-hj.mean(dim=-1,keepdims=True)).pow(2).sum(dim=1, keepdim=True)+1e-6).pow(0.5).t()
        pr = inner_product / (norm + 1e-6)
        return (1 - (pr+1)/2)


    def compute_similarity_matrix(self, images):
        N = images.shape[0]
        images_flat = images.view(N, -1)
        similarity_matrix = torch.nn.functional.cosine_similarity(images_flat.unsqueeze(1),
                                                                  images_flat.unsqueeze(0), dim=-1)
        return similarity_matrix


    def forward(self, u, label, img_fp):
        """
        Args:
            u (tensor): Hash vectors of shape (batch, bit).
            label (tensor): Ground truth labels of shape (batch,).
            img_fp (tensor): image fingerprints of the input data
        Returns:
            tuple: (loss, cauchy_loss, quantization_loss)
        """
        SimMatrix = self.compute_similarity_matrix(img_fp)

        batch, _ = u.shape
        one = torch.ones((batch, self.bit)).to(self.device)
        y = (label.unsqueeze(0) == label.unsqueeze(1)).float()

        one_raw = torch.ones((batch)).to(self.device)

        if (1 - y).sum() != 0 and y.sum() != 0:
            sw = one_raw * self.global_weights[label]
            w = (sw.unsqueeze(1) @ sw.unsqueeze(0))
        else:
            w = 1

        d_hi_hj = self.d(u, u).clamp(min=1e-6, max=1 - 1e-6)
        contrast_loss = (w * (SimMatrix * y * -torch.log(1 - d_hi_hj) - torch.exp(SimMatrix) * (1 - y) * torch.log(
            d_hi_hj))).mean()

        quantization_loss = torch.log(1 + self.d(u.abs(), one)).mean()

        loss = contrast_loss + 0.5 * quantization_loss
        return loss, contrast_loss, quantization_loss