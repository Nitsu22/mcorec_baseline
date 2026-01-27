import torch

EPS = 1e-6

def cal_SISNR(source, estimate_source):
    """Calculate scale-invariant SNR.
    Args:
        source: torch.Tensor, [B, T]
        estimate_source: torch.Tensor, [B, T]
    Returns:
        sisnr: torch.Tensor, [B]
    """
    assert source.size() == estimate_source.size()

    source = source - torch.mean(source, axis=-1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis=-1, keepdim=True)

    ref_energy = torch.sum(source ** 2, axis=-1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis=-1, keepdim=True) * source / ref_energy
    noise = estimate_source - proj
    ratio = torch.sum(proj ** 2, axis=-1) / (torch.sum(noise ** 2, axis=-1) + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)

    return sisnr
