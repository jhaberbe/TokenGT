import torch
from torch.utils.data import DataLoader

def extract_embeddings(model, dataset, batch_size=256, use_mean=True, device=None):
    """
    Extract latent embeddings for all samples in `dataset`.

    Args:
        model: trained VAE model with encoder
        dataset: dataset object (e.g. SpatialSingleCellDataSet)
        batch_size: batch size for DataLoader
        use_mean: if True, use encoder's mean (log_mu) as embedding,
                  else sample from latent distribution
        device: torch device (e.g. 'cuda' or 'cpu'), default auto-detect

    Returns:
        embeddings: Tensor of shape (n_samples, embedding_dim)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

    all_embeddings = []

    with torch.no_grad():
        for batch_samples in loader:
            batch_data = {key: torch.stack([sample[key] for sample in batch_samples]).to(device) for key in batch_samples[0].keys()}

            log_mu, log_var = model.encoder(batch_data["log_normalized"])

            if use_mean:
                embeddings = log_mu
            else:
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                embeddings = log_mu + eps * std

            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)
