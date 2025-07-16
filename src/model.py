import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

# Gradient Reversal Layer implementation
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)  # Identity forward pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)

# Variational Encoder (unchanged)
class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, embedding_dim)
        self.log_var = nn.Linear(hidden_dim, embedding_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.shared(x)
        log_mu = self.mu(h)
        log_var = self.log_var(h)
        return log_mu, log_var

# Spatial Decoder (same as before, you can copy your existing decoder here)
class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_genes, n_batches):
        super().__init__()
        self.batch_emb = nn.Embedding(n_batches, embedding_dim)
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU()
        )
        self.nb_mu = nn.Linear(hidden_dim, n_genes)
        self.log_theta = nn.Parameter(torch.zeros(1))
        self.hurdle_logits = nn.ModuleDict({
            "plin2": nn.Linear(hidden_dim, 1),
            "oil_red_o": nn.Linear(hidden_dim, 1),
            "lipid_droplet": nn.Linear(hidden_dim, 1)
        })
        self.hurdle_mu = nn.ModuleDict({
            "plin2": nn.Linear(hidden_dim, 1),
            "oil_red_o": nn.Linear(hidden_dim, 1),
            "lipid_droplet": nn.Linear(hidden_dim, 1)
        })
        self.hurdle_log_var = nn.ModuleDict({
            "plin2": nn.Linear(hidden_dim, 1),
            "oil_red_o": nn.Linear(hidden_dim, 1),
            "lipid_droplet": nn.Linear(hidden_dim, 1)
        })
        self.near_amyloid_logit = nn.Linear(hidden_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z, specimen_ids):
        batch_embedding = self.batch_emb(specimen_ids)
        h = self.shared(torch.cat([z, batch_embedding], dim=-1))
        log_mu_counts = self.nb_mu(h)
        log_theta = self.log_theta.expand_as(log_mu_counts)
        hurdle_out = {}
        for k in self.hurdle_logits.keys():
            hurdle_out[k] = {
                "logit_p": self.hurdle_logits[k](h),
                "mu": self.hurdle_mu[k](h),
                "log_var": self.hurdle_log_var[k](h)
            }
        near_amyloid_logit = self.near_amyloid_logit(h)
        return {
            "log_mu_counts": log_mu_counts,
            "log_theta": log_theta,
            "hurdle": hurdle_out,
            "near_amyloid_logit": near_amyloid_logit
        }

# Discriminator to predict specimen_ids from latent z
class Discriminator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_batches):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_batches)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)

# Full VAE with adversarial batch correction
class VAEWithAdversarial(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, n_genes, n_batches):
        super().__init__()
        self.encoder = VariationalEncoder(input_dim, hidden_dim, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim, n_genes, n_batches)
        self.discriminator = Discriminator(embedding_dim, hidden_dim // 2, n_batches)

    def reparameterize(self, log_mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return log_mu + eps * std

    def forward(self, input_data):
        x = input_data["log_normalized"]
        specimen_ids = input_data["specimen_ids"]

        log_mu, log_var = self.encoder(x)
        z = self.reparameterize(log_mu, log_var)

        outputs = self.decoder(z, specimen_ids)
        outputs["size_factors"] = input_data["size_factors"]
        return outputs, log_mu, log_var, z

    def discriminate(self, z, lambda_grl=1.0):
        # Apply gradient reversal on z before discriminator
        z_rev = grad_reverse(z, lambda_grl)
        logits = self.discriminator(z_rev)
        return logits

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

token_embeddings = extract_embeddings(vae, dataset)