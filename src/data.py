import torch

class SpatialSingleCellDataSet:
    
    def __init__(
        self, 
        counts,
        log_normalized,
        plin2_area,
        oil_red_o_area,
        lipid_droplet_area,
        near_amyloid,
        neighbor_indices,
        specimen_ids
    ):
        # Gene Expression Information
        self.counts = self._to_tensor(counts, torch.float)
        self.log_normalized = self._to_tensor(log_normalized, torch.float)

        self.size_factors = (self.counts.sum(axis=1) / self.counts.sum(axis=1).mean()).log()

        # Pathology Information
        self.plin2_area = self._to_tensor(plin2_area, torch.float)
        self.oil_red_o_area = self._to_tensor(oil_red_o_area, torch.float)
        self.lipid_droplet_area = self._to_tensor(lipid_droplet_area, torch.float)
        self.near_amyloid = self._to_tensor(near_amyloid, torch.float)

        # Neighborhood Information
        self.specimen_ids = self._to_tensor(specimen_ids, torch.long)
        self.neighbor_indices = self._to_tensor(neighbor_indices, torch.long)

    @staticmethod
    def _to_tensor(x, dtype=torch.float):
        if isinstance(x, torch.Tensor):
            return x.detach().clone().to(dtype)
        else:
            return torch.tensor(x, dtype=dtype)

    def __len__(self):
        return self.counts.size(0)

    def __getitem__(self, idx):
        return {
            # Expression Information
            "counts": self.counts[idx],
            "log_normalized": self.log_normalized[idx],
            "size_factors": self.size_factors[idx],

            # Pathology Information
            "plin2_area": self.plin2_area[idx],
            "oil_red_o_area": self.oil_red_o_area[idx],
            "lipid_droplet_area": self.lipid_droplet_area[idx],
            "near_amyloid": self.near_amyloid[idx],

            # Neighborhood Information
            "neighbor_indices": self.neighbor_indices[idx],

            # Cell Metadata
            "specimen_ids": self.specimen_ids[idx],
        }


