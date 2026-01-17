import torch
import torchmetrics


class Entropy(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state('indices', default=[], dist_reduce_fx='cat')
        self.add_state('counts', default=[], dist_reduce_fx='cat')

    def update(self, indices):
        unique_indices, counts = torch.unique(indices, return_counts=True)
        self.indices.append(unique_indices)
        self.counts.append(counts)

    def compute(self):
        sparse_counts = torch.sparse_coo_tensor(self.indices[None, :], self.counts)
        sparse_counts = sparse_counts.coalesce()
        p_sparse = sparse_counts / sparse_counts.sum()
        p_masked = torch.masked.masked_tensor(
            p_sparse, p_sparse.to(torch.bool)
        )  # mask out zero entries because log is not defined & implemented
        entropy = -(p_masked * torch.log2(p_masked)).sum()
        entropy = entropy.get_data()  # unmask
        return entropy


class Perplexity(Entropy):

    def compute(self):
        return 2 ** super().compute()
