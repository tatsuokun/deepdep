import torch


def create_random_mask(tensor):
    batch, size, _size = tensor.size()
    assert(size == _size)

    _, random_indices = \
        (torch.diag(-torch.ones(size)).expand(tensor.size()) + torch.rand(tensor.size())).topk(1)
    mask = torch.zeros(tensor.size()).scatter_(2, random_indices, 1).byte()
    return mask, random_indices.squeeze()


def calcurate_random_graph_cost(prob):
    batch_size, size, _ = prob.size()
    mask, indices = create_random_mask(prob)
    selected_path_cost = torch.stack(torch.masked_select(prob, mask).split(size))
    return torch.prod(selected_path_cost, dim=1), indices
