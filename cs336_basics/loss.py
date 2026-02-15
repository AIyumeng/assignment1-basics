import torch


def cross_entropy(inputs, targets):
    # inputs: (B, C) logits
    # targets: (B,) class indices

    # log-sum-exp trick
    max_logits = inputs.max(dim=-1, keepdim=True).values
    stabilized = inputs - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(stabilized), dim=-1, keepdim=True))
    log_probs = stabilized - log_sum_exp  # log-softmax

    loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1))
    return loss.mean()


def perplexity(losses):
    return torch.exp(torch.mean(losses))
