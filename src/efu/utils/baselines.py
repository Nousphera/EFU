import copy

import torch
import torch.nn.functional as F
import torchmetrics


def _map_labels(model, y):
    return torch.tensor([model.label2class[l.item()] for l in y], dtype=torch.long, device=model.device)


def _metrics(model):
    n = len(model.label2class)
    return (torchmetrics.Accuracy(task="multiclass", num_classes=n).to(model.device),
            torchmetrics.MeanMetric().to(model.device))


def _flat(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


def pgd_halimi(model, data, num_epochs=1, lr=2e-4, delta=3.0):
    model.train()
    acc, loss = _metrics(model)
    ref = _flat(model.parameters()).clone()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(num_epochs):
        acc.reset(); loss.reset()
        for x, y in data:
            x, y = x.to(model.device), _map_labels(model, y)
            opt.zero_grad()
            logits = model(x)
            (-F.cross_entropy(logits, y)).backward()         # gradient ascent
            opt.step()
            with torch.no_grad():                            # project onto L2 ball
                cur = _flat(model.parameters())
                diff = cur - ref; dist = diff.norm()
                if dist > delta:
                    cur = ref + diff * (delta / dist)
                    i = 0
                    for p in model.parameters():
                        n = p.numel(); p.copy_(cur[i:i + n].view_as(p)); i += n
            loss.update((-F.cross_entropy(logits, y)).detach().item()); acc.update(logits, y)
    return model, len(data.dataset), {"loss": loss.compute().item(), "accuracy": acc.compute().item()}


def sga_ewc(model, data, num_epochs=1, lr=2e-4, lambda_ewc=1.0):
    model.train()
    acc, loss = _metrics(model)
    ref = {n: p.detach().clone() for n, p in model.named_parameters()}
    # diagonal Fisher importance from the client's data
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    fmodel = copy.deepcopy(model)
    for x, y in data:
        x, y = x.to(model.device), _map_labels(model, y)
        fmodel.zero_grad()
        F.nll_loss(F.log_softmax(fmodel(x), dim=1), y).backward()
        for n, p in fmodel.named_parameters():
            if p.grad is not None and n in fisher:
                fisher[n] += p.grad.detach() ** 2
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(num_epochs):
        acc.reset(); loss.reset()
        for x, y in data:
            x, y = x.to(model.device), _map_labels(model, y)
            opt.zero_grad()
            logits = model(x)
            ascent = -F.cross_entropy(logits, y)
            ewc = sum((fisher[n] * (p - ref[n]).pow(2)).sum()
                      for n, p in model.named_parameters() if n in fisher)
            (ascent + lambda_ewc * ewc).backward()
            opt.step()
            loss.update(ascent.detach().item()); acc.update(logits, y)
    return model, len(data.dataset), {"loss": loss.compute().item(), "accuracy": acc.compute().item()}


def fedosd(model, data, num_epochs=1, lr=2e-4):
    model.train()
    acc, loss = _metrics(model)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(num_epochs):
        acc.reset(); loss.reset()
        # average data gradient (the direction unlearning must not fight)
        g = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        for x, y in data:
            x, y = x.to(model.device), _map_labels(model, y)
            model.zero_grad()
            F.cross_entropy(model(x), y).backward()
            for n, p in model.named_parameters():
                if p.grad is not None and n in g:
                    g[n] += p.grad.detach()
        for x, y in data:
            x, y = x.to(model.device), _map_labels(model, y)
            opt.zero_grad()
            logits = model(x)
            p_y = F.softmax(logits, dim=1).gather(1, y.view(-1, 1)).squeeze(1).clamp(max=1 - 1e-6)
            (-torch.log(1 - p_y).mean()).backward()          # bounded unlearning CE
            with torch.no_grad():                            # remove conflicting component
                dot = sum((p.grad * g[n]).sum() for n, p in model.named_parameters()
                          if p.grad is not None and n in g)
                nrm = sum((g[n] ** 2).sum() for n in g).clamp(min=1e-12)
                if dot < 0:
                    for n, p in model.named_parameters():
                        if p.grad is not None and n in g:
                            p.grad -= (dot / nrm) * g[n]
            opt.step()
            loss.update((-torch.log(1 - p_y).mean()).detach().item()); acc.update(logits, y)
    return model, len(data.dataset), {"loss": loss.compute().item(), "accuracy": acc.compute().item()}
