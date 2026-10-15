# Modified 2026-10-15: route encrypted aggregation to the coordinate-labelled backend.
import torch
import copy
import numpy as np

from efu.utils.fe import fe_secure_aggregate

def fedavg_aggr(dicts, num_samples, rnd, eta=0.001, model=None, **kwargs):
    assert len(dicts) == len(num_samples), "Mismatch: dicts and num_samples must have the same length"

    total_samples = sum(num_samples)
    weights = [ns / total_samples for ns in num_samples]

    aggregated_dict = {}
    for k, v in dicts[0].items():
        if "num_batches_tracked" in k:
            continue  # Skip num_batches_tracked, handled separately
        elif "running_mean" in k or "running_var" in k:
            aggregated_dict[k] = torch.zeros_like(v, dtype=torch.float32, device="cpu")  # Float32 for batch stats
        else:
            aggregated_dict[k] = torch.zeros_like(v, dtype=torch.float32, device="cpu")  # Regular model weights

    # Weighted sum for model weights
    for client_dict, weight in zip(dicts, weights):
        for k in aggregated_dict.keys():
            if "running_mean" in k or "running_var" in k:
                aggregated_dict[k] += client_dict[k] / len(dicts)  # Arithmetic mean for BatchNorm stats
            else:
                aggregated_dict[k] += client_dict[k] * weight  # Weighted sum for model weights

    # Handle num_batches_tracked separately (take max across clients)
    for k in dicts[0].keys():
        if "num_batches_tracked" in k:
            aggregated_dict[k] = max(client_dict[k] for client_dict in dicts)

    next_rnd_clients_lr = eta / np.sqrt(rnd)

    # Clone aggregated weights
    aggregated_dict = {k: v.detach().clone() for k, v in aggregated_dict.items()}

    # Update model weights if provided
    if model is not None:
        model.set_weights(copy.deepcopy(aggregated_dict))

    return aggregated_dict, None, None, next_rnd_clients_lr

def unweighted_aggr(dicts, num_samples, rnd, eta=0.001, model=None, **kwargs):
	num_clients = len(dicts)
	aggregated_dict = {k: torch.zeros_like(v, dtype=torch.float32, device="cpu")  for k, v in dicts[0].items() if "num_batches_tracked" not in k}
	for client_dict in dicts:
		for k in aggregated_dict.keys():
			aggregated_dict[k] += client_dict[k]
	for k in aggregated_dict.keys():
		aggregated_dict[k] /= num_clients
	next_rnd_clients_lr = eta / np.sqrt(rnd)
	aggregated_dict = {k: v.detach().clone() for k, v in aggregated_dict.items()}
	if model is not None: model.set_weights(copy.deepcopy(aggregated_dict))
	return aggregated_dict, None, None, next_rnd_clients_lr

def fedadam_aggr(dicts, num_samples, rnd, m_t=None, v_t=None, beta_1=0.9, beta_2=0.999, eta=0.001, tau=1e-8, model=None, **kwargs):
	assert len(dicts) == len(num_samples), "Mismatch: dicts and num_samples must have the same length"
	total_samples = sum(num_samples)
	weights = [ns / total_samples for ns in num_samples]
	aggregated_dict = {k: torch.zeros_like(v, dtype=torch.float32, device="cpu")  for k, v in dicts[0].items() if "num_batches_tracked" not in k}
	for client_dict, weight in zip(dicts, weights):
		for k in aggregated_dict.keys():
			aggregated_dict[k] += client_dict[k] * weight
	fedavg_weights = {k: v.detach().numpy() for k, v in aggregated_dict.items()}
	if model is not None:
		current_weights = {k: v.cpu().detach().numpy() for k, v in model.state_dict().items()}
	else:
		current_weights = {k: np.zeros_like(v) for k, v in fedavg_weights.items()}  # If no model, assume zeros
	delta_t = {k: fedavg_weights[k] - current_weights[k] for k in fedavg_weights.keys()}
	if m_t is None:
		m_t = {k: np.zeros_like(v) for k, v in delta_t.items()}
	if v_t is None:
		v_t = {k: np.zeros_like(v) for k, v in delta_t.items()}
	m_t = {k: beta_1 * m_t[k] + (1 - beta_1) * delta_t[k] for k in delta_t.keys()}
	v_t = {k: beta_2 * v_t[k] + (1 - beta_2) * np.multiply(delta_t[k], delta_t[k]) for k in delta_t.keys()}
	new_weights = {k: current_weights[k] + eta * m_t[k] / (np.sqrt(v_t[k]) + tau) for k in delta_t.keys()}

	updated_model_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in new_weights.items()}
	next_rnd_clients_lr = eta / np.sqrt(rnd)
	if model is not None: model.set_weights(copy.deepcopy(updated_model_dict))
	return updated_model_dict, m_t, v_t, next_rnd_clients_lr

def fedyogi_aggr(dicts, num_samples, rnd, m_t=None, v_t=None, beta_1=0.9, beta_2=0.999, eta=0.001, tau=1e-8, model=None, **kwargs):
	assert len(dicts) == len(num_samples), "Mismatch: dicts and num_samples must have the same length"
	total_samples = sum(num_samples)
	weights = [ns / total_samples for ns in num_samples]
	aggregated_dict = {k: torch.zeros_like(v, dtype=torch.float32, device="cpu")  for k, v in dicts[0].items() if "num_batches_tracked" not in k}
	for client_dict, weight in zip(dicts, weights):
		for k in aggregated_dict.keys():
			aggregated_dict[k] += client_dict[k] * weight
	fedavg_weights = {k: v.detach().numpy() for k, v in aggregated_dict.items()}
	if model is not None:
		current_weights = {k: v.cpu().detach().numpy() for k, v in model.state_dict().items()}
	else:
		current_weights = {k: np.zeros_like(v) for k, v in fedavg_weights.items()}
	delta_t = {k: fedavg_weights[k] - current_weights[k] for k in fedavg_weights.keys()}
	if m_t is None:
		m_t = {k: np.zeros_like(v) for k, v in delta_t.items()}
	if v_t is None:
		v_t = {k: np.zeros_like(v) for k, v in delta_t.items()}
	m_t = {k: beta_1 * m_t[k] + (1 - beta_1) * delta_t[k] for k in delta_t.keys()}
	v_t = {k: v_t[k] - (1.0 - beta_2) * np.multiply(delta_t[k], delta_t[k]) * np.sign(v_t[k] - np.multiply(delta_t[k], delta_t[k])) for k in delta_t.keys()}
	new_weights = {k: current_weights[k] + eta * m_t[k] / (np.sqrt(v_t[k]) + tau) for k in delta_t.keys()}
	next_rnd_clients_lr = eta / np.sqrt(rnd)
	updated_model_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in new_weights.items()}
	if model is not None: model.set_weights(copy.deepcopy(updated_model_dict))
	return updated_model_dict, m_t, v_t, next_rnd_clients_lr

def fe_aggr(dicts, num_samples, rnd, eta=0.001, model=None, kappa=64, tag=b"efu", **kwargs):
	"""EFU secure aggregation: weighted FedAvg computed under functional encryption.

	Drop-in replacement for fedavg_aggr. Each client's weights are clustered, expanded and
	encrypted using the coordinate-labelled backend, with one ciphertext per model
	coordinate. Returns the same 4-tuple as the other aggregators:
	(aggregated_dict, m_t, v_t, next_round_lr).
	"""
	aggregated_dict = fe_secure_aggregate(dicts, num_samples, kappa=kappa, tag=tag)
	next_rnd_clients_lr = eta / np.sqrt(rnd)
	if model is not None:
		model.set_weights(copy.deepcopy(aggregated_dict))
	return aggregated_dict, None, None, next_rnd_clients_lr
