import os
import logging

def euclidean_distance(dict1, dict2, exclude_keys=["classifier", "model.base_model.model.classifier"]):
    return sum((dict1[k] - dict2[k]).pow(2).sum().item() for k in dict1.keys() if not any(excluded in k for excluded in exclude_keys))

def logger_fn(ckpt_dir=None):
	logger = logging.getLogger("FederatedTrainingLogger")
	if getattr(logger, "_already_configured", False):
		return logger
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter("%(message)s")
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.INFO)
	console_handler.setFormatter(formatter)
	logger.addHandler(console_handler)
	if ckpt_dir is not None:
		os.makedirs(ckpt_dir, exist_ok=True)
		log_file = os.path.join(ckpt_dir, "train_logs.log")
		file_handler = logging.FileHandler(log_file, mode='a')
		file_handler.setLevel(logging.INFO)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)
	logger.propagate = False
	logger._already_configured = True
	return logger
