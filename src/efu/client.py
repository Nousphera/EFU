# Modified 2026-10-15: cache offline attacks and initial MAS/reference across unlearning updates.
import copy
import torch
import random
import torchmetrics
from efu.utils.apgd import apgd
from efu.utils.mas import get_weights_importance

def local_train(model, data, lr=3e-4, num_epochs=1, loss_fn=torch.nn.CrossEntropyLoss()):
	model.train()
	num_classes = len(model.label2class)
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
	acc, loss = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(model.device), torchmetrics.MeanMetric().to(model.device)
	for _ in range(num_epochs):
		acc.reset(); loss.reset()
		for (x, y) in data:
			x, y = x.to(model.device, non_blocking=True), torch.tensor([model.label2class[l.item()] for l in y], dtype=torch.long, device=model.device)
			optimizer.zero_grad()
			logits = model(x)
			_loss = loss_fn(logits, y).float()
			_loss.backward(); optimizer.step()
			loss.update(_loss.detach().item()); acc.update(logits, y)
	return model, len(data.dataset), {"loss": loss.compute().item(), "accuracy": acc.compute().item()}

def prepare_unlearning(model, data, eps=0.4, attack_steps=100):
    """Prepare Eq. (3)'s fixed reference, inverted MAS and offline attacks.

    Inputs must be raw [0,1] pixels/features; put any normalization inside the
    model so APGD's projection and the model operate in consistent coordinates.
    Reuse the returned preparation for every epoch/round of the same request.
    """
    adv_model = copy.deepcopy(model).eval()
    num_classes = len(model.label2class)
    if num_classes < 2:
        raise ValueError("Unlearning requires at least two output classes")
    batches = []
    for x, y in data:
        x = x.to(model.device)
        if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError("APGD expects [0,1] inputs; put normalization inside the model")
        y = torch.tensor([model.label2class[int(label)] for label in y],device=model.device)
        offsets = torch.randint(1,num_classes,y.shape,device=model.device)
        adv_y = (y + offsets) % num_classes
        adv_x = apgd(model=adv_model,inputs=x,labels=adv_y,eps=eps,norm=2,
                     targeted=True,n_iter=attack_steps,n_restarts=1,loss_function="ce")
        batches.append((x.detach(),y.detach(),adv_x.detach(),adv_y.detach()))
    if not batches:
        raise ValueError("Forget dataset must be nonempty")
    return {"batches": batches, "importance": get_weights_importance(model,data),
            "reference": {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}}


def local_unlearn(model, data, num_epochs=1, lr=1e-3, lambda_imp=1.0, eps=0.4,
                  prepared=None, attack_steps=100):
    """Optimize negative CE + offline targeted-adversarial CE + inverted MAS."""
    if prepared is None:
        prepared = prepare_unlearning(model,data,eps=eps,attack_steps=attack_steps)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-2)
    for _ in range(num_epochs):
        total_loss,correct,count = 0.0,0,0
        for x,y,adv_x,adv_y in prepared["batches"]:
            optimizer.zero_grad()
            logits = model(x)
            forget = -torch.nn.functional.cross_entropy(logits,y)
            adversarial = torch.nn.functional.cross_entropy(model(adv_x),adv_y)
            regularizer = sum((prepared["importance"][n]*(p-prepared["reference"][n]).pow(2)).sum()
                              for n,p in model.named_parameters() if n in prepared["importance"])
            objective = forget + adversarial + lambda_imp*regularizer
            objective.backward()
            optimizer.step()
            total_loss += float(objective.detach())*len(y)
            correct += int((logits.argmax(1)==y).sum())
            count += len(y)
    return model,len(data.dataset),{"loss":total_loss/count,"accuracy":correct/count}


def local_train_dp(model, data, lr=3e-4, num_epochs=10, loss_fn=torch.nn.CrossEntropyLoss(), delta=1e-5, max_grad=1.5, max_noise=1.0, verbose=False):
	import opacus
	import transformers
	model.train()
	# Training variables
	device = model.device
	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
	scheduler = transformers.get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=10, num_training_steps=len(data) * num_epochs)
	acc, loss, epsilon = torchmetrics.Accuracy(task="multiclass", num_classes=data.dataset.num_classes).to(device), \
		torchmetrics.MeanMetric().to(device), torchmetrics.MaxMetric().to(device)
	# Enable DP
	dp = opacus.PrivacyEngine(secure_mode=False)
	model, optimizer, data = dp.make_private(module=model, optimizer=optimizer,data_loader=data, max_grad_norm=max_grad, noise_multiplier=max_noise)
	# Train loop
	for e in range(num_epochs):
		acc.reset(); loss.reset(); # epsilon.reset() # reset metrics
		for _, (x,y) in enumerate(data):
			x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
			optimizer.zero_grad()
			logits = model(x)
			_loss = loss_fn(logits, y)
			_loss.backward(); optimizer.step(); scheduler.step() # Update model
			loss.update(_loss.detach().item()); acc.update(logits, y); epsilon.update(dp.get_epsilon(delta)) # Update metrics
		if verbose: print(f"Epoch {e+1}/{num_epochs} - Loss: {loss.compute().item():.4f}, Accuracy: {acc.compute().item():.2%}, (ε:{epsilon.compute().item():.2f}, δ:{delta})")
	return model._module, {'loss':loss.compute().item(), 'accuracy': acc.compute().item(), 'epsilon': epsilon.compute().item(), 'delta': delta}
