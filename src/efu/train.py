"""Reproducible local simulation with decentralized coordinate-labelled FE.

This is a research execution harness. It is not a distributed deployment or a
reproduction of the compressed ciphertext cost in the paper.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

PROJECT = 'EFU'


class Tiny(torch.nn.Module):
    def __init__(self, classes=3, image=False):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((2,2)),torch.nn.Flatten(),torch.nn.Linear(12,classes)) if image else torch.nn.Linear(4,classes)
        self.label2class = {i:i for i in range(classes)}
    @property
    def device(self):
        return next(self.parameters()).device
    def forward(self,x):
        return self.body(x)


class ImageModel(torch.nn.Module):
    def __init__(self,name,classes,pretrained=False):
        super().__init__()
        import torchvision.models as models
        if PROJECT == 'EFU':
            if name == 'resnet18':
                self.body = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
                self.body.fc = torch.nn.Linear(self.body.fc.in_features,classes)
            elif name == 'vit_b_32':
                self.body = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT if pretrained else None)
                self.body.heads.head = torch.nn.Linear(self.body.heads.head.in_features,classes)
            else:
                raise ValueError('Model must be tiny, resnet18 or vit_b_32')
            self.input_mean = (.485,.456,.406)
            self.input_std = (.229,.224,.225)
        else:
            from network import Model
            self.body = Model(name,classes)
            self.input_mean = (.4914,.4822,.4465)
            self.input_std = (.2470,.2435,.2616)
        self.label2class = {i:i for i in range(classes)}
    @property
    def device(self):
        return next(self.parameters()).device
    def forward(self,x):
        mean=x.new_tensor(self.input_mean).view(1,3,1,1)
        std=x.new_tensor(self.input_std).view(1,3,1,1)
        return self.body((x-mean)/std)


def partition(targets,clients,alpha,seed):
    """Class-wise Dirichlet partitions; reject empty clients explicitly."""
    if clients < 2 or alpha <= 0:
        raise ValueError('Use at least two clients and positive Dirichlet alpha')
    rng=np.random.default_rng(seed)
    result=[[] for _ in range(clients)]
    labels=np.asarray(targets)
    for label in np.unique(labels):
        indices=np.flatnonzero(labels==label); rng.shuffle(indices)
        proportions=rng.dirichlet(np.repeat(alpha,clients))
        cuts=(np.cumsum(proportions)*len(indices)).astype(int)[:-1]
        for subset,indices in zip(result,np.split(indices,cuts)):
            subset.extend(indices.tolist())
    if any(not x for x in result):
        raise ValueError('Empty Dirichlet client; increase data/alpha or choose another seed')
    return result


def load_data(args):
    if args.dataset == 'synthetic':
        generator=torch.Generator().manual_seed(args.seed)
        x=torch.rand(args.clients*args.samples_per_client,4,generator=generator)
        y=(x[:,:3].argmax(1)).long()
        train=TensorDataset(x,y)
        test_x=torch.rand(48,4,generator=generator)
        test=TensorDataset(test_x,test_x[:,:3].argmax(1))
        indices=[list(range(i*args.samples_per_client,(i+1)*args.samples_per_client)) for i in range(args.clients)]
        return train,test,indices,y.numpy(),3
    from torchvision import datasets,transforms
    size=224 if args.model == 'vit_b_32' else 32
    transform=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor()])
    cls=datasets.CIFAR10 if args.dataset == 'cifar10' else datasets.CIFAR100
    train=cls(args.data_root,train=True,download=args.download,transform=transform)
    test=cls(args.data_root,train=False,download=args.download,transform=transform)
    return train,test,partition(train.targets,args.clients,args.alpha,args.seed),np.asarray(train.targets),len(train.classes)


def loader(dataset,indices,args,shuffle=True):
    return DataLoader(Subset(dataset,indices),batch_size=args.batch_size,shuffle=shuffle,num_workers=0)


def fit(model,data,epochs,lr):
    model.train(); optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    for _ in range(epochs):
        for x,y in data:
            x,y=x.to(model.device),y.to(model.device)
            optimizer.zero_grad(); loss=torch.nn.functional.cross_entropy(model(x),y)
            loss.backward(); optimizer.step()
    return model


def evaluate(model,data):
    model.eval(); correct,total,loss=0,0,0.0
    with torch.no_grad():
        for x,y in data:
            x,y=x.to(model.device),y.to(model.device)
            logits=model(x)
            correct+=int((logits.argmax(1)==y).sum()); total+=len(y)
            loss+=float(torch.nn.functional.cross_entropy(logits,y,reduction='sum'))
    return {'accuracy':correct/total,'loss':loss/total} if total else {'accuracy':None,'loss':None}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['synthetic','cifar10','cifar100'],default='synthetic')
    p.add_argument('--model',default='tiny')
    p.add_argument('--pretrained',action='store_true',help='Allow torchvision pretrained weight download (EFU image models)')
    p.add_argument('--clients',type=int,default=2)
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--samples-per-client',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--local-epochs',type=int,default=1)
    p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--alpha',type=float,default=.1)
    p.add_argument('--participation',type=float,default=1.)
    p.add_argument('--kappa',type=int,default=4)
    p.add_argument('--precision',type=int,default=2)
    p.add_argument('--max-abs',type=float,default=8.)
    p.add_argument('--seed',type=int,default=2025 if PROJECT=='EFU' else 42)
    p.add_argument('--device',default='cpu')
    p.add_argument('--data-root',default='data')
    p.add_argument('--download',action='store_true')
    p.add_argument('--output',default=None,help='Write configuration, partition indices and measured metrics as JSON')
    if PROJECT=='EFU':
        p.add_argument('--method',choices=['efu','fedavg','retrain'],default='efu')
        p.add_argument('--unlearn-method',choices=['efu','pgd','sga_ewc','fedosd'],default='efu')
        p.add_argument('--unlearn-start',type=int,default=2)
        p.add_argument('--unlearn-rounds',type=int,default=1)
        p.add_argument('--unlearn-epochs',type=int,default=1)
        p.add_argument('--forget-client',type=int,default=0)
        p.add_argument('--forget-fraction',type=float,default=.25)
        p.add_argument('--forget-class',type=int,default=None)
        p.add_argument('--attack-steps',type=int,default=2)
    else:
        p.add_argument('--method',choices=['enccluster','bfuse','cluster','fedavg'],default='enccluster')
        p.add_argument('--bpe',type=int,choices=[8,16],default=8)
    args=p.parse_args(argv)
    if args.clients<2 or args.rounds<1 or min(args.batch_size,args.local_epochs,args.samples_per_client,args.kappa)<1:
        p.error('Positive sizes/rounds and at least two clients are required')
    if not 0<args.participation<=1 or math.floor(args.clients*args.participation)<2:
        p.error('Participation must select at least two clients')
    if args.dataset=='synthetic' and args.model!='tiny': p.error('Synthetic features require --model tiny')
    if PROJECT=='EFU' and (not 0<=args.forget_client<args.clients or not 0<args.forget_fraction<1 or min(args.unlearn_start,args.unlearn_rounds,args.unlearn_epochs,args.attack_steps)<1):
        p.error('Invalid forget-client or forget-fraction')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    train,test,partitions,targets,classes=load_data(args)
    model=(Tiny(classes,args.dataset!='synthetic') if args.model=='tiny' else ImageModel(args.model,classes,args.pretrained)).to(args.device)
    test_loader=DataLoader(test,batch_size=args.batch_size,num_workers=0)
    forget=[];prepared={};retain=copy.deepcopy(partitions)
    if PROJECT=='EFU':
        original=partitions[args.forget_client]
        forget=([i for i in original if targets[i]==args.forget_class] if args.forget_class is not None
                else original[:max(1,int(len(original)*args.forget_fraction))])
        if not forget: p.error('The selected client has no samples to forget')
        retain[args.forget_client]=[i for i in original if i not in set(forget)]
    history=[];rng=np.random.default_rng(args.seed)
    for round_id in range(1,args.rounds+1):
        retained_only = PROJECT=='EFU' and (args.method=='retrain' or (args.method=='efu' and round_id>=args.unlearn_start+args.unlearn_rounds))
        active=[i for i in range(args.clients) if not retained_only or retain[i]]
        count=math.floor(len(active)*args.participation)
        if count<2: raise ValueError('At least two active participants must remain')
        selected=sorted(rng.choice(active,size=count,replace=False).tolist())
        states=[];sizes=[]
        for cid in selected:
            local=copy.deepcopy(model)
            unlearning=PROJECT=='EFU' and args.method=='efu' and cid==args.forget_client and args.unlearn_start<=round_id<args.unlearn_start+args.unlearn_rounds
            if unlearning:
                from efu.client import prepare_unlearning,local_unlearn
                data=loader(train,forget,args)
                if args.unlearn_method=='efu':
                    if cid not in prepared: prepared[cid]=prepare_unlearning(local,data,attack_steps=args.attack_steps)
                    local,_,_=local_unlearn(local,data,num_epochs=args.unlearn_epochs,lr=args.lr,prepared=prepared[cid])
                else:
                    from efu.utils.baselines import pgd_halimi,sga_ewc,fedosd
                    local,_,_={'pgd':pgd_halimi,'sga_ewc':sga_ewc,'fedosd':fedosd}[args.unlearn_method](local,data,num_epochs=args.unlearn_epochs,lr=args.lr)
                sizes.append(len(partitions[cid]))
            else:
                use_retain=PROJECT=='EFU' and (args.method=='retrain' or (args.method=='efu' and round_id>=args.unlearn_start+args.unlearn_rounds))
                indices=retain[cid] if use_retain else partitions[cid]
                if not indices: raise ValueError('An empty retained client cannot train; reduce participation/cohort explicitly')
                local=fit(local,loader(train,indices,args),args.local_epochs,args.lr)
                sizes.append(len(indices))
            states.append({k:v.detach().cpu().clone() for k,v in local.state_dict().items()})
        if PROJECT=='EFU':
            from efu.utils.fe import fe_secure_aggregate
            from efu.aggregate import fedavg_aggr
            updated=(fe_secure_aggregate(states,sizes,kappa=args.kappa,precision=args.precision,max_abs=args.max_abs,seed=args.seed)
                     if args.method=='efu' else fedavg_aggr(states,sizes,rnd=round_id)[0])
        else:
            from methods import METHODS
            updated=METHODS[args.method].aggregate(states,sizes,kappa=args.kappa,bpe=args.bpe,seed=args.seed,precision=args.precision,max_abs=args.max_abs)
        # Preserve integer counters and non-aggregated constants by name.
        full=model.state_dict()
        full.update({k:torch.as_tensor(v,dtype=full[k].dtype,device=args.device) for k,v in updated.items()})
        model.load_state_dict(full,strict=True)
        metrics={'round':round_id,'participants':selected,'sample_counts':sizes,**evaluate(model,test_loader)}
        if forget: metrics['forget_train_accuracy']=evaluate(model,loader(train,forget,args,False))['accuracy']
        history.append(metrics);print(json.dumps(metrics))
    result={'project':PROJECT,'configuration':vars(args),'partitions':partitions,'forget_indices':forget,'history':history,
            'protocol':'decentralized-coordinate-fe' if args.method in ('efu','enccluster') else 'plaintext-baseline',
            'benchmark_reproduction':False}
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    return result

def cli():
    main()

if __name__=='__main__': cli()
