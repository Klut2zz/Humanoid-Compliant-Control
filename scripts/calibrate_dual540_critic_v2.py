#new change
"""Offline critic-only terminal-balanced regression. No simulator/PPO updates."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--critic',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--seed',type=int,default=42)
    #new change
    p.add_argument('--terminal_fraction',type=float,default=.05)
    a=p.parse_args();out=Path(a.output);assert not out.exists()
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    source=torch.load(a.critic,map_location='cpu',weights_only=False)
    assert source['artifact_type']=='critic_only_not_ppo_resume'
    actor_file=Path(source['source_checkpoint']).resolve()
    manifest=json.loads((actor_file.parent/'base_controllers.json').read_text())
    protected=[actor_file,Path(a.critic),*[Path(k) for k in manifest['sha256']]]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    f=np.load(a.data);r=f['rewards'].astype(np.float64);done=f['dones'];phase=f['phase'];health=f['health'];tr=f['train_env']
    target=f['target'].copy();x=f['observations'][:len(target)].copy();n,E=target.shape;T=len(r)
    distance=np.full((T,E),100000,dtype=np.int32);mc=np.zeros((T,E),dtype=np.float64)
    for t in range(T-1,-1,-1):
        distance[t]=np.where(done[t],0,1+distance[t+1] if t+1<T else 100000)
        mc[t]=r[t]+np.where(done[t],0,.99*mc[t+1] if t+1<T else 0)
    near=distance[:n]<50
    assert np.max(abs(target[near]-mc[:n][near]))<1e-5
    target[near]=mc[:n][near]  # Explicit direct terminal return, no bootstrap.
    # All fitting pools are restricted to the original training environments.
    recovered=np.zeros((n,E),bool)
    for t,e in zip(*np.where(phase[:n]==4)):
        if done[t:t+150,e].any():continue
        if np.isin(phase[t:t+150,e],[1,2,3]).any():continue
        q=health[t+150,e]
        recovered[t,e]=(q[0]<.05 and q[1]<.03 and q[2]<np.deg2rad(5) and q[3]<.02)
    model=nn.Sequential(nn.Linear(x.shape[-1],128),nn.ELU(),nn.Linear(128,64),nn.ELU(),nn.Linear(64,1)).to(a.device)
    state={k.removeprefix('mlp.'):v for k,v in source['critic_state_dict'].items()}
    model.load_state_dict(state,strict=True)
    assert len(state)==6
    X=torch.from_numpy(x.reshape(-1,x.shape[-1])).to(a.device)
    Y=torch.from_numpy(target.reshape(-1)).float().to(a.device)
    train_mask=np.broadcast_to(tr,(n,E));val_mask=~train_mask
    masks={'general':np.ones((n,E),bool),'recovered':recovered,
           'last_half_second':distance[:n]<25,'preceding_half_second':(distance[:n]>=25)&(distance[:n]<50),
           'terminal':distance[:n]==0}
    proportions={'general':.50,'recovered':.20,'last_half_second':.15,'preceding_half_second':.10,'terminal':.05}
    #new change
    assert .05<=a.terminal_fraction<=.20
    proportions['terminal']=a.terminal_fraction
    proportions['general']=.55-a.terminal_fraction
    pools={k:torch.as_tensor(np.flatnonzero(m&train_mask),device=a.device) for k,m in masks.items()}
    assert all(len(v)>0 for v in pools.values())
    valid=torch.as_tensor(np.flatnonzero(val_mask),device=a.device)
    def predictions():
        with torch.no_grad():return torch.cat([model(X[ix]).squeeze(-1).cpu() for ix in valid.split(8192)]).numpy().astype(float)
    y=target[val_mask]
    def metrics(pred):
        result={}
        for k,m in masks.items():
            s=m[val_mask];err=pred[s]-y[s]
            result[k]=dict(n=int(s.sum()),rmse=float(np.sqrt(np.mean(err**2))),bias=float(err.mean()),
                prediction_mean=float(pred[s].mean()),target_mean=float(y[s].mean()))
        return result
    initial=metrics(predictions());print('V2_INITIAL',json.dumps(initial),flush=True)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    history=[];best=None;best_score=float('inf');best_epoch=0;best_metrics=None
    # Fixed validation gates. They are not used to fit any minibatch.
    def acceptable(m):
        return (m['general']['rmse']<=initial['general']['rmse']*1.10 and
            m['recovered']['rmse']<=initial['recovered']['rmse']*1.10 and
            all(m[k]['rmse']<=initial[k]['rmse']*.5 and abs(m[k]['bias'])<=.25
                for k in ('last_half_second','preceding_half_second','terminal')))
    for epoch in range(1,a.epochs+1):
        for _ in range(50):
            opt.zero_grad(set_to_none=True);loss=0
            for name,pool in pools.items():
                ix=pool[torch.randint(len(pool),(1024,),device=a.device)]
                error=model(X[ix]).squeeze(-1)-Y[ix]
                loss=loss+proportions[name]*error.square().mean()
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
        pred=predictions();m=metrics(pred);passed=acceptable(m)
        score=sum(proportions[k]*m[k]['rmse']**2 for k in proportions)
        if passed and score<best_score:
            best_score=score;best=copy.deepcopy(model.state_dict());best_epoch=epoch;best_metrics=m
        history.append(dict(epoch=epoch,passed=passed,metrics=m));print('V2_EPOCH',epoch,'passed',passed,json.dumps(m),flush=True)
    if best is None:
        best=copy.deepcopy(model.state_dict());best_epoch=a.epochs;best_metrics=m
    model.load_state_dict(best);final_pred=predictions()
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    out.mkdir(parents=True,exist_ok=False)
    artifact=dict(critic_state_dict={'mlp.'+k:v.detach().cpu() for k,v in best.items()},
        source_checkpoint=source['source_checkpoint'],artifact_type='critic_only_not_ppo_resume',
        calibration_parent=str(Path(a.critic).resolve()),epoch=best_epoch,validation_pass=acceptable(best_metrics))
    torch.save(artifact,out/'critic_only.pt')
    np.savez_compressed(out/'validation.npz',prediction=final_pred,target=y,
        env_ids=np.broadcast_to(np.arange(E),(n,E))[val_mask],distance_to_failure=distance[:n][val_mask],
        recovered=recovered[val_mask])
    summary=dict(initial=initial,selected=best_metrics,selected_epoch=best_epoch,validation_pass=acceptable(best_metrics),
        history=history,pool_proportions=proportions,train_pool_counts={k:len(v) for k,v in pools.items()},
        source_checkpoint=source['source_checkpoint'],calibration_parent=str(Path(a.critic).resolve()),
        original_models_unchanged=True,protected_sha256=hashes,actor_updates=0,ppo_updates=0,
        train_environment_ids=np.flatnonzero(tr).tolist(),validation_environment_ids=np.flatnonzero(~tr).tolist(),
        learning_rate=1e-4,validation_gate='overall/recovered RMSE <=110% V1; each terminal stratum RMSE <=50% V1 and abs bias <=0.25')
    (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    print('V2_FINISHED',json.dumps({'validation_pass':summary['validation_pass'],'selected_epoch':best_epoch,'original_models_unchanged':True}),flush=True)


if __name__=='__main__':main()
