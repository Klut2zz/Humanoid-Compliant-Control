#new change
"""Independent stochastic rollout/gradient audit; never updates an optimizer."""
from pathlib import Path


def diagnose(runner, wrapped, env, term, args, digest, frozen):
    import hashlib
    import json
    import numpy as np
    import torch
    out = Path(args.output)
    assert not out.exists(), out
    alg = runner.alg
    actor, critic = alg.actor, alg.critic
    #new change
    if args.calibrated_critic:
        candidate=torch.load(args.calibrated_critic,map_location=env.device,weights_only=False)
        assert candidate['artifact_type']=='critic_only_not_ppo_resume'
        assert Path(candidate['source_checkpoint']).resolve()==Path(args.init_checkpoint).resolve()
        critic.load_state_dict(candidate['critic_state_dict'],strict=True)
    params = list(actor.parameters())
    #new change
    parameter_names=[name for name,_ in actor.named_parameters()]
    mean_mask=torch.cat([torch.full((p.numel(),),not name.startswith('distribution.'),device=env.device,dtype=torch.bool)
                         for name,p in zip(parameter_names,params)])
    names = ('quiet', 'rise', 'hold', 'fall', 'recovery0_1', 'recovery1_2', 'recovery2_3', 'recovery3plus')
    def optimizer_hash():
        h = hashlib.sha256()
        def visit(x):
            if isinstance(x, torch.Tensor):
                h.update(x.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(x, dict):
                for k in sorted(x, key=str):
                    h.update(str(k).encode()); visit(x[k])
            elif isinstance(x, (list, tuple)):
                for v in x: visit(v)
            else: h.update(repr(x).encode())
        visit(alg.optimizer.state_dict())
        return h.hexdigest()
    before = [digest(m) for m in (actor, critic, *frozen)]
    opt_before = optimizer_hash()
    def forbidden(*a, **kw):
        raise RuntimeError('Optimizer updates forbidden in phase diagnosis')
    alg.optimizer.step = forbidden
    records, snapshots, rows, vectors = [], [], [], []
    last_force = torch.zeros(env.num_envs, device=env.device)
    age = torch.full_like(last_force, 100.)
    obs = wrapped.get_observations()
    for block in range(11):
        phases = []
        with torch.no_grad():
            for step in range(runner.cfg['num_steps_per_env']):
                c = env.command_manager.get_term('upper_body_joints')
                force = c.force_world.norm(dim=-1)
                age[(last_force > 1e-4) & (force <= 1e-4)] = 0
                active = force > 1e-4
                phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
                t, r, h = c._event_time, c._ramp_time, c._hold_time
                phase[active & (t < r)] = 1
                phase[active & (t >= r) & (t < r+h)] = 2
                phase[active & (t >= r+h)] = 3
                for lo, hi, label in ((0,1,4),(1,2,5),(2,3,6),(3,99,7)):
                    phase[(~active) & (age >= lo) & (age < hi)] = label
                phases.append(phase.clone())
                last_force.copy_(force)
                obs, reward, done, extra = wrapped.step(alg.act(obs))
                alg.process_env_step(obs, reward, done, extra)
                age += env.step_dt
                age[active | done.bool()] = 100.
                last_force[done.bool()] = 0
            alg.compute_returns(obs)
        st = alg.storage
        #new change
        cpu_rng=torch.get_rng_state()
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        phase = torch.stack(phases).flatten()
        observations = st.observations.flatten(0, 1)
        actions = st.actions.flatten(0, 1)
        old_logp = st.actions_log_prob.flatten()
        advantages = st.advantages.flatten()
        raw_adv = (st.returns-st.values).flatten()
        env_ids = torch.arange(env.num_envs, device=env.device).repeat(runner.cfg['num_steps_per_env'])
        records.append({k:v.detach().cpu().numpy().copy() for k,v in dict(
            phase=torch.stack(phases), advantage=st.advantages, raw_advantage=st.returns-st.values,
            value=st.values, returns=st.returns, reward=st.rewards, done=st.dones).items()})
        for label in range(len(names)):
            selected = (phase == label).nonzero().flatten()
            if not len(selected): continue
            shards = []
            total = torch.zeros(sum(p.numel() for p in params), device=env.device)
            ratio_clipped = 0
            # Environment-disjoint quarters reveal cancellation, without treating
            # sequential transitions from one environment as independent samples.
            for shard in range(4):
                ids = selected[env_ids[selected] % 4 == shard]
                if not len(ids): continue
                grad = torch.zeros_like(total)
                for chunk in ids.split(2048):
                    actor(observations[chunk], stochastic_output=True)
                    ratio = (actor.get_output_log_prob(actions[chunk])-old_logp[chunk]).exp()
                    ratio_clipped += int(((ratio-1).abs() > alg.clip_param).sum())
                    loss = torch.maximum(-advantages[chunk]*ratio,
                        -advantages[chunk]*ratio.clamp(1-alg.clip_param,1+alg.clip_param)).sum()/len(ids)
                    gs = torch.autograd.grad(loss, params, allow_unused=True)
                    grad += torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(params,gs)])
                assert torch.isfinite(grad).all()
                shards.append(grad)
                total += grad*(len(ids)/len(selected))
            norm = float(total.norm())
            cosines = [float(torch.nn.functional.cosine_similarity(a,b,dim=0))
                       for i,a in enumerate(shards) for b in shards[i+1:]]
            row = dict(block=block,phase=names[label],samples=len(selected),
                advantage_mean=float(advantages[selected].mean()),
                #new change: finite statistic for singleton diagnostic bins.
                advantage_std=float(advantages[selected].std(unbiased=False)),
                raw_advantage_mean=float(raw_adv[selected].mean()),
                positive_advantage_fraction=float((advantages[selected]>0).float().mean()),
                actor_gradient_norm=norm, hypothetical_clip_scale=min(1.,alg.max_grad_norm/(norm+1e-6)),
                shard_gradient_cosines=cosines,ratio_clip_fraction=ratio_clipped/len(selected))
            #new change
            row.update(mean_gradient_norm=float(total[mean_mask].norm()),
                std_gradient_norm=float(total[~mean_mask].norm()),
                mean_shard_cosines=[float(torch.nn.functional.cosine_similarity(a[mean_mask],b[mean_mask],dim=0))
                    for i,a in enumerate(shards) for b in shards[i+1:]])
            rows.append(row)
            vectors.append(total.detach().cpu().numpy())
        # Exact sample-weighted all-phase PPO actor gradient (one full batch).
        current=[(row,vec) for row,vec in zip(rows,vectors) if row['block']==block]
        full=sum(row['samples']*vec for row,vec in current)/len(phase)
        snapshots.append(dict(block=block,full_gradient_norm=float(np.linalg.norm(full)),
            hypothetical_clip_scale=min(1.,alg.max_grad_norm/(float(np.linalg.norm(full))+1e-6))))
        #new change: recompute the four frozen-parameter PPO minibatch gradients.
        # These are current diagnostic clipping values, not historical updates.
        minibatches=[]
        order=torch.randperm(len(phase),device=env.device)
        for ids in order.chunk(alg.num_mini_batches):
            grad=torch.zeros(sum(p.numel() for p in params),device=env.device)
            clipped=0
            for chunk in ids.split(2048):
                actor(observations[chunk],stochastic_output=True)
                ratio=(actor.get_output_log_prob(actions[chunk])-old_logp[chunk]).exp()
                clipped+=int(((ratio-1).abs()>alg.clip_param).sum())
                loss=torch.maximum(-advantages[chunk]*ratio,-advantages[chunk]*ratio.clamp(1-alg.clip_param,1+alg.clip_param)).sum()/len(ids)
                gs=torch.autograd.grad(loss,params,allow_unused=True)
                grad+=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(params,gs)])
            norm=float(grad.norm())
            minibatches.append(dict(norm=norm,clip_scale=min(1.,alg.max_grad_norm/(norm+1e-6)),
                mean_norm=float(grad[mean_mask].norm()),std_norm=float(grad[~mean_mask].norm()),ratio_clip_fraction=clipped/len(ids)))
        snapshots[-1]['minibatches']=minibatches
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
        alg.storage.clear()
        actor.zero_grad(set_to_none=True);critic.zero_grad(set_to_none=True)
        print('PHASE_AUDIT_BLOCK',block,'rows',len(current),'full_grad_norm',snapshots[-1]['full_gradient_norm'],flush=True)
    assert before == [digest(m) for m in (actor, critic, *frozen)]
    assert opt_before == optimizer_hash()
    assert all(p.grad is None for m in frozen for p in m.parameters())
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out/'samples.npz',**{k:np.concatenate([r[k] for r in records]) for k in records[0]},
        phase_names=np.asarray(names),dt=np.asarray(env.step_dt),seed=np.asarray(args.seed))
    np.savez_compressed(out/'gradients.npz',gradient=np.stack(vectors),
        parameter_names=np.asarray([n for n,p in actor.named_parameters()]),
        parameter_sizes=np.asarray([p.numel() for p in params]))
    (out/'diagnostics.json').write_text(json.dumps(dict(checkpoint=args.init_checkpoint,
        #new change
        calibrated_critic=args.calibrated_critic,
        environments=env.num_envs,steps=528,zero_force_fraction=args.zero_force_fraction,
        rows=rows,blocks=snapshots,optimizer_updates=0,weights_and_optimizer_unchanged=True,
        gradient_definition='normalized-advantage PPO actor loss; phase-conditional means; full batch diagnostic, not historical minibatch clipping',
        exploration='checkpoint stochastic distribution',hashes=before),indent=2))
    print('PHASE_DIAG_PASS weights_and_optimizer_unchanged=True optimizer_updates=0',flush=True)


if __name__ == '__main__':
    path=Path(__file__).with_name('train_dual540_recovery_window.py')
    wrapper=path.read_text()
    ns={'__file__':str(path)}
    exec(compile(wrapper[:wrapper.rindex('exec(compile(')],str(path),'exec'),ns)
    source=ns['source']
    #new change
    old="p.add_argument('--output',default=None)"
    assert source.count(old)==1
    source=source.replace(old,old+"\np.add_argument('--calibrated_critic',default=None)")
    start=source.index("        if args.mode=='gradient':")
    end=source.index('        assert args.output and not Path(args.output).exists()',start)
    source=source[:start]+"        if args.mode=='gradient':\n            diagnose(runner,wrapped,e,term,args,digest,frozen)\n            return\n"+source[end:]
    exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path),'diagnose':diagnose})
