#new change
"""Read-only decomposition of frozen exported upper controller."""
import torch
import json

def install(env):
    term=env.action_manager.get_term('joint_pos')
    m=term.upper
    original=term.process_actions
    rows=[]
    count=0
    def process(actions):
        nonlocal count
        original(actions)
        with torch.no_grad():
            obs=env.obs_buf
            base=m.mlp(m.obs_normalizer(obs['legacy']))
            parts=[]
            for side in ('left','right'):
                f=obs[side+'_arm_adapter']
                net=getattr(m,side+'_arm_residual_adapter')
                raw=torch.cat([net.shoulder(f),net.elbow(f),net.wrist(f)],-1)
                limit=getattr(m,side+'_arm_adapter_limits')
                learned=torch.tanh(raw)*limit
                feedback=(-m.analytic_error_gain*f[:,14:21]/.15).clamp(-limit,limit)
                projection=getattr(m,side+'_arm_adapter_projection')
                parts.append((learned@projection,feedback@projection))
            total=m.deterministic_output(base+parts[0][0]+parts[0][1]+parts[1][0]+parts[1][1])
            from neutral_dual_adapt_cfg import ARM
            torch.testing.assert_close(total[:,ARM],env._dual_arm_raw,rtol=1e-5,atol=1e-6)
            if 50<=count<900:
                j=[24,26,28]
                rows.append(torch.stack([base[:,j],parts[1][0][:,j],parts[1][1][:,j],total[:,j]],-1).cpu())
        count+=1
    term.process_actions=process
    def report():
        values=torch.stack(rows)*(.15*180/torch.pi)
        names=['right_wrist_roll','right_wrist_pitch','right_wrist_yaw']
        labels=['actor','supervised_adapter','analytic_feedback','total_target_minus_ref']
        result={n:{label:{'mean_deg':float(values[:,:,i,k].mean()),'std_deg':float(values[:,:,i,k].std())} for k,label in enumerate(labels)} for i,n in enumerate(names)}
        print('TARGET_COMPONENTS',json.dumps(result),flush=True)
        print('OUTPUT_RECOMPOSITION_PASS',count,flush=True)
    env._report_arm_components=report
