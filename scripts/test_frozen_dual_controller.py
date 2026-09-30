#new change
"""Independent disjoint-output dual-controller inference. Never trains."""
import argparse,sys,json
from pathlib import Path
ROOT=Path('/home/noah/mingkai/general_softmimic')
sys.path.insert(0,str(ROOT/'softmimic_gym'))
sys.path.insert(0,str(ROOT/'scripts'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['baseline','shadow','connected'],required=True)
p.add_argument('--scale',type=float,default=0.)
p.add_argument('--steps',type=int,default=500)
p.add_argument('--num_envs',type=int,default=32)
p.add_argument('--seed',type=int,default=0)
p.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args();app=AppLauncher(args).app
import copy,numpy as np,torch,gymnasium as gym
import softmimic_gym.tasks
from neutral_stand_geometry_cfg import NeutralStandGeometryCfg,LegAction,LegActionCfg,state,LEG_IDS,UPPER_IDS,NAMES
from softmimic_gym.tasks.locomotion.tracking.config.g1.general_tracking_v1 import G1StandCoordinationAccuracyV2EnvCfg
from softmimic_gym.assets.robots.g1_compat import G1_SOFTMIMIC_BILATERAL_ARM_ACTION_IDS,G1_SOFTMIMIC_KEYPOINT_BODY_NAMES
ARM=list(G1_SOFTMIMIC_BILATERAL_ARM_ACTION_IDS)
LOWER=ROOT/'logs/rsl_rl/g1_neutral_stand_geometry_v1/2026-09-09_21-13-26_scratch_cached_v1/model_250.jit'
UPPER=ROOT/'logs/rsl_rl/g1_stand_coordination_v1/2026-08-21_10-59-05_stand_coordination_accuracy_v2_alpha050_supervised_validation/model_1375.jit'

def delivered_actions(env):
    return getattr(env,'_dual_delivered',torch.zeros((env.num_envs,29),device=env.device))

class DualAction(LegAction):
    def apply_actions(self):
        super().apply_actions()
        if args.mode=='connected':
            c=self._env.command_manager.get_term('upper_body_joints')
            raw=getattr(self._env,'_dual_arm_raw',torch.zeros((self.num_envs,len(ARM)),device=self.device))
            ids=[self._asset.joint_names.index(NAMES[i]) for i in ARM]
            self._asset.set_joint_position_target(c.dof_pos[:,1,ARM]+raw*.15,joint_ids=ids)

def main():
    assert 0<=args.scale<=1
    if args.scale:
        raise ValueError('Neutral connected gate has not passed; motion stage is disabled pending acceptance.')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=NeutralStandGeometryCfg();cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;cfg.sim.device=args.device
    old=G1StandCoordinationAccuracyV2EnvCfg()
    # Exact old upper-body actuator configuration, no silent gain migration.
    assert str(old.scene.robot.actuators['arms'])==str(cfg.scene.robot.actuators['arms'])
    assert all(old.actions.joint_pos.scale[NAMES[i]]==.15 for i in ARM)
    if args.scale:
        source=ROOT/'datasets/motions_csv/stand_coordination_v1'
        neutral=np.loadtxt(cfg.commands.upper_body_joints.demo_recording_path[0],delimiter=',')
        sagittal=np.loadtxt(source/'01_bilateral_sagittal.csv',delimiter=',')
        order=json.load(open(source/'manifest.json'))['joint_order']
        motion=np.repeat(neutral[:1],len(sagittal),axis=0)
        for i in ARM:
            col=7+order.index(NAMES[i])
            motion[:,col]=neutral[0,col]+args.scale*(sagittal[:,col]-neutral[0,col])
        np.savetxt(out/'reference.csv',motion,delimiter=',',fmt='%.12f')
        cfg.commands.upper_body_joints.demo_recording_path=[str(out/'reference.csv')]
        # Only arm q changes; loader FK regenerates spatial references.
        fixed=[i for i in range(motion.shape[1]) if i not in [7+order.index(NAMES[j]) for j in ARM]]
        np.testing.assert_array_equal(motion[:,fixed],np.repeat(neutral[:1,fixed],len(motion),axis=0))
    if args.mode!='baseline':
        cfg.observations.legacy=copy.deepcopy(old.observations.policy)
        cfg.observations.legacy.enable_corruption=False
        cfg.observations.legacy.actions.func=delivered_actions
        cfg.observations.legacy.actions.params={}
        cfg.observations.left_arm_adapter=copy.deepcopy(old.observations.left_arm_adapter)
        cfg.observations.right_arm_adapter=copy.deepcopy(old.observations.right_arm_adapter)
    cfg.actions.joint_pos.class_type=DualAction
    task='Isaac-G1-FrozenDualController-Diagnostic-v0'
    gym.register(id=task,entry_point='isaaclab.envs:ManagerBasedRLEnv',disable_env_checker=True)
    env=gym.make(task,cfg=cfg);e=env.unwrapped
    try:
        obs,_=env.reset(seed=args.seed)
        lower=torch.jit.load(str(LOWER),map_location=e.device).eval()
        upper=torch.jit.load(str(UPPER),map_location=e.device).eval() if args.mode!='baseline' else None
        r,c,ids,fids,feet,fe,com,forces=state(e)
        assert obs['policy'].shape[1]==109 and e.action_manager.total_action_dim==12
        if upper:assert obs['legacy'].shape[1]==705,(obs['legacy'].shape)
        torch.testing.assert_close(r.data.root_pos_w[:,2],c.root_state['root_pos'][:,1,2],rtol=0,atol=1e-5)
        assert abs(float(r.data.root_pos_w[0,2])-.7841210365)<1e-5
        arm_body_names=['left_shoulder_yaw_link','left_elbow_link','left_wrist_yaw_link','right_shoulder_yaw_link','right_elbow_link','right_wrist_yaw_link']
        body_ids=[r.body_names.index(n) for n in arm_body_names];kp_ids=[list(G1_SOFTMIMIC_KEYPOINT_BODY_NAMES).index(n) for n in arm_body_names]
        (out/'parameters.json').write_text(json.dumps({'mode':args.mode,'scale':args.scale,'lower':str(LOWER),'upper':str(UPPER),'lower_ids':LEG_IDS,'arm_ids':ARM,'waist_ids':[2,5,8],'legacy_history':'delivered disjoint 29DOF normalized action; discarded legacy legs/waist never executed','gains':str(cfg.scene.robot.actuators),'force':'none'},indent=2))
        alive=torch.ones(args.num_envs,dtype=torch.bool,device=e.device);first=torch.full((args.num_envs,),-1,device=e.device)
        trace=[];aggregates=[];maxroot=0.;maxfoot=0.;maxtilt=0.
        with torch.no_grad():
            for step in range(args.steps):
                action=lower(obs['policy']);raw=torch.zeros((args.num_envs,29),device=e.device)
                if upper:raw=upper(torch.cat([obs['legacy'],obs['left_arm_adapter'],obs['right_arm_adapter']],dim=1))
                assert torch.isfinite(raw).all() and torch.isfinite(action).all()
                delivered=torch.zeros_like(raw);delivered[:,LEG_IDS]=action
                if args.mode=='connected':delivered[:,ARM]=raw[:,ARM]
                e._dual_delivered=delivered;e._dual_arm_raw=raw[:,ARM]
                r,c,ids,fids,feet,fe,com,forces=state(e)
                trace.append(torch.cat([obs['policy'],action,r.data.root_state_w,r.data.joint_pos,r.data.joint_vel],dim=1).cpu())
                valid=alive
                if valid.any():
                    root=(r.data.root_pos_w[:,:2]-c.root_state['root_pos'][:,1,:2]).norm(dim=1)
                    tilt=torch.acos((-r.data.projected_gravity_b[:,2]).clamp(-1,1))*180/torch.pi
                    maxroot=max(maxroot,float(root[valid].max()));maxfoot=max(maxfoot,float(fe[valid].norm(dim=-1).max()));maxtilt=max(maxtilt,float(tilt[valid].max()))
                    qerr=(r.data.joint_pos[:,ids][:,ARM]-c.dof_pos[:,1,ARM]).abs()
                    kerr=(r.data.body_pos_w[:,body_ids]-c.keypoints[:,1,kp_ids]).norm(dim=-1)
                    aggregates.append([step,int(alive.sum()),float(root[valid].mean()),float(r.data.root_lin_vel_w[valid,:2].norm(dim=1).mean()),float((com[valid,:2]-feet[valid].mean(1)[:,:2]).norm(dim=1).mean()),*qerr[valid].mean(0).cpu().tolist(),*kerr[valid].mean(0).cpu().tolist()])
                expected=c.dof_pos[:,1].clone()+delivered*torch.tensor([.30 if i in LEG_IDS else (.20 if i in [2,5,8] else .15) for i in range(29)],device=e.device)
                obs,_,done,timeout,_=env.step(action)
                # Check actual actuator target, including disjoint ownership.
                good=~(done|timeout)
                if good.any():
                    torch.testing.assert_close(r.data.joint_pos_target[good][:,ids],expected[good],atol=1e-5,rtol=0)
                first[alive&done]=step+1;alive&=~done
                if not alive.any():break
        np.save(out/'lower_trace.npy',torch.stack(trace).numpy())
        import csv
        with (out/'metrics.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['step','alive','root_error_mean_m','root_speed_mean_mps','com_offset_mean_m']+['q_mae:'+NAMES[i] for i in ARM]+['keypoint_mae:'+n for n in arm_body_names]);w.writerows(aggregates)
        result={'mode':args.mode,'scale':args.scale,'executed_steps':step+1,'num_envs':args.num_envs,'failures':int((first>=0).sum()),'first_failure_min':int(first[first>=0].min()) if (first>=0).any() else None,'root_max_m':maxroot,'foot_max_m':maxfoot,'tilt_max_deg':maxtilt}
        (out/'summary.json').write_text(json.dumps(result,indent=2));print('DUAL_RESULT',json.dumps(result),flush=True)
    finally:env.close()
try:main()
except BaseException:
    import traceback
    traceback.print_exc();sys.stderr.flush();raise
finally:app.close()
