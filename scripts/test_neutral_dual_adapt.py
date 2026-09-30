#new change
"""Use the same acceptance checks, with the frozen arms internal to the env."""
from pathlib import Path
import sys
root=Path('/home/noah/mingkai/general_softmimic');sys.path.insert(0,str(root/'scripts'))
path=root/'scripts/test_neutral_stand_geometry.py';s=path.read_text()
needle='def main():'
assert s.count(needle)==1
s=s.replace(needle,'from neutral_dual_adapt_cfg import NeutralDualAdaptCfg, ARM\n\n'+needle)
s=s.replace('cfg=NeutralStandGeometryCfg();','cfg=NeutralDualAdaptCfg();')
s=s.replace('r.data.joint_pos[:,ids],c.dof_pos[:,1]','r.data.joint_pos[:,ids],c.dof_pos[:,0]')
needle="if not (terminated|truncated).any():torch.testing.assert_close(r.data.joint_pos_target[:,upper_ids],c.dof_pos[:,1,UPPER_IDS],atol=1e-5,rtol=0)"
assert s.count(needle)==1
s=s.replace('                obs,reward,terminated,truncated,_=env.step(action)','                reference_before=c.dof_pos[:,1].clone()\n                obs,reward,terminated,truncated,_=env.step(action)')
s=s.replace(needle,"if not (terminated|truncated).any():\n                    expected=reference_before.clone()\n                    expected[:,ARM]+=e._dual_arm_raw*.15\n                    torch.testing.assert_close(r.data.joint_pos_target[:,upper_ids],expected[:,UPPER_IDS],atol=1e-5,rtol=0)")
#new change
s=s.replace('        with torch.no_grad():','''        arm_error_sum=torch.zeros(14,device=e.device)
        kp_error_sum=torch.zeros(6,device=e.device)
        arm_samples=0
        from softmimic_gym.assets.robots.g1_compat import G1_SOFTMIMIC_KEYPOINT_BODY_NAMES
        kp_names=['left_shoulder_yaw_link','left_elbow_link','left_wrist_yaw_link','right_shoulder_yaw_link','right_elbow_link','right_wrist_yaw_link']
        kp_ids=[list(G1_SOFTMIMIC_KEYPOINT_BODY_NAMES).index(n) for n in kp_names]
        body_ids=[r.body_names.index(n) for n in kp_names]
        with torch.no_grad():''')
s=s.replace('                if step>=50:', '''                if step>=50 and valid.any():
                    arm_error_sum+=(r.data.joint_pos[valid][:,ids][:,ARM]-c.dof_pos[valid,1][:,ARM]).abs().sum(0)
                    kp_error_sum+=(r.data.body_pos_w[valid][:,body_ids]-c.keypoints[valid,1][:,kp_ids]).norm(dim=-1).sum(0)
                    arm_samples+=int(valid.sum())
                if step>=50:''')
s=s.replace("        print('SUPPORT_ACCEPTANCE'", "        print('ARM_METRICS',json.dumps({'post1s_valid_samples':arm_samples,'q_mae_deg':dict(zip([NAMES[i] for i in ARM],(arm_error_sum/max(arm_samples,1)*180/torch.pi).cpu().tolist())),'keypoint_world_mae_m':dict(zip(kp_names,(kp_error_sum/max(arm_samples,1)).cpu().tolist()))}),flush=True)\n        print('SUPPORT_ACCEPTANCE'")
exec(compile(s,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
