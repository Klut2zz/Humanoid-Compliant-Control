#new change
"""Observation-only instrumentation. No actuator/material/collision edits."""
import os
import json
from pathlib import Path
import numpy as np
from fixed100_prefailure_cfg import Fixed100PrefailureCfg as Base
from isaaclab.sensors import ContactSensorCfg

BODIES = [s+'_'+b for s in ('left','right') for b in ('ankle_roll_link','ankle_pitch_link','knee_link')]
class ObservationCfg(Base):
    def __post_init__(self):
        super().__post_init__()
        for b in BODIES:
            setattr(self.scene,'observe_'+b,ContactSensorCfg(
                prim_path='{ENV_REGEX_NS}/Robot/'+b,update_period=0.,history_length=0,
                filter_prim_paths_expr=['/World/ground/terrain/mesh'],
                track_contact_points=True,track_friction_forces=True,
                max_contact_data_count_per_prim=64))

def arr(x):
    return x.detach().cpu().numpy().copy()

def capture(e):
    r=e.scene['robot']; out={}
    if not hasattr(e,'_observation_meta_done'):
        from pxr import Usd,UsdPhysics,UsdGeom
        colliders=[]
        for p in Usd.PrimRange(e.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot'),Usd.TraverseInstanceProxies()):
            if not p.HasAPI(UsdPhysics.CollisionAPI): continue
            owner=p
            while owner and not owner.IsPseudoRoot() and not owner.HasAPI(UsdPhysics.RigidBodyAPI): owner=owner.GetParent()
            cache=UsdGeom.XformCache()
            colliders.append(dict(path=str(p.GetPath()),owner=str(owner.GetPath()),type=p.GetTypeName(),
                local_to_body=np.array(cache.GetLocalToWorldTransform(p)*cache.GetLocalToWorldTransform(owner).GetInverse()).tolist(),
                attrs={a.GetName():str(a.Get()) for a in p.GetAttributes() if a.GetName().startswith(('physics:','physx','radius','extent'))}))
        meta=dict(body_names=r.body_names,joint_names=r.joint_names,observed_bodies=BODIES,colliders=colliders,
            physics_dt=e.physics_dt,step_dt=e.step_dt,
            semantics='PRE-step; contact forces are last physics substep; raw contact slots carry body identity, not exact shape identity; computed/applied torque are implicit PD estimates.')
        for name in ('get_masses','get_coms','get_inertias','get_material_properties'):
            try: meta[name]=arr(getattr(r.root_physx_view,name)()).tolist()
            except Exception as ex: meta[name]=str(ex)
        for name in ('joint_stiffness','joint_damping','joint_armature','joint_effort_limits','joint_vel_limits','joint_pos_limits'):
            try: meta[name]=arr(getattr(r.data,name)[0]).tolist()
            except Exception as ex: meta[name]=str(ex)
        meta['shapes']={}
        for b in BODIES:
            view=e.scene['observe_'+b].body_physx_view
            meta['shapes'][b]={}
            for name in ('get_contact_offsets','get_rest_offsets','get_material_properties'):
                try: meta['shapes'][b][name]=arr(getattr(view,name)())[0].tolist()
                except Exception as ex: meta['shapes'][b][name]=str(ex)
        path=Path(os.environ['TRACE_PATH']).with_suffix('.metadata.json')
        assert not path.exists(),path
        path.write_text(json.dumps(meta,indent=2))
        e._observation_meta_done=True
    for name in ('joint_pos','joint_vel','joint_pos_target','joint_vel_target','computed_torque','applied_torque','body_pos_w','body_quat_w','body_com_pos_w','body_com_vel_w'):
        out['audit_'+name]=arr(getattr(r.data,name))
    out['audit_masses']=arr(r.root_physx_view.get_masses())
    out['audit_all_body_normal']=arr(e.scene['contact_forces'].data.net_forces_w)
    for b in BODIES:
        ss=e.scene['observe_'+b]; d=ss.data
        for name in ('force_matrix_w','friction_forces_w','contact_pos_w'):
            out[b+'_'+name]=arr(getattr(d,name))
        f,p,n,dist,count,start=ss.contact_physx_view.get_contact_data(dt=e.physics_dt)
        # Fixed padded manifold for each environment. Counts retain overflow evidence.
        counts=arr(count).reshape(e.num_envs,-1).sum(1)
        starts=arr(start).reshape(e.num_envs,-1)[:,0]
        raw=np.full((e.num_envs,16,8),np.nan,dtype=np.float32)
        ff,pp,nn,dd=arr(f),arr(p),arr(n),arr(dist)
        for i,(cnt,st) in enumerate(zip(counts,starts)):
            k=min(int(cnt),16); sl=slice(int(st),int(st)+k)
            if k: raw[i,:k]=np.concatenate((pp[sl],nn[sl],ff[sl].reshape(k,1),dd[sl].reshape(k,1)),axis=1)
        out[b+'_raw_point_normal_force_distance']=raw
        out[b+'_contact_count']=counts
    return out
