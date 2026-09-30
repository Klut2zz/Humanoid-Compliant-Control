#new change
"""Read-only runtime parameter dump; implicit torque buffers are estimates."""
import json

def metadata(robot):
    if getattr(robot,'_wrist_audit_printed',False):return 0
    j=robot.joint_names.index('right_wrist_pitch_joint')
    data={}
    for key in ('joint_stiffness','joint_damping','joint_armature','joint_friction_coeff','joint_pos_limits','soft_joint_pos_limits','joint_vel_limits','joint_effort_limits'):
        value=getattr(robot.data,key,None)
        if value is not None:data[key]=value[:,j].detach().cpu().tolist()[:1]
    for name,a in robot.actuators.items():
        if 'right_wrist_pitch_joint' in a.joint_names:
            k=a.joint_names.index('right_wrist_pitch_joint')
            data['actuator_name']=name;data['actuator_type']=type(a).__name__
            for key in ('stiffness','damping','effort_limit','velocity_limit','armature'):
                value=getattr(a,key,None)
                if value is not None:data['actuator_'+key]=value[:,k].detach().cpu().tolist()[:1]
            data['cfg']=str(a.cfg)
    print('WRIST_ACTUATOR_AUDIT',json.dumps(data),flush=True)
    robot._wrist_audit_printed=True
    return 0
