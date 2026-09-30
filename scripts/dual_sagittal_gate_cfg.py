#new change
"""15% arms-only sagittal safety gate; requires accepted neutral adaptation."""
import json
import numpy as np
from isaaclab.utils import configclass
from neutral_dual_adapt_cfg import NeutralDualAdaptCfg,ROOT,ARM,NAMES

@configclass
class Sagittal15Cfg(NeutralDualAdaptCfg):
    #new change
    arm_motion_scale: float = 0.15
    reference_folder: str = 'neutral_dual_sagittal015_gate'
    def __post_init__(self):
        super().__post_init__()
        for seed in (0,7,123):
            log=ROOT/f'logs/stand_coordination_v1/neutral_dual_adapt_v1/model299_seed{seed}_512_eval.log'
            lines=log.read_text().splitlines()
            support=[json.loads(s.split(' ',1)[1]) for s in lines if s.startswith('SUPPORT_ACCEPTANCE ')][-1]
            result=[json.loads(s.split(' ',1)[1]) for s in lines if s.startswith('EVAL_RESULT ')][-1]
            if support['qualified_fraction']!=1 or result['executed_steps']!=1500 or result['failures']!=0:
                raise RuntimeError('Neutral multiseed gate not accepted')
        # folder=ROOT/'datasets/motions_csv/neutral_dual_sagittal015_gate'
        #new change
        folder=ROOT/'datasets/motions_csv'/self.reference_folder
        folder.mkdir(exist_ok=True)
        source=ROOT/'datasets/motions_csv/stand_coordination_v1'
        neutral=np.loadtxt(self.commands.upper_body_joints.demo_recording_path[0],delimiter=',')
        sagittal=np.loadtxt(source/'01_bilateral_sagittal.csv',delimiter=',')
        order=json.load(open(source/'manifest.json'))['joint_order']
        motion=np.repeat(neutral[:1],len(sagittal),axis=0)
        for i in ARM:
            col=7+order.index(NAMES[i])
            # motion[:,col]=neutral[0,col]+.15*(sagittal[:,col]-neutral[0,col])
            #new change
            motion[:,col]=neutral[0,col]+self.arm_motion_scale*(sagittal[:,col]-neutral[0,col])
        path=folder/'reference.csv'
        if path.exists():np.testing.assert_allclose(np.loadtxt(path,delimiter=','),motion,rtol=0,atol=1e-11)
        else:np.savetxt(path,motion,delimiter=',',fmt='%.12f')
        self.commands.upper_body_joints.demo_recording_path=[str(path)]

#new change
@configclass
class Sagittal25Cfg(Sagittal15Cfg):
    """Independent 25% reference; all controller and physics settings unchanged."""
    arm_motion_scale: float = 0.25
    reference_folder: str = 'neutral_dual_sagittal025_gate'

    def __post_init__(self):
        for seed in (0, 7, 123):
            log = ROOT/f'logs/stand_coordination_v1/neutral_dual_adapt_v1/sagittal015_long_validation/seed{seed}.log'
            lines = log.read_text().splitlines()
            result = [json.loads(s.split(' ', 1)[1]) for s in lines if s.startswith('EVAL_RESULT ')][-1]
            support = [json.loads(s.split(' ', 1)[1]) for s in lines if s.startswith('SUPPORT_ACCEPTANCE ')][-1]
            if result['executed_steps'] != 1500 or result['failures'] != 0 or support['qualified_fraction'] != 1:
                raise RuntimeError('15% long-rollout multiseed gate not accepted')
        super().__post_init__()
