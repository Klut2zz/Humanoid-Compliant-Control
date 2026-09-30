#new change
"""Target-only Roll bias; unchanged raw actions, actor and adapter weights."""
import torch

def install(env,bias):
    assert bias in (0.,.01,.02)
    term=env.action_manager.get_term('joint_pos')
    robot=env.scene['robot']
    index=robot.joint_names.index('right_wrist_roll_joint')
    original=term.apply_actions
    def apply():
        original()
        before=robot.data.joint_pos_target.clone()
        robot.set_joint_position_target(before[:,index:index+1]+bias,joint_ids=[index])
        expected=before.clone();expected[:,index]+=bias
        torch.testing.assert_close(robot.data.joint_pos_target,expected,atol=1e-7,rtol=0)
    term.apply_actions=apply
    print('ROLL_TARGET_PROBE',{'bias_rad':bias,'joint':robot.joint_names[index],'raw_actions_unchanged':True},flush=True)
