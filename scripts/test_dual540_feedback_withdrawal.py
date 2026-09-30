#new change
"""Continue versus withdraw the diagnostic hip/ankle feedback, never PPO.

Replays identical feedback from step 100 through step 150. At 150 the fade
branch latches the delivered correction and takes it to zero with a half-cosine
over 0.25 seconds. This diagnostic time is not a deployable gate condition.
"""
from pathlib import Path
from test_dual540_sagittal_feedback import build_source as parent_source
from test_dual540_residual_release_branch import snapshot


def build_source():
    source,path=parent_source()
    changes={
        "choices=['baseline','repeat','feedback']":"choices=['continue','fade']",
        "args=p.parse_args()":
        "p.add_argument('--fade_step',type=int,default=150)\np.add_argument('--fade_duration',type=float,default=.25)\nargs=p.parse_args()\nassert args.fork_step==100 and args.fade_step==150 and args.fade_duration==.25",
        '        kinematic_slope=None':'''        kinematic_slope=None
        fade_start=torch.zeros_like(feedback)
        fade_selected=torch.zeros(e.num_envs,device=e.device,dtype=torch.bool)''',
        "                    if args.branch=='feedback' and step>args.fork_step:":'''                    #new change: freeze the current added correction, not
                    # model84's output, at the common diagnostic branch point.
                    if step==args.fade_step:
                        fade_start.copy_(feedback)
                        fade_selected.copy_(alive)
                        snap=snapshot(e,term,wrapped)
                        snap['feedback_state']=dict(smooth=smooth.cpu().clone(),anchor=anchor.cpu().clone(),
                            forward=forward.cpu().clone(),fade_start=fade_start.cpu().clone(),
                            selected=fade_selected.cpu().clone(),pending_actor_action=action.cpu().clone())
                        torch.save(snap,Path(args.output).with_suffix('.withdrawal_snapshot.pt'))
                        print('WITHDRAWAL_FORK',step,'alive',int(alive.sum()),flush=True)
                    if args.branch=='fade' and step>args.fade_step:
                        phase=min((step-args.fade_step)*e.step_dt/args.fade_duration,1.)
                        envelope=.5*(1.+np.cos(np.pi*phase)) if phase<1. else 0.
                        feedback.copy_(fade_start*envelope)
                        feedback[~alive]=0
                    if step>args.fork_step:''',
        'branch=np.asarray(args.branch),kinematic_slope=kinematic_slope)':
        'branch=np.asarray(args.branch),kinematic_slope=kinematic_slope,fade_step=np.asarray(args.fade_step),fade_duration=np.asarray(args.fade_duration),fade_selected=fade_selected.cpu().numpy())',
    }
    for old,new in changes.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    return source,path


if __name__=='__main__':
    source,path=build_source()
    exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path),'snapshot':snapshot})
