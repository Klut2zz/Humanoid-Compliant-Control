#new change
"""Video-only wrapper around the accepted frozen amplitude test."""
from pathlib import Path
original=Path.read_text
def read(self,*args,**kwargs):
    s=original(self,*args,**kwargs)
    if self.name=='test_neutral_stand_geometry.py':
        old='env=gym.make(TASK,cfg=cfg);e=env.unwrapped'
        assert s.count(old)==1
        s=s.replace(old,"""cfg.viewer.eye=(2.8,2.8,1.8);cfg.viewer.lookat=(0.,0.,.85)
    cfg.viewer.resolution=(1280,720)
    #new change: prevent asset-root camera tracking from adding torso height twice.
    cfg.viewer.origin_type='world'
    env=gym.make(TASK,cfg=cfg,render_mode='rgb_array');e=env.unwrapped
    import imageio.v2 as imageio
    from pathlib import Path
    import os
    video_path=Path(os.environ['VIDEO_PATH'])
    assert not video_path.exists(),video_path
    writer=imageio.get_writer(str(video_path),fps=round(1/e.step_dt))""")
        needle='                assert torch.isfinite(reward).all()'
        assert s.count(needle)==1
        s=s.replace(needle,needle+'\n                writer.append_data(env.render())')
        #new change: frame the live robot location, including terrain/environment origin.
        needle='        obs,_=env.reset(seed=args.seed)'
        assert s.count(needle)==1
        s=s.replace(needle,needle+"\n        anchor=e.scene['robot'].data.root_pos_w[0].detach().cpu().numpy()\n        e.sim.set_camera_view(eye=(float(anchor[0]+2.5),float(anchor[1]+2.5),1.7), target=(float(anchor[0]),float(anchor[1]),.75))")
        s=s.replace('finally:env.close()',"finally:\n        writer.close()\n        print('VIDEO_SAVED',str(video_path),flush=True)\n        env.close()")
    return s
Path.read_text=read
path=Path(__file__).with_name('test_dual_amplitude_50_100.py')
exec(compile(path.read_text(),str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
