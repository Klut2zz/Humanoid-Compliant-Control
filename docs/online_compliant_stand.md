# Online Compliant Stand

<!-- new change -->
`Isaac-G1-StandCoordinationOnlineCompliant-100Pct-29DOF-v0` is the online,
on-policy compliance task.  It does not read an augmented CSV and does not
write a transition dataset for later training.

## Rollout data flow

For every IsaacLab environment, the command term keeps GPU tensors for:

1. force body (torso, left/right elbow, or left/right wrist),
2. external force and desired stiffness,
3. a random wait followed by ramp-up/hold/ramp-down,
4. the Cartesian request `x_compliant = x_reference + force/stiffness`,
5. a constrained 29DOF reference produced by batched GPU DLS-IK.

The same command buffers drive the physics wrench, the actor/critic
observations, and the compliant-link reward.  RSL-RL consumes those samples in
the current 48-step rollout and updates PPO immediately.

The IK objective retains the priority structure of the original offline
augmentation: active-link position, both feet, whole-body COM, pelvis
orientation, all keypoints, and posture.  Candidate corrections are bounded
per joint group and scaled or rejected when predicted active-link residual,
foot motion, or COM residual exceeds the low-disturbance gate.  The accepted
solution generates mutually consistent `q_ref`, `dq_ref`, and keypoint
references for every future observation step.

<!-- new change -->
The feasibility gate also checks the robot soft joint-position limits and 50%
of the configured actuator velocity limits after combining nominal motion
velocity with IK ramp velocity.  An unreachable or excessively fast request
is rejected before it can become a wrench/reference training sample.  A
rejected event exposes zero force to physics and policy observations.

Events sample a 0.5--1.5 s quiet period, a 0.2--0.6 s linear ramp, a 0.5--1.0
s hold, then a symmetric ramp down.  Torso events are downward; elbow/wrist
events use low-vertical random directions.  The first curriculum uses 2--6 N;
the generator configuration retains an explicit range for later expansion.
The accepted wrench and the
accepted IK reference share one feasibility scale, so physics never receives
a force whose compliant reference was rejected.

## Verified gates (seed 42)

- zero-force task, 32 environments x 500 steps: zero failures and exact zero
  q/keypoint/reference delta;
- 2--6 N task, 32 environments x 500 steps: zero failures, 96/128 feasible
  events accepted;
- accepted-event direction cosine: mean 0.9966, minimum 0.9029;
- predicted maximum foot motion: 0.106 mm; maximum COM residual: 17.73 mm;
<!-- new change -->
- closed-loop active-link error, zero-force versus 2--6 N: mean 6.77 mm
  versus 8.70 mm, maximum 26.33 mm versus 33.65 mm.  Thus the low-force
  online augmentation adds about 1.93 mm mean and 7.33 mm worst-case error;
- 16 environments x one PPO iteration: rollout, backward update, checkpoint,
  and JIT export completed.  This run is diagnostic-only and is not a model
  candidate.

Reproduce a finite diagnostic with:

```bash
python scripts/rsl_rl/test.py \
  --task Isaac-G1-StandCoordinationOnlineCompliant-100Pct-29DOF-v0 \
  --num_envs 32 \
  --load_run stand_coordination_accuracy_v2_alpha050_supervised_bootstrap \
  --checkpoint model_1375_alpha050_supervised.pt \
  --seed 42 \
  --max_test_steps 500 \
  --online_reference_diagnostics \
  --headless
```

The JSON report is saved inside that run's `logs/test_runs/...` directory.

## Visual data-quality gate

<!-- new change -->
The visualization reuses the original SoftMimic marker style from
`compliant_motion_augmentation/visualization.py` and the shared strict 14-body
mapping from `g1_compat.py`.  It draws actual/nominal/online-IK keypoints,
accepted or rejected Cartesian targets, the synchronized force arrow, COM and
the support midpoint.  The video overlay records the force vector and the
position/velocity/foot/COM/direction feasibility measurements.

```bash
python scripts/rsl_rl/test.py \
  --task Isaac-G1-StandCoordinationOnlineCompliant-VisualQuality-100Pct-29DOF-v0 \
  --num_envs 1 \
  --load_run stand_coordination_accuracy_v2_alpha050_supervised_bootstrap \
  --checkpoint model_1375_alpha050_supervised.pt \
  --seed 42 \
  --online_reference_diagnostics \
  --video \
  --video_length 1500 \
  --headless
```

Color legend: blue is measured, red is nominal, yellow is online IK, orange is
force, green is an accepted target, purple is a rejected target, pink is COM,
and cyan is the support midpoint.

## Protected policy structure

- permanent fallback: `model_1375_alpha050_supervised.pt`;
- frozen: legacy actor, accepted bilateral arm adapters, lower-body actions;
- trainable: only the zero-initialized 54D online compliance adapter;
- output authority: 4 mrad on waist joints, 20 mrad on arm joints;
- exploration/log-prob/KL: only the 17 waist/arm rows owned by the new head;
- critic: reinitialized because force context adds new critic inputs.

## Diagnostic-only offline artifact

`scripts/generate_stand_low_disturbance_data.py` and its `.pt` files are kept
only for safety/data diagnostics.  They are not inputs to this online task.

## Conservative first training command

```bash
mkdir -p logs/stand_coordination_v1/online_compliant

python scripts/rsl_rl/train.py \
  --task Isaac-G1-StandCoordinationOnlineCompliant-100Pct-29DOF-v0 \
  --num_envs 4096 \
  --max_iterations 20 \
  --resume True \
  --load_run stand_coordination_accuracy_v2_alpha050_supervised_bootstrap \
  --checkpoint model_1375_alpha050_supervised.pt \
  --reset_optimizer \
  --reset_action_std 0.01 \
  --run_name stand_online_compliant_low_force_v1 \
  --seed 42 \
  --headless \
  2>&1 | tee logs/stand_coordination_v1/online_compliant/train_v1_console.log
```

The zero-force equivalence, force-direction, IK feasibility, multi-environment
stability, and one-iteration training gates above have passed.  This remains a
short low-force curriculum, not a large-scale compliant PPO run.  Keep
`model_1375_alpha050_supervised.pt` unchanged as the permanent rollback point.
