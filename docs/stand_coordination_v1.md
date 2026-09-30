# StandCoordinationV1

StandCoordinationV1 is the force-free, full-amplitude control basis for the
future RobustStand and CompliantStand experiments.  It is isolated from the
older multi-motion StationaryCore task and never loads T-pose, Pour, or Walk.

## Reference contract

- Robot interface: strict G1 29DOF body joints; the 14 finger joints are not
  policy actions.
- Root: fixed at `(0, 0, 0.74)` with identity orientation.
- Lower body: symmetric neutral pose from
  `G1_SOFTMIMIC_29DOF_COMPAT_CFG.init_state`.
- Contacts: both feet remain commanded in contact.
- Runtime motion scale: exactly `1.0`; the CSV amplitudes are the 100% task.
- Sampling: 30 Hz source data, 24 seconds per file, minimum-jerk transitions,
  20-second RL episodes sampled from valid continuous windows.

The eight references are:

1. neutral standing;
2. bilateral sagittal arm/waist coordination;
3. bilateral lateral arm coordination;
4. alternating unilateral reach with waist counterbalance;
5. waist/shoulder axial counter-rotation;
6. waist-roll/shoulder coordination;
7. waist-pitch/shoulder coordination;
8. combined asymmetric coordination.

The generator and exact amplitudes live in
`scripts/generate_stand_coordination_v1.py`.  Generated files and the machine
readable manifest live under
`datasets/motions_csv/stand_coordination_v1/`.

## Task ids and curriculum

Stage A learns active balance from scratch on the new neutral pose:

```text
Isaac-G1-StandCoordinationV1-NeutralBootstrap-100Pct-29DOF-v0
```

Stage B loads the accepted Stage-A checkpoint and trains all eight 100%
coordination references with one unified 29DOF actor:

```text
Isaac-G1-StandCoordinationV1-100Pct-29DOF-v0
```

Both stages have the same 705D actor observation and 29D action contract.  No
hard gate, residual adapter, external force, dynamics randomization, or frozen
actor block is enabled.  Stage B uses 48 rollout steps and a more conservative
optimizer because the balance policy is already initialized.

The older `model_1598` must not initialize Stage A: a fixed-condition test on
the new neutral reference failed before the first coordinated movement.  It
remains an archived GeneralTrackingV1 checkpoint, not a StandCoordinationV1
checkpoint.

## Reproducible checks

Validate dimensions, limits, exact plateaus, continuity, and MuJoCo FK:

```bash
python scripts/validate_stand_coordination_v1.py
```

Construct the final task without training:

```bash
python scripts/smoke_general_tracking.py \
  --task Isaac-G1-StandCoordinationV1-100Pct-29DOF-v0 \
  --num_envs 8 \
  --steps 5 \
  --exact_reference_reset \
  --headless
```

Long zero-residual failure is expected because the low-stiffness humanoid needs
an active balance policy; it is not a reference-contract failure.

## Training hand-off

Stage A, scratch neutral bootstrap:

```bash
mkdir -p logs/stand_coordination_v1

python scripts/rsl_rl/train.py \
  --task Isaac-G1-StandCoordinationV1-NeutralBootstrap-100Pct-29DOF-v0 \
  --num_envs 8192 \
  --max_iterations 1000 \
  --run_name stand_coordination_v1_neutral_bootstrap_v1 \
  --seed 42 \
  --headless \
  2>&1 | tee logs/stand_coordination_v1/neutral_bootstrap_v1_console.log
```

Do not start Stage B until a 512-environment, 1500-step, multi-seed Stage-A
validation shows a stable neutral stance.  Then resume the accepted Stage-A
checkpoint into Stage B with a fresh optimizer and action standard deviation:

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-G1-StandCoordinationV1-100Pct-29DOF-v0 \
  --num_envs 4096 \
  --max_iterations 1000 \
  --resume True \
  --load_run <NEUTRAL_BOOTSTRAP_RUN> \
  --checkpoint <ACCEPTED_CHECKPOINT> \
  --reset_optimizer \
  --reset_action_std 0.08 \
  --run_name stand_coordination_v1_full_v1 \
  --seed 42 \
  --headless \
  2>&1 | tee logs/stand_coordination_v1/full_v1_console.log
```

Stage-B acceptance must be based on all eight references, per-joint and
per-keypoint tracking, waist/base tilt, foot slip, failure phase, and population
success over seeds 0, 7, 42, and 123.  Compliance forces are deliberately out
of scope until this nominal controller passes.

## Phase-resolved model_1375 diagnosis and AccuracyV1

<!-- new change -->
`model_1375.pt` from
`2026-08-20_18-57-37_stand_coordination_v1_full_v1` is the accepted nominal
StandCoordinationV1 checkpoint. The independent diagnostic does not mutate
the environment or policy:

```bash
python scripts/diagnose_stand_coordination.py \
  --task Isaac-G1-StandCoordinationV1-100Pct-29DOF-v0 \
  --num_envs 512 \
  --steps 1500 \
  --seed 42 \
  --phase_bins 48 \
  --episode_warmup_steps 50 \
  --jit_policy logs/rsl_rl/g1_stand_coordination_v1/2026-08-20_18-57-37_stand_coordination_v1_full_v1/model_1375.jit \
  --output_dir logs/stand_coordination_v1/model1375_phase_diagnosis_seed42_512x1500 \
  --headless
```

The diagnosis found fixed signed right-wrist bias on every motion, a smaller
dynamic right-wrist-roll lag in Combined Coordination, and phase-local
waist-pitch error in Sagittal/Pitch Coordination. The official G1 contact
sensor currently attributes the dominant support forces to knee-link entries,
not the ankle-roll entries assumed by the inherited `feet_stumble` term.
AccuracyV1 therefore does not increase that unreliable contact-gated reward.

The independent refinement task is:

```text
Isaac-G1-StandCoordinationAccuracyV1-100Pct-29DOF-v0
```

It keeps all references, mapping, actuator settings, lower-body rewards, and
stability terms unchanged. Only the three waist and six wrist actor output
rows train; the actor trunk and every other output row remain frozen. When
training is authorized, use:

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-G1-StandCoordinationAccuracyV1-100Pct-29DOF-v0 \
  --num_envs 4096 \
  --max_iterations 25 \
  --resume True \
  --load_run 2026-08-20_18-57-37_stand_coordination_v1_full_v1 \
  --checkpoint model_1375.pt \
  --reset_optimizer \
  --reset_action_std 0.02 \
  --run_name stand_coordination_accuracy_v1_from_model1375 \
  --seed 42 \
  --headless \
  2>&1 | tee logs/stand_coordination_v1/accuracy_v1_from_model1375_console.log
```

Do not promote the final checkpoint automatically. Compare checkpoints every
five iterations against `model_1375` using the same phase report, and reject
any candidate whose multi-seed first-bad count rises even if wrist MAE falls.

## StandCoordinationAccuracyV2 structured bilateral adapters

<!-- new change -->
AccuracyV1 did not materially change fixed-rollout wrist accuracy because it
trained wrist output rows while the endpoint-position error also accumulated
through shoulder roll/yaw and elbow.  The independent V2 task is:

```text
Isaac-G1-StandCoordinationAccuracyV2-100Pct-29DOF-v0
```

V2 loads the accepted `model_1375.pt` actor unchanged and adds one structured
seven-DOF adapter per arm.  Each side receives 37 values: current q, q-ref,
q-q-ref, current dq, reference dq, and motion phase sin/cos.  The original
actor trunk/output rows, waist, and legs remain frozen.  Shoulder/elbow target
corrections are limited to 0.04 rad and wrist corrections to 0.08 rad.

The zero-initialized adapter has been verified bit-for-bit equivalent to the
original model_1375 action output.  The accepted supervised bootstrap uses
256000 StandCoordination samples and the bounded alpha=0.50 feedback target.
Its pre-PPO multi-seed results and exact checkpoint paths are recorded in
`logs/stand_coordination_v1/accuracy_v2/validation_report.md`.

Only after those checks pass may the two adapters receive the configured
15-iteration PPO refinement.  The V2 runner exposes 6503 trainable parameters
per side, uses 48-step rollouts, learning rate 2e-5, action std 0.02, and saves
every three iterations.  No legacy actor or balance-policy parameter is
optimizer-visible.
