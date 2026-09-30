# GeneralTrackingV1 migration boundary

This project starts from the untouched SoftMimic release. The Pour experiment
tree at `/home/noah/mingkai/softmimic` remains available and is not used as the
source tree for training this controller.

## Migrated

- Strict canonical ordering for the 29 G1 body joints.
- IsaacLab's maintained G1 USD, with the compatible nominal orientation and
  actuator gains; the 14 finger joints remain outside policy tensors.
- Name-based keypoint body resolution instead of brittle USD body indices.
- 29DOF-safe reference metrics, resets, simulator writes, observations, and
  reward mappings.
- A single gate-free action equation: `q_target = q_reference + residual`.
- Raw 100% stand, walk, T-pose, box-pick, and Pour trajectories in one task.
- Dataset and simulator smoke-test scripts.

## Deliberately not migrated

- 15/35/60-percent trajectory scaling.
- Pour-only right-hand grasp targets.
- Absolute-reference gates, frozen bias, target bias, waist blend, and joint
  probes.
- Checkpoint-specific constants for model 3650/750/925/949.
- Right/left arm adapters and supervised shoulder compensation.
- Force-field and compliant augmented data. These belong to a later robustness
  stage after nominal 100% tracking passes acceptance tests.

## Task contract

- Task: `Isaac-G1-GeneralTrackingV1-29DOF-v0`
- Motions: five complete raw CSV recordings at their original 100% amplitude.
- Policy action dimension: 29.
- Isaac articulation dimension: 43 (29 body + 14 unlearned fingers).
- Episode: a random continuous 20-second segment from one motion.
- Training: scratch PPO; no Pour checkpoint is loaded.

## Training curriculum

All stages have identical policy observations, 29D actions, rewards, and
network shapes, so checkpoints transfer without adapters or partial loading.

1. `Isaac-G1-GeneralTrackingV1-StandBootstrap-29DOF-v0`: active balance.
2. `Isaac-G1-GeneralTrackingV1-LocomotionBootstrap-29DOF-v0`: concentrate
   learning on stand and full-amplitude walk.
3. `Isaac-G1-GeneralTrackingV1-Core-29DOF-v0`: stand, walk, T-pose, and 100%
   Pour.
4. `Isaac-G1-GeneralTrackingV1-29DOF-v0`: add the deep-bending box-pick motion.

This is dataset curriculum, not motion-amplitude scaling: every included
reference remains at its original 100% amplitude.

## Environment selection

The existing Python environment has the old project installed editable. To run
this project without changing that installation, prepend this project to
`PYTHONPATH`:

```bash
source /home/noah/env_isaaclab/bin/activate
cd /home/noah/mingkai/general_softmimic
export PYTHONPATH=/home/noah/mingkai/general_softmimic/softmimic_gym:/home/noah/mingkai/general_softmimic:${PYTHONPATH}
```

## First formal training run

Start from scratch on the stand bootstrap. Do not resume a checkpoint from the
Pour experiment tree and do not use the one-iteration smoke-test checkpoint.

```bash
mkdir -p logs/general_tracking

python scripts/rsl_rl/train.py \
  --task Isaac-G1-GeneralTrackingV1-StandBootstrap-29DOF-v0 \
  --num_envs 4096 \
  --max_iterations 600 \
  --run_name general_tracking_v1_stand_bootstrap_scratch \
  --seed 42 \
  --headless \
  2>&1 | tee logs/general_tracking/stand_bootstrap_scratch_console.log
```

The 600 iterations are the first acceptance gate, not a claim that the final
controller is trained. Promote a checkpoint to the Core stage only after
population-level stability and tracking diagnostics pass. Then resume that
checkpoint under the Core task; after Core passes, resume under the complete
five-motion task. The curriculum changes which full-amplitude motions are
sampled, never their amplitude or the policy interface.
