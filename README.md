# SoftMimic Code Release

This repository provides an implementation of the paper:


<td style="padding:20px;width:75%;vertical-align:middle">
      <a href="https://gmargo11.github.io/softmimic/" target="_blank">
      <b> SoftMimic: Learning Compliant Whole-body Control from Examples </b>
      </a>
      <br>
      <a href="https://gmargo11.github.io/" target="_blank">Gabriel B. Margolis</a>*, <a href="https://www.linkedin.com/in/mwangjoy/" target="_blank">Michelle Wang</a>*, <a href="https://nolie-rolie.github.io/" target="_blank">Nolan Fey</a>, and <a href="https://people.csail.mit.edu/pulkitag" target="_blank">Pulkit Agrawal</a>
      <br>
      <em>preprint</em>, 2025
      <br>
      <a href="https://arxiv.org/abs/2510.17792" target="_blank">paper</a> /
      <a href="https://gmargo11.github.io/softmimic/" target="_blank">project page</a>
    <br>
</td>

<br>

If you use this repository in your work, consider citing:

```
@article{margolis2025softmimic,
    title={{SoftMimic}: Learning Compliant Whole-body Control from Examples},
    author={Margolis, Gabriel B. and Wang, Michelle and Fey, Nolan and Agrawal, Pulkit},
    journal={arXiv preprint arXiv:2510.17792},
    year={2025}
}
```

The project is distributed under the MIT License (see `LICENSE`). Third-party components such as `unitree_sdk2` retain their original licenses in `third-party/unitree_sdk2`.

Code in `unitree_sdk2` for LCM interface comes from OpenHomie: https://github.com/InternRobotics/OpenHomie which is based on walk-these-ways: https://github.com/Improbable-AI/walk-these-ways

This code depends on a number of open-source projects without which the system would not be possible:
- IsaacLab: https://github.com/isaac-sim/IsaacLab
- RSL-RL: https://github.com/leggedrobotics/rsl_rl
- Mink: https://github.com/kevinzakka/mink
- MuJoCo: https://github.com/google-deepmind/mujoco
- unitree_sdk2: https://github.com/unitreerobotics/unitree_sdk2



## 1. Installation <a name="installation"></a>

```bash
git clone https://github.com/Improbable-AI/softmimic_release_dev.git
cd softmimic_release_dev
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[all]"  # installs optional extras (mujoco, lcm, mink, seaborn)
```

Use `pip install -e softmimic_gym[isaac]` if you intend to run Isaac-based training scripts.

---

## 2. Repository Layout <a name="layout"></a>

```
softmimic_release_dev/
├── compliant_motion_augmentation/   # Compliant motion augmentation pipeline
├── pretrained_models/               # Exported policies (JIT + params)
├── scripts/                         # RSL-RL training/eval helpers
├── softmimic_deploy/                # Deployment interfaces, sensors, utils
├── softmimic_gym/                   # Isaac Lab extension + packaging
└── third-party/unitree_sdk2/        # Unitree SDK source (for deployment)
```

---

## 3. Deployment Interfaces <a name="deployment"></a>

### 3.1 MuJoCo simulation

```bash
python softmimic_deploy/src/deploy_policy_interface.py \
  --interface mujoco \
  --policy ../../pretrained_models/2025-09-26_03-54-58_StaticStand-SoftMimic/model_48000.jit \
  --motion_path stand.csv \
  --render
```

Controls (`mujoco` viewer):

- `L`: calibrate to the initial pose,
- `X`: start the policy,
- `Y`: enter damping mode,
- Arrow keys/PageUp/PageDown: joystick emulation.

The CLI accepts any motion CSV under `datasets/motions_csv/` or `SOFTMIMIC_DATA_ROOT`. Paths outside that tree can be passed explicitly.

### 3.2 Unitree G1 hardware

1. **Network setup**
   - Connect via Ethernet and assign a static IP on `192.168.123.0/24`.
   - Ensure `ssh unitree@ROBOT_IP` is reachable (default G1 `ROBOT_IP` is `192.168.123.164` and default password `123`).

2. **Repository sync**

   ```bash
   # Optional overrides: ROBOT_HOST, TARGET_DIR
   ./send_to_g1.sh
   ```

3. **On-robot environment**

   ```bash
   ssh unitree@192.168.123.164
   cd ~/softmimic_release
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
   bash Miniconda3-latest-Linux-aarch64.sh
   conda create -n softmimic python=3.10
   conda activate softmimic
   pip install -e ".[lcm]"
   sudo apt-get install liblcm-dev
   ```

4. **Unitree SDK build**

   ```bash
   cd third-party/unitree_sdk2
   mkdir -p build && cd build
   cmake ..
   make -j
   ```

5. **Deployment**

   - Terminal 1 (`./g1_control eth0`) to bridge between LCM and the Unitree SDK.
   - Terminal 2 to launch the policy:

     ```bash
     python softmimic_deploy/src/deploy_policy_interface.py \
       --interface lcm \
       --policy ../../pretrained_models/2025-09-26_03-54-58_StaticStand-SoftMimic/model_48000.jit
     ```

   - Button guide matches the physical remote (L1: initial pose, X: deploy, Y: e-stop/damping mode).

---

## 4. Training, Evaluation, and Augmentation <a name="training"></a>

### 4.1 Compliant Motion Augmentation

Visualize the augmentation process in Mujoco:
```bash
python mink_generator_ff.py interactive     --motion_path ../datasets/motions_csv/stand.csv     --force_mode forcefield
```
```bash
python mink_generator_ff.py interactive     --motion_path ../datasets/motions_csv/boxpick.csv     --force_mode collision-emulator
```

Generate the full augmented dataset:

```bash
cd compliant_motion_augmentation
./generate_all.sh  # batch-generate datasets listed in config.py
```

To generate a compliant dataset for a new csv file, simply edit `generate_all.sh`. 

We follow the CSV formatting convention of Unitree's LAFAN1 Retargeting Dataset, appending two columns that annotate the foot contact state.


### 4.2 Isaac Lab tasks (softmimic_gym)

Install the extension then register environments:

```bash
cd softmimic_gym
pip install -e ".[isaac]"
cd ..
python scripts/list_envs.py  # prints registered task IDs
```

Training example (requires Isaac Sim running or headless):

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-G1-AugmentedReference-ForceTorque-Control-VariableStiffness-Hybrid-Deployable-Mimic-v0 --headless
```

### 4.3 Policy evaluation and video export

```bash
python scripts/rsl_rl/test.py \
  --task Isaac-G1-AugmentedReference-ForceTorque-Control-VariableStiffness-Hybrid-Deployable-Mimic-v0 \
  --num_envs 1 \
  --video --video_length 2000 --headless
```

---

## 5. Configuration and Safety Notes <a name="notes"></a>

- Follow all safety guidelines given by Unitree.
- Ensure that only one instance of the deployment script runs on the robot at a time.
- This is research code provided as-is to facilitate future research -- the user assumes all liability.

