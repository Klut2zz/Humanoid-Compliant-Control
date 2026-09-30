import os
import torch
from rsl_rl.runners import OnPolicyRunner
# from isaaclab_rl.rsl_rl import export_policy_as_jit
# from rsl_rl.modules import StudentTeacher
# new change
# Current rsl_rl exposes policy export on OnPolicyRunner and no longer exports
# StudentTeacher from rsl_rl.modules. This runner does not use StudentTeacher.

class JitSavingOnPolicyRunner(OnPolicyRunner):
    """Custom OnPolicyRunner that also saves JIT-compiled policies during training."""
    
    def save(self, path: str, infos=None):
        """Overridden save method that also saves a JIT-compiled version of the policy."""
        # Call the original save method to save the model normally
        super().save(path, infos)
        
        # # Also save a JIT-compiled version of the policy
        self.save_jit_policy(path)

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """Overridden log method to include std annealing."""
        # Call the original log method
        super().log(locs, width, pad)
      
    def save_jit_policy(self, path: str):
        """Save a JIT-compiled version of the policy."""
        # Get the base path without extension

        # Get the parent path
        jit_path = os.path.dirname(path)
        # jit_name = f"policy_{self.alg.iteration}.jit"
        jit_name = os.path.basename(path).replace(".pt", ".jit")
        
        # Switch to eval mode for tracing
        # self.eval_mode()
        
        exported_files_to_log = None
        
        # new change
        self.export_policy_to_jit(jit_path, filename=jit_name)
            
        print(f"Saved JIT-compiled policy to {jit_path}/{jit_name}")
        
        # Upload JIT model to external logging service if applicable
        if hasattr(self, 'writer') and self.logger_type in ["neptune", "wandb"]:
            
            if not exported_files_to_log:
                 self.writer.save_file(jit_path + "/" + jit_name)
            else:
                print(f"Uploading {len(exported_files_to_log)} JIT file(s) to logger...")
                for jit_file in exported_files_to_log:
                    if os.path.exists(jit_file): # Double check existence
                        self.writer.save_file(jit_file)
                    else:
                        print(f"  Warning: File {jit_file} not found for upload.")
        
        # Switch back to previous mode (train mode if we were in it)
        # new change
        self.alg.train_mode()
