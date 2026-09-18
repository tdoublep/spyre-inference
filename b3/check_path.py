import torch  # noqa: F401  (must precede torch_spyre backend autoload)
import spyre_inference

print("spyre_inference from:", spyre_inference.__file__)
print("IS WORKTREE:", "worktrees/s3-dev2" in spyre_inference.__file__)
