#new change
"""Frozen 25% sagittal inference using the unchanged dual-controller checks."""
from pathlib import Path

path = Path(__file__).resolve().with_name('test_dual_sagittal_gate.py')
source = path.read_text()
needle = 'Sagittal15Cfg as NeutralDualAdaptCfg'
assert source.count(needle) == 1
source = source.replace(needle, 'Sagittal25Cfg as NeutralDualAdaptCfg')
exec(compile(source, str(path), 'exec'), {'__name__': '__main__', '__file__': str(path)})
