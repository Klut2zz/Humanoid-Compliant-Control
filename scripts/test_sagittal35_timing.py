#new change
"""35% trace: only substitute the audited reference configuration."""
from pathlib import Path
path=Path(__file__).resolve().with_name('test_sagittal_csv_v2_timing.py')
source=path.read_text()
assert source.count('SagittalCsvV2_25Cfg')==1
source=source.replace('SagittalCsvV2_25Cfg','SagittalCsvV2_35Cfg')
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
