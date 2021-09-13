
from  pathlib import Path
import shutil
import os



submission_path = Path("submit-pytorch")
submission_path.mkdir(exist_ok=True)
submission_assets_path = submission_path / "assets"
submission_assets_path.mkdir(exist_ok=True)

weight_path = submission_assets_path / "flood_model.pt"
weight_saved_path = 'model-outputs/flood_model.pt'
shutil.copyfile(weight_saved_path,weight_path)
if os.path.exists(submission_assets_path / "torch"):
    shutil.rmtree(submission_assets_path / "torch")
shutil.copytree('/home/1796/.cache/torch',submission_assets_path / "torch")
