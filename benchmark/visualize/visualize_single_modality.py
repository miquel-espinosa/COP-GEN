import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path("/home/egm/Data/Projects/CopGen")
sys.path.append(str(ROOT))

from benchmark.visualize import Visualizer
# Visualize a single file and save
InOutput = "input"
modality = "coords"
tile = "152D_133R"
visualizer = Visualizer(
    InOutput=InOutput,
    modality=modality,
    tile=tile,
    root=ROOT
)
visualizer.visualize(save=True, show=False)