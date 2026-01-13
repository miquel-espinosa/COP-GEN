from huggingface_hub import snapshot_download
import tarfile, pathlib

# Set project root directory
PROJECT_ROOT = pathlib.Path("/home/egm/Data/Projects/CopGen")
data_dir = PROJECT_ROOT / "data" / "cop-gen-small-test"

# downloads all files into ./data/cop-gen-small-test
snapshot_download(
    repo_id="mespinosami/cop-gen-small-test",
    repo_type="dataset",
    local_dir=str(data_dir)
)

# Extract all tar.gz files in the downloaded directory
for tar_path in data_dir.glob("*.tar.gz"):
    print(f"Extracting {tar_path.name} to {data_dir}/")
    
    with tarfile.open(tar_path, "r:gz") as t:
        t.extractall(data_dir)