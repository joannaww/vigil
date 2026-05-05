import os
import sys
import logging
from dotenv import load_dotenv
from huggingface_hub import snapshot_download


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

repo_id = "joannaww/VIGIL"

scratch_dir = os.environ.get("PWD")
local_folder = os.path.join(scratch_dir, "data")

logger.info(f"Starting download of {repo_id}...")
logger.info(f"Files will be saved to: {local_folder}")

try:
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_folder,
        token=hf_token,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    logger.info(f"Success! Data is ready at: {path}")

except Exception as e:
    logger.error(f"Download failed: {e}")
