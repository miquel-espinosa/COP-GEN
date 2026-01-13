import logging
import sys
from datetime import datetime
from pathlib import Path
import pytz  

def setup_logging(root_path, experiment_name=None):
    """
    Set up logging configuration for experiments
    
    Args:
        root_path: Path to the project root (where logs/ will be created)
        experiment_name: Optional experiment name for log filename
    
    Returns:
        str: Path to the created log file
    """
    root_path = Path(root_path)
    
    # Create logs directory
    log_dir = Path(root_path)  # caller passes the logs dir path directly
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Create daily subdirectory
    cet = pytz.timezone('Europe/Amsterdam') 
    now = datetime.now(cet)
    date_str = now.strftime("%Y-%m-%d")
    daily_log_dir = log_dir / date_str
    daily_log_dir.mkdir(exist_ok=True, parents=True)
    
    # Create log file with timestamp
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        log_file = daily_log_dir / f"{experiment_name}_{timestamp}.log"
    else:
        log_file = daily_log_dir / f"experiment_{timestamp}.log"

    # Clear any existing handlers to avoid conflicts
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Force reconfiguration
    )
    
    logging.info(f"Logging initialized - saving to {log_file}")
    return str(log_file)