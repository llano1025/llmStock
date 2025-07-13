from pathlib import Path

def get_project_root():
    """Get the project root directory"""
    # Assuming utils is in project_root/utils/
    return Path(__file__).resolve().parent.parent

def get_data_path():
    """Get the data directory path"""
    return get_project_root() / "data"

def get_save_path():
    """Get the save directory path for models"""
    return get_project_root() / "checkpoints"

def get_plot_path():
    """Get the plot directory path"""
    return get_project_root() / "plot"

def get_exc_path():
    """Get the exc directory path"""
    return get_project_root() / "exc"

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [get_data_path(), get_save_path(), get_plot_path()]
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
    return True