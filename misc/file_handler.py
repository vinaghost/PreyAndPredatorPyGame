import os
import shutil

def delete_folder(folder_path: str) -> None:
    try:
        shutil.rmtree(folder_path)
    except Exception as e:
        print(f'Failed to delete {folder_path}. Reason: {e}')

def create_folder(folder_path: str) -> None:
    try:
        os.makedirs(folder_path, exist_ok=True)
    except Exception as e:
        print(f'Failed to create the folder {folder_path}. Reason: {e}')

def check_folder_exists(folder_path: str) -> bool:
    return os.path.exists(folder_path)
