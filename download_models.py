#!/usr/bin/env python3

import os

import huggingface_hub


def download_model_files(repo_id, subfolder=None, local_dir=None):
    """Download model files individually from HuggingFace Hub"""
    if local_dir is None:
        local_dir = os.path.join(
            os.environ.get("HUNYUAN3D_MODELS_DIR", "/opt/hunyuan3d/models"),
            repo_id.split("/")[-1],
        )

    if subfolder:
        local_dir = os.path.join(local_dir, subfolder)

    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading model {repo_id}/{subfolder} to {local_dir}")

    try:
        # Get the list of files in the repository or subfolder
        repo_info = huggingface_hub.repo_info(repo_id)
        files_info = huggingface_hub.list_repo_files(repo_id)

        # Filter files based on subfolder if specified
        if subfolder:
            files_to_download = [
                file for file in files_info if file.startswith(subfolder)
            ]
        else:
            files_to_download = files_info

        # Download each file individually
        for file_path in files_to_download:
            try:
                # Skip directories
                if file_path.endswith("/"):
                    continue

                # Get local path
                if subfolder and file_path.startswith(subfolder):
                    # Remove subfolder prefix to get relative path
                    relative_path = file_path[len(subfolder) :].lstrip("/")
                    local_file_path = os.path.join(local_dir, relative_path)
                else:
                    local_file_path = os.path.join(local_dir, file_path)

                # Create parent directory if it doesn't exist
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download file
                print(f"Downloading {file_path} to {local_file_path}")
                huggingface_hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    local_dir=os.path.dirname(local_file_path),
                    local_dir_use_symlinks=False,
                )
            except Exception as e:
                print(f"Error downloading file {file_path}: {e}")

        print(f"Successfully downloaded model to {local_dir}")
        return local_dir
    except Exception as e:
        print(f"Error downloading model {repo_id}: {e}")
        return None


def main():
    """Main function to download all required models"""
    # Create model directory if it doesn't exist
    models_dir = os.environ.get("HUNYUAN3D_MODELS_DIR", "/opt/hunyuan3d/models")
    os.makedirs(models_dir, exist_ok=True)

    # List of models to download
    models = [
        # Shape generation models
        {"repo_id": "tencent/Hunyuan3D-2", "subfolder": "hunyuan3d-dit-v2-0"},
        {"repo_id": "tencent/Hunyuan3D-2", "subfolder": "hunyuan3d-dit-v2-0-fast"},
        # Texture generation models
        {"repo_id": "tencent/Hunyuan3D-2", "subfolder": "hunyuan3d-paint-v2-0"},
    ]

    # Download each model
    for model in models:
        download_model_files(**model)

    print("Model download complete!")


if __name__ == "__main__":
    main()
