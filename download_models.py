#!/usr/bin/env python3

import os

import huggingface_hub


def download_model(repo_id, subfolder=None, local_dir=None):
    """Download a model from HuggingFace Hub"""
    if local_dir is None:
        local_dir = os.path.join(
            os.environ.get("HUNYUAN3D_MODELS_DIR", "/opt/hunyuan3d/models"),
            repo_id.split("/")[-1],
        )

    print(f"Downloading model {repo_id} to {local_dir}")

    try:
        path = huggingface_hub.snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            subfolder=subfolder,
            local_files_only=False,
        )
        print(f"Successfully downloaded model to {path}")
        return path
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
        download_model(**model)

    print("Model download complete!")


if __name__ == "__main__":
    main()
