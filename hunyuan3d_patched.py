#!/usr/bin/env python3

import argparse
import os
import sys

import huggingface_hub

# Patch huggingface_hub to use home directory cache
original_snapshot_download = huggingface_hub.snapshot_download


def patched_snapshot_download(*args, **kwargs):
    # Force cache_dir to be in user's home directory
    cache_dir = os.path.join(os.environ["HOME"], ".hunyuan3d_cache")
    os.makedirs(cache_dir, exist_ok=True)
    kwargs["cache_dir"] = cache_dir
    print(f"Using cache directory: {cache_dir}")
    return original_snapshot_download(*args, **kwargs)


# Apply the patch
huggingface_hub.snapshot_download = patched_snapshot_download

# Now import the models
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


def main():
    parser = argparse.ArgumentParser(description="Run Hunyuan3D 2.0")
    parser.add_argument(
        "--image", type=str, default="assets/demo.png", help="Input image path"
    )
    parser.add_argument("--output", type=str, default=None, help="Output GLB file path")
    args = parser.parse_args()

    # Set output path
    if args.output is None:
        output_dir = os.path.join(os.environ["HOME"], "hunyuan3d_outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "output.glb")
    else:
        output_path = args.output
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Input image: {args.image}")
    print(f"Output path: {output_path}")

    # Make sure input image exists
    if not os.path.exists(args.image):
        print(f"Error: Input image {args.image} does not exist")
        sys.exit(1)

    # Load only the shape generation pipeline
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")

    # Generate a mesh without texture
    mesh = pipeline(image=args.image)[0]

    # Save the mesh
    mesh.export(output_path)
    print(f"Mesh saved to {output_path}")


if __name__ == "__main__":
    main()
