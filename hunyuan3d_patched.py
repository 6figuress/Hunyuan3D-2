#!/usr/bin/env python3

import argparse
import os
import sys

import huggingface_hub
import numpy as np
import torch
import trimesh
from PIL import Image

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


class BackgroundRemover:
    def __init__(self):
        try:
            from rembg import remove

            self.remove = remove
        except ImportError:
            print("Warning: rembg not installed. Installing it now...")
            import subprocess

            subprocess.check_call([sys.executable, "-m", "pip", "install", "rembg"])
            from rembg import remove

            self.remove = remove

    def __call__(self, image):
        if isinstance(image, Image.Image):
            return Image.fromarray(self.remove(np.array(image)))
        else:
            return self.remove(image)


class FloaterRemover:
    def __call__(self, mesh):
        # Simple implementation to remove disconnected components
        components = trimesh.graph.connected_components(mesh.face_adjacency)
        if len(components) > 1:
            # Keep only the largest component
            main_component = max(components, key=len)
            mesh = mesh.submesh([main_component], append=True)
        return mesh


class DegenerateFaceRemover:
    def __call__(self, mesh):
        # Remove degenerate faces
        valid_faces = ~(mesh.area_faces < 1e-8)  # Using area_faces instead of areas
        if not all(valid_faces):
            mesh = mesh.submesh(np.where(valid_faces)[0], append=True)
        return mesh


class FaceReducer:
    def __call__(self, mesh, target_ratio=0.5):
        if len(mesh.faces) > 10000:
            mesh = mesh.simplify_quadratic_decimation(
                int(len(mesh.faces) * target_ratio)
            )
        return mesh


def load_hunyuan_dit_pipeline(model_name):
    """Helper function to load the HunyuanDiT text-to-image pipeline"""
    try:
        from hy3dgen.text2image import HunyuanDiTPipeline

        return HunyuanDiTPipeline(model_name)
    except ImportError:
        # Fallback if the specialized pipeline is not available
        print(
            "HunyuanDiTPipeline not found, falling back to generic diffusers pipeline"
        )
        from diffusers import DiffusionPipeline

        return DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda" if torch.cuda.is_available() else "cpu")


def image_to_3d(
    image_path="assets/demo.png", output_path=None, seed=2025, texture=True
):
    """Convert an image to a 3D model using Hunyuan3D"""
    if output_path is None:
        output_dir = os.path.join(os.environ["HOME"], "hunyuan3d_outputs")
        os.makedirs(output_dir, exist_ok=True)
        mesh_path = os.path.join(output_dir, "mesh.glb")
        texture_path = os.path.join(output_dir, "texture.glb")
    else:
        mesh_path = output_path
        texture_path = (
            os.path.splitext(output_path)[0]
            + "_textured"
            + os.path.splitext(output_path)[1]
        )

    rembg = BackgroundRemover()
    model_path = "tencent/Hunyuan3D-2"

    print(f"Loading image from {image_path}")
    image = Image.open(image_path)

    if image.mode == "RGB":
        print("Removing background...")
        image = rembg(image)

    print("Loading Hunyuan3D-DiT pipeline...")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    print("Generating 3D mesh...")
    mesh = pipeline(
        image=image,
        num_inference_steps=30,
        mc_algo="mc",
        generator=torch.manual_seed(seed),
    )[0]

    print("Post-processing mesh...")
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)

    mesh.export(mesh_path)
    print(f"Saved mesh to {mesh_path}")

    if texture:
        try:
            print("Generating texture...")
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            pipeline = Hunyuan3DPaintPipeline.from_pretrained(model_path)
            textured_mesh = pipeline(mesh, image=image)
            textured_mesh.export(texture_path)
            print(f"Saved textured mesh to {texture_path}")
            return texture_path
        except Exception as e:
            print(f"Error during texturing: {e}")
            print("Please try to install texture requirements by following README.md")
            return mesh_path

    return mesh_path


def image_to_3d_fast(
    image_path="assets/demo.png", output_path=None, seed=2025, texture=True
):
    """Convert an image to a 3D model using Hunyuan3D-DiT-Fast"""
    if output_path is None:
        output_dir = os.path.join(os.environ["HOME"], "hunyuan3d_outputs")
        os.makedirs(output_dir, exist_ok=True)
        mesh_path = os.path.join(output_dir, "mesh_fast.glb")
        texture_path = os.path.join(output_dir, "texture_fast.glb")
    else:
        mesh_path = output_path
        texture_path = (
            os.path.splitext(output_path)[0]
            + "_textured"
            + os.path.splitext(output_path)[1]
        )

    rembg = BackgroundRemover()
    model_path = "tencent/Hunyuan3D-2"

    print(f"Loading image from {image_path}")
    image = Image.open(image_path)

    if image.mode == "RGB":
        print("Removing background...")
        image = rembg(image)

    print("Loading Hunyuan3D-DiT-Fast pipeline...")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path, subfolder="hunyuan3d-dit-v2-0-fast", variant="fp16"
    )

    print("Generating 3D mesh (fast mode)...")
    mesh = pipeline(
        image=image,
        num_inference_steps=30,
        mc_algo="mc",
        generator=torch.manual_seed(seed),
    )[0]

    print("Post-processing mesh...")
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)

    mesh.export(mesh_path)
    print(f"Saved mesh to {mesh_path}")

    if texture:
        try:
            print("Generating texture...")
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            pipeline = Hunyuan3DPaintPipeline.from_pretrained(model_path)
            textured_mesh = pipeline(mesh, image=image)
            textured_mesh.export(texture_path)
            print(f"Saved textured mesh to {texture_path}")
            return texture_path
        except Exception as e:
            print(f"Error during texturing: {e}")
            print("Please try to install texture requirements by following README.md")
            return mesh_path

    return mesh_path


def text_to_3d(prompt, output_path=None, seed=2025, texture=True):
    """Generate a 3D model from a text prompt"""
    if output_path is None:
        output_dir = os.path.join(os.environ["HOME"], "hunyuan3d_outputs")
        os.makedirs(output_dir, exist_ok=True)
        mesh_path = os.path.join(output_dir, "text_to_3d.glb")
        texture_path = os.path.join(output_dir, "text_to_3d_textured.glb")
    else:
        mesh_path = output_path
        texture_path = (
            os.path.splitext(output_path)[0]
            + "_textured"
            + os.path.splitext(output_path)[1]
        )

    print(f"Starting text-to-3D generation with prompt: '{prompt}'")

    rembg = BackgroundRemover()

    # Load text-to-image model
    print("Loading text-to-image model...")
    t2i = load_hunyuan_dit_pipeline(
        "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"
    )

    # Load 3D generation model
    print("Loading 3D generation model...")
    model_path = "tencent/Hunyuan3D-2"
    i23d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    # Generate image from text
    print(f"Generating image from prompt: '{prompt}'")
    image = t2i(prompt)

    # Save intermediate image
    img_output_path = os.path.splitext(mesh_path)[0] + "_generated.png"
    image.save(img_output_path)
    print(f"Saved generated image to {img_output_path}")

    # Remove background
    print("Removing background...")
    image = rembg(image)

    # Save processed image
    img_processed_path = os.path.splitext(mesh_path)[0] + "_processed.png"
    image.save(img_processed_path)
    print(f"Saved processed image to {img_processed_path}")

    # Generate 3D mesh
    print("Generating 3D mesh...")
    mesh = i23d(
        image, num_inference_steps=30, mc_algo="mc", generator=torch.manual_seed(seed)
    )[0]

    # Post-process mesh
    print("Post-processing mesh...")
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)

    mesh.export(mesh_path)
    print(f"Saved mesh to {mesh_path}")

    if texture:
        try:
            print("Generating texture...")
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            pipeline = Hunyuan3DPaintPipeline.from_pretrained(model_path)
            textured_mesh = pipeline(mesh, image=image, prompt=prompt)
            textured_mesh.export(texture_path)
            print(f"Saved textured mesh to {texture_path}")
            return texture_path
        except Exception as e:
            print(f"Error during texturing: {e}")
            print("Please try to install texture requirements by following README.md")
            return mesh_path

    return mesh_path


def texture_existing_obj(obj_path, image_path=None, prompt=None, output_path=None):
    """Texture an existing OBJ file using Hunyuan3D-Paint"""
    # Set output path
    if output_path is None:
        output_dir = os.path.join(os.environ["HOME"], "hunyuan3d_outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "textured_output.glb")

    # Input validation
    if not os.path.exists(obj_path):
        print(f"Error: Input OBJ file {obj_path} does not exist")
        sys.exit(1)

    if image_path and not os.path.exists(image_path):
        print(f"Error: Reference image {image_path} does not exist")
        sys.exit(1)

    # Load the input mesh
    try:
        mesh = trimesh.load(obj_path)
        print(
            f"Mesh loaded successfully with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces"
        )
    except Exception as e:
        print(f"Error loading mesh: {e}")
        sys.exit(1)

    # Check if the mesh has UV coordinates
    if not hasattr(mesh.visual, "uv") or mesh.visual.uv is None:
        print("Warning: The input mesh doesn't have UV coordinates.")
        print("Attempting to generate simple UV mapping...")
        try:
            # Create simple planar UV mapping if needed
            mesh = mesh.unwrap()
            print("UV mapping generated.")
        except Exception as e:
            print(f"Failed to generate UV mapping: {e}")
            print("The model may not texture correctly without UV coordinates.")

    # Load image if provided
    image = None
    if image_path:
        image = Image.open(image_path).convert("RGB")

    # Import texture pipeline
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    print("Loading Hunyuan3D-Paint pipeline...")
    pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")

    # Generate texture
    print("Generating texture...")
    if prompt:
        print(f"Using text prompt: {prompt}")
        textured_mesh = pipeline(mesh, image=image, prompt=prompt)
    else:
        textured_mesh = pipeline(mesh, image=image)

    # Save the textured mesh
    textured_mesh.export(output_path)
    print(f"Textured mesh saved to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Hunyuan3D 2.0 - Text/Image to 3D Generation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Image to 3D command
    i2d_parser = subparsers.add_parser(
        "image-to-3d", help="Generate 3D model from image"
    )
    i2d_parser.add_argument(
        "--image", type=str, default="assets/demo.png", help="Input image path"
    )
    i2d_parser.add_argument(
        "--output", type=str, default=None, help="Output GLB file path"
    )
    i2d_parser.add_argument(
        "--seed", type=int, default=2025, help="Random seed for generation"
    )
    i2d_parser.add_argument(
        "--no-texture", action="store_true", help="Skip texture generation"
    )

    # Fast Image to 3D command
    fast_parser = subparsers.add_parser(
        "fast", help="Generate 3D model from image using faster model"
    )
    fast_parser.add_argument(
        "--image", type=str, default="assets/demo.png", help="Input image path"
    )
    fast_parser.add_argument(
        "--output", type=str, default=None, help="Output GLB file path"
    )
    fast_parser.add_argument(
        "--seed", type=int, default=2025, help="Random seed for generation"
    )
    fast_parser.add_argument(
        "--no-texture", action="store_true", help="Skip texture generation"
    )

    # Text to 3D command
    t2d_parser = subparsers.add_parser(
        "text-to-3d", help="Generate 3D model from text prompt"
    )
    t2d_parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt for generation"
    )
    t2d_parser.add_argument(
        "--output", type=str, default=None, help="Output GLB file path"
    )
    t2d_parser.add_argument(
        "--seed", type=int, default=2025, help="Random seed for generation"
    )
    t2d_parser.add_argument(
        "--no-texture", action="store_true", help="Skip texture generation"
    )

    # Texture existing model command
    tex_parser = subparsers.add_parser("texture", help="Texture an existing OBJ file")
    tex_parser.add_argument(
        "--obj",
        type=str,
        default="/opt/hunyuan3d/static_model/model.obj",
        help="Path to input .obj file (default is the built-in model)",
    )
    tex_parser.add_argument(
        "--image", type=str, default=None, help="Reference image for texturing"
    )
    tex_parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional text prompt to guide texturing",
    )
    tex_parser.add_argument(
        "--output", type=str, default=None, help="Output GLB file path"
    )

    args = parser.parse_args()

    # Check if no command was given
    if args.command is None:
        # Default to fast image-to-3d with demo image
        print("No command specified. Running fast image-to-3d with demo image...")
        args.command = "fast"
        args.image = "assets/demo.png"
        args.output = None
        args.seed = 2025
        args.no_texture = False

    # Create cache directory
    os.makedirs(os.path.join(os.environ["HOME"], ".hunyuan3d_cache"), exist_ok=True)

    # Execute the selected command
    if args.command == "image-to-3d":
        image_to_3d(args.image, args.output, args.seed, not args.no_texture)
    elif args.command == "fast":
        image_to_3d_fast(args.image, args.output, args.seed, not args.no_texture)
    elif args.command == "text-to-3d":
        text_to_3d(args.prompt, args.output, args.seed, not args.no_texture)
    elif args.command == "texture":
        texture_existing_obj(args.obj, args.image, args.prompt, args.output)


if __name__ == "__main__":
    # Import required modules only at runtime
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    main()
