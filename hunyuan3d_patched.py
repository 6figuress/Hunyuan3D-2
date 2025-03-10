#!/usr/bin/env python3

import argparse
import os
import sys
import warnings

import huggingface_hub
import numpy as np
import torch
import trimesh
from PIL import Image

# Suppress warning messages
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Patch huggingface_hub to use home directory cache
original_snapshot_download = huggingface_hub.snapshot_download


def patched_snapshot_download(*args, **kwargs):
    # Force cache_dir to be in user's home directory or a writable location
    if "HF_HOME" in os.environ:
        cache_dir = os.environ["HF_HOME"]
    else:
        cache_dir = os.path.join(os.environ["HOME"], ".hunyuan3d_cache")

    # Ensure the directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Set cache_dir in kwargs
    kwargs["cache_dir"] = cache_dir
    print(f"Using cache directory: {cache_dir}")

    # Call the original function
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


class SafeMeshProcessor:
    """Base class for safe mesh processing operations"""

    def __call__(self, mesh):
        try:
            return self._process(mesh)
        except Exception as e:
            print(f"Error in {self.__class__.__name__}: {e}")
            print("Returning original mesh")
            return mesh

    def _process(self, mesh):
        # To be implemented by subclasses
        return mesh


class FloaterRemover(SafeMeshProcessor):
    def _process(self, mesh):
        # Check if mesh has face_adjacency attribute
        if not hasattr(mesh, "face_adjacency"):
            print("Warning: Mesh doesn't have face_adjacency attribute")
            return mesh

        try:
            components = trimesh.graph.connected_components(mesh.face_adjacency)
            if len(components) > 1:
                main_component = max(components, key=len)
                return mesh.submesh([main_component], append=True)
            return mesh
        except Exception as e:
            print(f"Error in connected components calculation: {e}")
            return mesh


class DegenerateFaceRemover(SafeMeshProcessor):
    def _process(self, mesh):
        # Check if mesh has necessary attributes
        if not hasattr(mesh, "area_faces"):
            print("Warning: Mesh doesn't have area_faces attribute")
            if hasattr(mesh, "faces") and hasattr(mesh, "vertices"):
                # Try to compute face areas manually
                try:
                    areas = []
                    for face in mesh.faces:
                        verts = mesh.vertices[face]
                        # Simple triangle area calculation
                        v0, v1, v2 = verts
                        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                        areas.append(area)
                    valid_faces = np.array(areas) >= 1e-8
                except Exception as e:
                    print(f"Could not compute face areas: {e}")
                    return mesh
            else:
                return mesh
        else:
            # Use the built-in face areas
            valid_faces = mesh.area_faces >= 1e-8

        # Check if we have any valid faces
        if np.count_nonzero(valid_faces) == 0:
            print("Warning: No valid faces found")
            return mesh

        # Get indices of valid faces
        try:
            valid_indices = np.where(valid_faces)[0]
            if len(valid_indices) == 0:
                return mesh

            # Create submesh with only valid faces
            return mesh.submesh(valid_indices, append=True)
        except Exception as e:
            print(f"Error in submesh creation: {e}")
            return mesh


class FaceReducer(SafeMeshProcessor):
    def _process(self, mesh):
        target_ratio = 0.5

        # Check if mesh has simplify_quadratic_decimation method
        if not hasattr(mesh, "simplify_quadratic_decimation"):
            print("Warning: Mesh doesn't have simplify_quadratic_decimation method")
            return mesh

        # Skip if mesh has too few faces
        if len(mesh.faces) <= 10000:
            return mesh

        try:
            return mesh.simplify_quadratic_decimation(
                int(len(mesh.faces) * target_ratio)
            )
        except Exception as e:
            print(f"Error in mesh simplification: {e}")
            return mesh


def safe_post_process(mesh):
    """Safely post-process a mesh, handling all errors"""
    print("Post-processing mesh...")
    try:
        # Skip if mesh is invalid
        if mesh is None:
            print("Error: Mesh is None")
            return mesh

        # Check if mesh has basic attributes
        if not hasattr(mesh, "faces") or not hasattr(mesh, "vertices"):
            print("Error: Mesh is missing basic attributes (faces or vertices)")
            return mesh

        # Apply post-processing steps
        print("1. Removing floating components...")
        mesh = FloaterRemover()(mesh)

        print("2. Removing degenerate faces...")
        mesh = DegenerateFaceRemover()(mesh)

        print("3. Reducing face count...")
        mesh = FaceReducer()(mesh)

        return mesh
    except Exception as e:
        print(f"Error during mesh post-processing: {e}")
        print("Continuing with original mesh...")
        return mesh


def load_hunyuan_dit_pipeline(model_name):
    """Helper function to load the HunyuanDiT text-to-image pipeline"""
    try:
        from hy3dgen.text2image import HunyuanDiTPipeline

        print(f"Loading HunyuanDiTPipeline from {model_name}")
        return HunyuanDiTPipeline(model_name)
    except ImportError:
        # Fallback if the specialized pipeline is not available
        print(
            "HunyuanDiTPipeline not found, falling back to generic diffusers pipeline"
        )
        from diffusers import DiffusionPipeline, StableDiffusionXLPipeline

        print("Loading StableDiffusionXL as fallback")
        try:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            # Create a wrapper function to make output consistent with HunyuanDiTPipeline
            def wrapped_pipeline(prompt):
                return pipeline(prompt)

            return wrapped_pipeline
        except Exception as e:
            print(f"Error loading StableDiffusionXL: {e}")
            print("Falling back to standard Stable Diffusion")

            pipeline = DiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            def wrapped_pipeline(prompt):
                return pipeline(prompt)

            return wrapped_pipeline


def find_model_path(model_name="tencent/Hunyuan3D-2", subfolder=None):
    """Find a valid model path from several possible locations"""
    # Check for a pre-downloaded model
    if "HUNYUAN3D_MODELS_DIR" in os.environ:
        base_dir = os.environ["HUNYUAN3D_MODELS_DIR"]
        model_dir = os.path.join(base_dir, model_name.split("/")[-1])
        if subfolder:
            model_dir = os.path.join(model_dir, subfolder)
        if os.path.exists(model_dir):
            return model_dir

    # Check standard locations
    cache_root = os.environ.get(
        "HF_HOME", os.path.join(os.environ["HOME"], ".hunyuan3d_cache")
    )
    potential_paths = [
        # Common Huggingface Hub cache locations
        os.path.join(cache_root, "hub"),
        os.path.join(
            cache_root, "models--" + model_name.replace("/", "--"), "snapshots"
        ),
        os.path.join(os.environ["HOME"], ".cache", "huggingface", "hub"),
        os.path.join("/opt/hunyuan3d/models"),
        os.path.join("/tmp/huggingface_cache"),
    ]

    # Try to find any existing path
    for path in potential_paths:
        if os.path.exists(path):
            print(f"Found potential model path: {path}")
            if subfolder:
                subpath = os.path.join(path, subfolder)
                if os.path.exists(subpath):
                    return subpath
            return path

    # If all else fails, return the model name and let huggingface_hub handle it
    print(f"No local model found, will try to download from HuggingFace: {model_name}")
    return model_name


def image_to_3d(
    image_path="assets/demo.png", output_path=None, seed=2025, texture=True
):
    """Convert an image to a 3D model using Hunyuan3D"""
    # Setup output paths
    if output_path is None:
        if "HUNYUAN3D_OUTPUT_DIR" in os.environ:
            output_dir = os.environ["HUNYUAN3D_OUTPUT_DIR"]
        else:
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

    # Initialize background remover
    rembg = BackgroundRemover()

    # Load the image
    print(f"Loading image from {image_path}")
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Using a blank image instead")
        image = Image.new("RGB", (512, 512), color="white")

    # Remove background if needed
    if image.mode == "RGB":
        print("Removing background...")
        try:
            image = rembg(image)
        except Exception as e:
            print(f"Error removing background: {e}")
            print("Continuing with original image")

    # Load shape generation model
    print("Loading Hunyuan3D-DiT pipeline...")
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        model_path = find_model_path("tencent/Hunyuan3D-2")
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading shape generation model: {e}")
        sys.exit(1)

    # Generate 3D mesh
    print("Generating 3D mesh...")
    try:
        mesh = pipeline(
            image=image,
            num_inference_steps=30,
            mc_algo="mc",
            generator=torch.manual_seed(seed),
        )[0]
    except Exception as e:
        print(f"Error generating mesh: {e}")
        sys.exit(1)

    # Post-process the mesh safely
    mesh = safe_post_process(mesh)

    # Save the untextured mesh
    try:
        mesh.export(mesh_path)
        print(f"Saved mesh to {mesh_path}")
    except Exception as e:
        print(f"Error saving mesh: {e}")
        sys.exit(1)

    # Generate texture if requested
    if texture:
        try:
            print("Generating texture...")
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            # Try to find the texture model locally first
            texture_model_path = find_model_path(
                "tencent/Hunyuan3D-2", "hunyuan3d-paint-v2-0"
            )
            print(f"Loading texture model from: {texture_model_path}")

            pipeline = Hunyuan3DPaintPipeline.from_pretrained(texture_model_path)
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
    # Setup output paths
    if output_path is None:
        if "HUNYUAN3D_OUTPUT_DIR" in os.environ:
            output_dir = os.environ["HUNYUAN3D_OUTPUT_DIR"]
        else:
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

    # Initialize background remover
    rembg = BackgroundRemover()

    # Load the image
    print(f"Loading image from {image_path}")
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Using a blank image instead")
        image = Image.new("RGB", (512, 512), color="white")

    # Remove background if needed
    if image.mode == "RGB":
        print("Removing background...")
        try:
            image = rembg(image)
        except Exception as e:
            print(f"Error removing background: {e}")
            print("Continuing with original image")

    # Load fast shape generation model
    print("Loading Hunyuan3D-DiT-Fast pipeline...")
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        model_path = find_model_path("tencent/Hunyuan3D-2")
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path, subfolder="hunyuan3d-dit-v2-0-fast", variant="fp16"
        )
    except Exception as e:
        print(f"Error loading fast shape generation model: {e}")
        sys.exit(1)

    # Generate 3D mesh
    print("Generating 3D mesh (fast mode)...")
    try:
        mesh = pipeline(
            image=image,
            num_inference_steps=30,
            mc_algo="mc",
            generator=torch.manual_seed(seed),
        )[0]
    except Exception as e:
        print(f"Error generating mesh: {e}")
        sys.exit(1)

    # Post-process the mesh safely
    mesh = safe_post_process(mesh)

    # Save the untextured mesh
    try:
        mesh.export(mesh_path)
        print(f"Saved mesh to {mesh_path}")
    except Exception as e:
        print(f"Error saving mesh: {e}")
        sys.exit(1)

    # Generate texture if requested
    if texture:
        try:
            print("Generating texture...")
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            # Try to find the texture model locally first
            texture_model_path = find_model_path(
                "tencent/Hunyuan3D-2", "hunyuan3d-paint-v2-0"
            )
            print(f"Loading texture model from: {texture_model_path}")

            pipeline = Hunyuan3DPaintPipeline.from_pretrained(texture_model_path)
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
    # Setup output paths
    if output_path is None:
        if "HUNYUAN3D_OUTPUT_DIR" in os.environ:
            output_dir = os.environ["HUNYUAN3D_OUTPUT_DIR"]
        else:
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

    # Initialize background remover
    rembg = BackgroundRemover()

    # Load text-to-image model
    print("Loading text-to-image model...")
    try:
        t2i = load_hunyuan_dit_pipeline(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"
        )
    except Exception as e:
        print(f"Error loading text-to-image model: {e}")
        sys.exit(1)

    # Load 3D generation model
    print("Loading 3D generation model...")
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        model_path = find_model_path("tencent/Hunyuan3D-2")
        i23d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading 3D generation model: {e}")
        sys.exit(1)

    # Generate image from text
    print(f"Generating image from prompt: '{prompt}'")
    try:
        output = t2i(prompt)
    except Exception as e:
        print(f"Error generating image from text: {e}")
        print("Using a blank image instead")
        output = Image.new("RGB", (512, 512), color="white")

    # Handle different return types from different text-to-image models
    if hasattr(output, "images"):
        # For StableDiffusionXL and similar models that return a pipeline output object
        image = output.images[0]
    elif isinstance(output, list) and isinstance(output[0], Image.Image):
        # For models that return a list of images
        image = output[0]
    elif isinstance(output, Image.Image):
        # For models like HunyuanDiT that return a single image directly
        image = output
    else:
        # Try to handle any other unexpected return type
        print(
            f"Warning: Unexpected return type from text-to-image model: {type(output)}"
        )
        if hasattr(output, "__getitem__"):
            try:
                image = output[0]
                if not isinstance(image, Image.Image):
                    raise ValueError(f"Expected PIL Image, got {type(image)}")
            except (IndexError, TypeError, ValueError) as e:
                print(f"Error extracting image from model output: {e}")
                print("Using default image instead.")
                image = Image.new("RGB", (512, 512), color="white")
        else:
            print("Using default image instead.")
            image = Image.new("RGB", (512, 512), color="white")

    # Save intermediate image
    img_output_path = os.path.splitext(mesh_path)[0] + "_generated.png"
    try:
        image.save(img_output_path)
        print(f"Saved generated image to {img_output_path}")
    except Exception as e:
        print(f"Error saving generated image: {e}")

    # Remove background
    print("Removing background...")
    try:
        image = rembg(image)
    except Exception as e:
        print(f"Error removing background: {e}")
        print("Continuing with original image")

    # Save processed image
    img_processed_path = os.path.splitext(mesh_path)[0] + "_processed.png"
    try:
        image.save(img_processed_path)
        print(f"Saved processed image to {img_processed_path}")
    except Exception as e:
        print(f"Error saving processed image: {e}")

    # Generate 3D mesh
    print("Generating 3D mesh...")
    try:
        mesh = i23d(
            image,
            num_inference_steps=30,
            mc_algo="mc",
            generator=torch.manual_seed(seed),
        )[0]
    except Exception as e:
        print(f"Error generating mesh: {e}")
        sys.exit(1)

    # Post-process the mesh safely
    mesh = safe_post_process(mesh)

    # Save the untextured mesh
    try:
        mesh.export(mesh_path)
        print(f"Saved mesh to {mesh_path}")
    except Exception as e:
        print(f"Error saving mesh: {e}")
        sys.exit(1)

    # Generate texture if requested
    if texture:
        try:
            print("Generating texture...")
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            # Try to find the texture model locally first
            texture_model_path = find_model_path(
                "tencent/Hunyuan3D-2", "hunyuan3d-paint-v2-0"
            )
            print(f"Loading texture model from: {texture_model_path}")

            pipeline = Hunyuan3DPaintPipeline.from_pretrained(texture_model_path)
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


def texture_local_model(model_path="rubber_duck.obj",
                       texture_image="duck_texture.png",
                       output_path="rubber_duck_textured.glb"):
    """Texture a local 3D model with a specific image"""

    print(f"Loading model from {model_path}")
    try:
        mesh = trimesh.load(model_path)
        print(f"Model loaded successfully with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    print(f"Loading texture image from {texture_image}")
    try:
        image = Image.open(texture_image)
        image = image.convert("RGB")
    except Exception as e:
        print(f"Error loading texture image: {e}")
        return None

    print("Loading Hunyuan3D-Paint pipeline...")
    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
    except Exception as e:
        print(f"Error loading texture pipeline: {e}")
        return None

    print("Applying texture...")
    try:
        textured_mesh = pipeline(mesh, image=image)
        textured_mesh.export(output_path)
        print(f"Textured model saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error during texturing: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Hunyuan3D 2.0 - Text/Image to 3D Generation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    local_tex_parser = subparsers.add_parser("texture-local",
        help="Texture a local model with a specific image")
    local_tex_parser.add_argument(
        "--model",
        type=str,
        default="rubber_duck.obj",
        help="Path to input 3D model file"
    )
    local_tex_parser.add_argument(
        "--texture",
        type=str,
        default="duck_texture.png",
        help="Path to texture image"
    )
    local_tex_parser.add_argument(
        "--output",
        type=str,
        default="rubber_duck_textured.glb",
        help="Output path for textured model"
    )

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
    elif args.command == "texture-local":
        texture_local_model(args.model, args.texture, args.output)


if __name__ == "__main__":
    # Import required modules only at runtime

    main()
