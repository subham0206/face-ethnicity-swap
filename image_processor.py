import os
import tempfile
from PIL import Image
import uuid
import numpy as np
import math
import hashlib

def save_uploaded_image(uploaded_file, max_size_mb=10, max_dimension=2048):
    """
    Save an uploaded image file to a temporary directory and resize if too large
    
    Args:
        uploaded_file: Streamlit uploaded file object
        max_size_mb: Maximum file size in MB before resize is enforced
        max_dimension: Maximum width/height dimension for resized images
    
    Returns:
        Path to the saved image file
    """
    try:
        # Create a temporary directory if it doesn't exist
        temp_dir = os.path.join(tempfile.gettempdir(), "face_ethnicity_swap")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate a unique filename
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        filename = f"{uuid.uuid4().hex}{file_extension}"
        file_path = os.path.join(temp_dir, filename)
        
        # Calculate file size in MB
        file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
        print(f"DEBUG: Uploaded file size: {file_size_mb:.2f} MB")
        
        # Check if the image is large and needs resizing
        if file_size_mb > max_size_mb:
            print(f"DEBUG: Large image detected ({file_size_mb:.2f} MB). Resizing before processing.")
            
            # For large images, load into PIL, resize, and save
            img = Image.open(uploaded_file)
            img = resize_image(img, max_size=max_dimension)
            
            # Save the resized image
            img.save(file_path, quality=90, optimize=True)
            print(f"DEBUG: Image resized and saved to {file_path}")
            
            # Get new file size for logging
            new_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"DEBUG: Resized image size: {new_size_mb:.2f} MB")
        else:
            # For smaller images, save directly
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        print(f"Error saving uploaded image: {str(e)}")
        return None

def resize_image(image, max_size=1024):
    """
    Resize an image while maintaining aspect ratio
    
    Args:
        image: PIL Image object
        max_size: Maximum width or height
    
    Returns:
        Resized PIL Image object
    """
    width, height = image.size
    
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image

def save_result_image(image, output_dir=None):
    """
    Save a generated image to disk with maximum quality preservation
    
    Args:
        image: PIL Image object
        output_dir: Directory to save the image (uses temp directory if None)
    
    Returns:
        Path to the saved image file
    """
    try:
        # Create output directory if it doesn't exist or use temp directory
        if output_dir is None:
            output_dir = os.path.join(tempfile.gettempdir(), "face_ethnicity_swap", "output")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a unique filename
        filename = f"{uuid.uuid4().hex}.png"
        file_path = os.path.join(output_dir, filename)
        
        # Save the image with maximum quality settings for PNG
        image.save(file_path, format="PNG", compress_level=1)
        
        return file_path
    except Exception as e:
        print(f"Error saving result image: {str(e)}")
        return None

def save_multiple_pose_images(pose_images, output_dir=None):
    """
    Save a dictionary of pose images to disk with maximum quality
    
    Args:
        pose_images: Dictionary of PIL Image objects with pose keys
        output_dir: Directory to save the images (uses temp directory if None)
    
    Returns:
        Dictionary of paths to the saved image files, keyed by pose
    """
    try:
        if not pose_images:
            return {}
            
        # Create output directory if it doesn't exist or use temp directory
        if output_dir is None:
            output_dir = os.path.join(tempfile.gettempdir(), "face_ethnicity_swap", "output")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a base unique ID for this set of poses
        base_id = uuid.uuid4().hex
        
        result_paths = {}
        
        # Save each pose with a consistent naming pattern
        for pose_key, image in pose_images.items():
            filename = f"{base_id}_{pose_key}.png"
            file_path = os.path.join(output_dir, filename)
            
            # Save the image with maximum quality settings
            image.save(file_path, format="PNG", compress_level=1)
            result_paths[pose_key] = file_path
        
        return result_paths
    except Exception as e:
        print(f"Error saving multiple pose images: {str(e)}")
        return {}

def hash_file_path(file_path):
    """
    Create a simple hash of a file path to use in reference filenames
    
    Args:
        file_path: Path to hash
        
    Returns:
        String hash representation
    """
    return hashlib.md5(file_path.encode()).hexdigest()[:10]

def save_color_changed_image(image, color_name, original_path, output_dir=None):
    """
    Save a color-changed image to disk with maximum quality preservation
    
    Args:
        image: PIL Image object with changed color
        color_name: Name of the color used
        original_path: Path to the original image file (used to derive name)
        output_dir: Directory to save the image (uses temp directory if None)
    
    Returns:
        Path to the saved image file
    """
    try:
        # Create output directory if it doesn't exist or use temp directory
        if output_dir is None:
            output_dir = os.path.join(tempfile.gettempdir(), "face_ethnicity_swap", "output")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a unique but descriptive filename
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        filename = f"{base_name}_{color_name}_{uuid.uuid4().hex[:8]}.png"
        file_path = os.path.join(output_dir, filename)
        
        # Save the image with maximum quality settings
        image.save(file_path, format="PNG", compress_level=1)
        
        return file_path
    except Exception as e:
        print(f"Error saving color-changed image: {str(e)}")
        return None

def get_last_color_changed_image(original_image_path=None):
    """
    Get the path to the last color-changed image
    
    Args:
        original_image_path: Path to the original image to find its color-changed version (not used)
        
    Returns:
        Path to the last color-changed image or None
    """
    # Simple implementation that doesn't try to match with original image
    # Just return None to reset the color changed image every time
    return None

def extract_model_features(model_info):
    """
    Extract model features from a model dictionary
    
    Args:
        model_info: Dictionary with model information
    
    Returns:
        Dictionary with extracted features (ethnicity, skin_tone, hairstyle)
    """
    features = {
        "ethnicity": None,
        "skin_tone": None,
        "hairstyle": None
    }
    
    if model_info:
        # Try to extract ethnicity from name (usually the first part before any dash)
        if "name" in model_info:
            name_parts = model_info["name"].split("–")
            if len(name_parts) > 0:
                features["ethnicity"] = name_parts[0].strip()
        
        # Try to extract skin tone and hairstyle from description
        if "description" in model_info:
            desc = model_info["description"].lower()
            
            # Extract skin tone
            skin_tones = ["fair", "light", "medium", "tan", "olive", "brown", "dark"]
            for tone in skin_tones:
                if tone in desc:
                    features["skin_tone"] = tone
                    break
            
            # Extract hairstyle
            hair_descriptors = ["curly", "straight", "wavy", "braids", "buzz", "fade", "bob", "pixie", "afro", "dreads"]
            for hair in desc:
                if hair in desc:
                    features["hairstyle"] = desc.split(hair)[0].split()[-1] + " " + hair
                    break
    
    return features

def is_product_only_image(image_path):
    """
    Determine if an image is a product-only image (no model) or a model wearing apparel
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Boolean: True if it's a product-only image, False if it likely contains a model
    """
    try:
        # For now, we'll rely on the user to specify this in the UI
        # In a more advanced implementation, this could use AI to detect human figures
        # or be based on user selection in the UI
        return True
    except Exception as e:
        print(f"Error detecting image type: {str(e)}")
        return True  # Default to product-only for safety

def preprocess_image_for_api(image_path, max_size=3072):
    """
    Preprocess an image before sending to AI API while preserving maximum quality.
    Only resizes dimensions if absolutely necessary to meet API requirements.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height) for the image
        
    Returns:
        Path to the optimized image (may be the same as input if no changes needed)
    """
    try:
        # Check file size and dimensions
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        
        # Open the image to check dimensions
        img = Image.open(image_path)
        width, height = img.size
        
        # Only resize if dimensions exceed max_size (necessary for API processing)
        if width > max_size or height > max_size:
            print(f"DEBUG: Image dimensions ({width}x{height}) exceed API limits. Resizing while preserving quality...")
            
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            # Use high-quality downsampling for resizing
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"DEBUG: Image resized to {new_width}x{new_height} using high-quality resampling")
            
            # Create a new optimized version with a different filename
            temp_dir = os.path.dirname(image_path)
            new_filename = f"hq_optimized_{os.path.basename(image_path)}"
            optimized_path = os.path.join(temp_dir, new_filename)
            
            # Save with highest quality settings but as PNG for better quality
            if optimized_path.lower().endswith(('.jpg', '.jpeg')):
                # Convert to PNG for better quality
                optimized_path = optimized_path.rsplit('.', 1)[0] + '.png'
                
            # Save with maximum quality
            img.save(optimized_path, format="PNG", compress_level=1)
            
            new_size_mb = os.path.getsize(optimized_path) / (1024 * 1024)
            print(f"DEBUG: High-quality image saved: {new_size_mb:.2f}MB with dimensions {new_width}x{new_height}")
            
            return optimized_path
        
        # If no resizing was needed, return original path
        return image_path
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        # Return original path if any error occurs
        return image_path