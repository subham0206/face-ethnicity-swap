import os
import base64
from io import BytesIO
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from utils import ImagenHandler, get_api_key
from model_manager import ModelManager
import tempfile
from image_processor import extract_model_features

print("DEBUG: google_integration.py loaded")

# Load environment variables
load_dotenv()

# Configure Google API with secure key retrieval
google_api_key = get_api_key("google")
if google_api_key:
    genai.configure(api_key=google_api_key)

# Initialize model manager
model_manager = ModelManager()

def generate_image_swap(
    original_image_path, 
    ethnicity=None,
    height=None, 
    body_type=None, 
    skin_color=None, 
    hairstyle=None,
    additional_features=None,
    use_predefined_model=False,
    predefined_model_id=None,
    predefined_model_gender=None,
    swap_skin_and_hair=True,  # Parameter to toggle enhanced swapping
    generate_multiple_poses=False,  # Parameter to generate multiple poses
    use_white_background=False  # Parameter to toggle white background
):
    """
    Generate a fashion model image by swapping model's face, skin tone, and optionally hairstyle,
    keeping everything else identical.
    If use_predefined_model is True, it will use a specific predefined model's features instead.
    
    Args:
        original_image_path: Path to the original image with model wearing apparel
        ethnicity: Target ethnicity for face swap
        height: Not used in face-only swap, kept for API compatibility
        body_type: Not used in face-only swap, kept for API compatibility
        skin_color: Target skin tone for face and visible skin areas
        hairstyle: Target hairstyle (only used if swap_skin_and_hair=True)
        additional_features: Any additional features to include in the face
        use_predefined_model: Whether to use a predefined model's features
        predefined_model_id: ID of the predefined model to use
        predefined_model_gender: Gender of the predefined model
        swap_skin_and_hair: Whether to also swap skin tone and hairstyle (True) or just face (False)
        generate_multiple_poses: Whether to generate the model in multiple poses/angles
        use_white_background: Whether to use white background instead of light grey
    
    Returns:
        If generate_multiple_poses=False: PIL Image object of the generated image with swapped features
        If generate_multiple_poses=True: Dictionary with PIL Image objects for different poses
    """
    try:
        # Debug logging
        print(f"DEBUG: Starting image generation with source image: {original_image_path}")
        
        # First preprocess large images to avoid API limits
        from image_processor import preprocess_image_for_api
        optimized_image_path = preprocess_image_for_api(original_image_path, max_size=2048)
        print(f"DEBUG: Using processed image: {optimized_image_path}")
        
        # Create an ImagenHandler instance with the same API key used for configuration
        imagen_handler = ImagenHandler(google_api_key)
        
        # First determine what kind of swap we're doing
        if use_predefined_model and predefined_model_id and predefined_model_gender:
            # Get predefined model info for reference in our prompt
            model_info = model_manager.get_model(predefined_model_id, predefined_model_gender)
            
            if model_info:
                print(f"Using predefined model reference: {predefined_model_gender} - {predefined_model_id} ({model_info['name']})")
                
                # Extract features from the model information
                features = extract_model_features(model_info)
                
                # Add any specific features from the model description
                model_features = model_info.get('description', '')
                
                # Add default "no tattoos" requirement to additional features
                model_instructions = f"Features similar to {model_info['name']} model: {model_features}"
                if additional_features:
                    user_instructions = f"User instructions: {additional_features}"
                else:
                    user_instructions = "No tattoos or identifying marks."
                
                full_additional_features = f"{model_instructions}. {user_instructions}"
                
                if swap_skin_and_hair:
                    # Perform complete swap using the ethnicity/features from the predefined model
                    print(f"Performing full swap (face, skin tone, and hair) with ethnicity: {features['ethnicity']}, skin tone: {features['skin_tone']}, hairstyle: {features['hairstyle']}")
                    
                    # First generate the front view
                    front_view_img = imagen_handler.swap_face_skin_hair(
                        original_model_image_path=optimized_image_path,
                        ethnicity=features['ethnicity'] or model_info['name'].split('–')[0].strip(),
                        skin_tone=features['skin_tone'] or skin_color,
                        hairstyle=features['hairstyle'],
                        additional_features=full_additional_features,
                        use_white_background=use_white_background
                    )
                    
                    if generate_multiple_poses and front_view_img:
                        # Save the front view temporarily to use as base for other poses
                        temp_path = os.path.join(tempfile.gettempdir(), f"front_view_{os.path.basename(original_image_path)}")
                        front_view_img.save(temp_path)
                        
                        print("Generating additional poses based on the front view...")
                        # Generate other poses using the front view as a base
                        all_poses = {
                            "front": front_view_img  # Add the already generated front view
                        }
                        
                        # Generate the other poses
                        other_poses = imagen_handler.generate_multiple_poses(
                            temp_path,
                            ethnicity=features['ethnicity'] or model_info['name'].split('–')[0].strip(),
                            skin_tone=features['skin_tone'] or skin_color,
                            hairstyle=features['hairstyle'],
                            additional_features=full_additional_features,
                            use_white_background=use_white_background
                        )
                        
                        # Clean up temporary files
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                            
                        # Add the other poses to the result dictionary
                        if other_poses:
                            for pose_key, pose_img in other_poses.items():
                                if pose_key != "front":  # Skip front as we already have it
                                    all_poses[pose_key] = pose_img
                        
                        # Clean up optimized image if needed
                        if optimized_image_path != original_image_path and os.path.exists(optimized_image_path):
                            try:
                                os.remove(optimized_image_path)
                            except:
                                pass
                                
                        return all_poses
                    else:
                        # Clean up optimized image if needed
                        if optimized_image_path != original_image_path and os.path.exists(optimized_image_path):
                            try:
                                os.remove(optimized_image_path)
                            except:
                                pass
                                
                        return front_view_img
                else:
                    # Perform face-only swap (original functionality)
                    print(f"Performing face-only swap with ethnicity: {features['ethnicity']}, skin tone: {features['skin_tone']}")
                    
                    front_view_img = imagen_handler.swap_face_only(
                        original_model_image_path=optimized_image_path,
                        ethnicity=features['ethnicity'] or model_info['name'].split('–')[0].strip(),
                        skin_tone=features['skin_tone'] or skin_color,
                        additional_features=full_additional_features,
                        use_white_background=use_white_background
                    )
                    
                    if generate_multiple_poses and front_view_img:
                        # Save the front view temporarily
                        temp_path = os.path.join(tempfile.gettempdir(), f"front_view_{os.path.basename(original_image_path)}")
                        front_view_img.save(temp_path)
                        
                        print("Generating additional poses based on the front view...")
                        # Generate other poses
                        all_poses = {
                            "front": front_view_img
                        }
                        
                        other_poses = imagen_handler.generate_multiple_poses(
                            temp_path,
                            ethnicity=features['ethnicity'] or model_info['name'].split('–')[0].strip(),
                            skin_tone=features['skin_tone'] or skin_color,
                            additional_features=full_additional_features,
                            use_white_background=use_white_background
                        )
                        
                        # Clean up temporary files
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                            
                        if other_poses:
                            for pose_key, pose_img in other_poses.items():
                                if pose_key != "front":
                                    all_poses[pose_key] = pose_img
                        
                        # Clean up optimized image if needed
                        if optimized_image_path != original_image_path and os.path.exists(optimized_image_path):
                            try:
                                os.remove(optimized_image_path)
                            except:
                                pass
                                
                        return all_poses
                    else:
                        # Clean up optimized image if needed
                        if optimized_image_path != original_image_path and os.path.exists(optimized_image_path):
                            try:
                                os.remove(optimized_image_path)
                            except:
                                pass
                                
                        return front_view_img
            else:
                print(f"Predefined model not found: {predefined_model_gender} - {predefined_model_id}")
                # Fall back to standard swap with provided attributes
        
        # Standard swap with specified attributes
        if swap_skin_and_hair:
            # Enhanced swap including face, skin tone, and hairstyle
            print(f"Performing enhanced swap with ethnicity: {ethnicity}, skin tone: {skin_color}, hairstyle: {hairstyle}")
            
            # Add default "no tattoos" requirement if no additional features provided
            final_additional_features = additional_features if additional_features else "No tattoos or identifying marks."
            
            front_view_img = imagen_handler.swap_face_skin_hair(
                original_model_image_path=optimized_image_path,
                ethnicity=ethnicity,
                skin_tone=skin_color,
                hairstyle=hairstyle,
                additional_features=final_additional_features,
                use_white_background=use_white_background
            )
            
            if generate_multiple_poses and front_view_img:
                # Save the front view temporarily
                temp_path = os.path.join(tempfile.gettempdir(), f"front_view_{os.path.basename(original_image_path)}")
                front_view_img.save(temp_path)
                
                print("Generating additional poses based on the front view...")
                # Generate other poses
                all_poses = {
                    "front": front_view_img
                }
                
                other_poses = imagen_handler.generate_multiple_poses(
                    temp_path,
                    ethnicity=ethnicity,
                    skin_tone=skin_color,
                    hairstyle=hairstyle,
                    additional_features=final_additional_features,
                    use_white_background=use_white_background
                )
                
                # Clean up temporary files
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
                if other_poses:
                    for pose_key, pose_img in other_poses.items():
                        if pose_key != "front":
                            all_poses[pose_key] = pose_img
                
                # Clean up optimized image if needed
                if optimized_image_path != original_image_path and os.path.exists(optimized_image_path):
                    try:
                        os.remove(optimized_image_path)
                    except:
                        pass
                        
                return all_poses
            else:
                # Clean up optimized image if needed
                if optimized_image_path != original_image_path and os.path.exists(optimized_image_path):
                    try:
                        os.remove(optimized_image_path)
                    except:
                        pass
                        
                return front_view_img
        else:
            # Original face-only swap
            print(f"Performing face-only swap with ethnicity: {ethnicity}, skin tone: {skin_color}")
            
            # Add default "no tattoos" requirement if no additional features provided
            final_additional_features = additional_features if additional_features else "No tattoos or identifying marks."
            
            front_view_img = imagen_handler.swap_face_only(
                original_model_image_path=optimized_image_path,
                ethnicity=ethnicity,
                skin_tone=skin_color,
                additional_features=final_additional_features,
                use_white_background=use_white_background
            )
            
            if generate_multiple_poses and front_view_img:
                # Save the front view temporarily
                temp_path = os.path.join(tempfile.gettempdir(), f"front_view_{os.path.basename(original_image_path)}")
                front_view_img.save(temp_path)
                
                print("Generating additional poses based on the front view...")
                # Generate other poses
                all_poses = {
                    "front": front_view_img
                }
                
                other_poses = imagen_handler.generate_multiple_poses(
                    temp_path,
                    ethnicity=ethnicity,
                    skin_tone=skin_color,
                    additional_features=final_additional_features,
                    use_white_background=use_white_background
                )
                
                # Clean up temporary files
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
                if other_poses:
                    for pose_key, pose_img in other_poses.items():
                        if pose_key != "front":
                            all_poses[pose_key] = pose_img
                
                # Clean up optimized image if needed
                if optimized_image_path != original_image_path and os.path.exists(optimized_image_path):
                    try:
                        os.remove(optimized_image_path)
                    except:
                        pass
                        
                return all_poses
            else:
                # Clean up optimized image if needed
                if optimized_image_path != original_image_path and os.path.exists(optimized_image_path):
                    try:
                        os.remove(optimized_image_path)
                    except:
                        pass
                        
                return front_view_img
        
    except Exception as e:
        # Clean up optimized image on error if needed
        try:
            if 'optimized_image_path' in locals() and optimized_image_path != original_image_path and os.path.exists(optimized_image_path):
                os.remove(optimized_image_path)
        except:
            pass
            
        print(f"Error generating image with Google AI: {str(e)}")
        return None

def change_apparel_color(
    image_path,
    swatch_path,
    apparel_type="top"
):
    print("DEBUG: Entered change_apparel_color function")
    print(f"DEBUG: change_apparel_color called with image_path={image_path}, swatch_path={swatch_path}, apparel_type={apparel_type}")
    try:
        from image_processor import preprocess_image_for_api
        optimized_image_path = preprocess_image_for_api(image_path, max_size=2048)
        print(f"DEBUG: optimized_image_path={optimized_image_path}")
        print("DEBUG: Creating ImagenHandler...")
        imagen_handler = ImagenHandler(google_api_key)
        print("DEBUG: ImagenHandler created. Calling change_apparel_color on handler...")
        color_changed_img = imagen_handler.change_apparel_color(
            image_path=optimized_image_path,
            swatch_path=swatch_path,
            apparel_type=apparel_type
        )
        print(f"DEBUG: imagen_handler.change_apparel_color returned: {type(color_changed_img)}")
        if optimized_image_path != image_path and os.path.exists(optimized_image_path):
            try:
                os.remove(optimized_image_path)
                print(f"DEBUG: Cleaned up temporary optimized image")
            except:
                pass
        return color_changed_img
    except Exception as e:
        print(f"DEBUG: Error changing apparel color with Google AI: {str(e)}")
        return None