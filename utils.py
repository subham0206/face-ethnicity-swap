import os
from dotenv import load_dotenv
import openai
import google.generativeai as genai
import base64
from PIL import Image
from io import BytesIO
import requests
import time
import streamlit as st

# Load environment variables
load_dotenv()

def check_api_keys():
    """
    Check if API keys are available and valid from environment variables or Streamlit secrets
    
    Returns:
        Dictionary with availability status for each API
    """
    result = {
        "openai": False,
        "google": False
    }
    
    # Check OpenAI API key - first try Streamlit secrets, then environment variables
    openai_key = None
    
    # Try getting from Streamlit secrets
    try:
        if 'OPENAI_API_KEY' in st.secrets:
            openai_key = st.secrets['OPENAI_API_KEY']
    except:
        pass
        
    # If not found, try environment variable
    if not openai_key:
        openai_key = os.getenv("OPENAI_API_KEY")
    
    if (openai_key and openai_key.startswith("sk-")):
        result["openai"] = True
    
    # Check Google API key - first try Streamlit secrets, then environment variables
    google_key = None
    
    # Try getting from Streamlit secrets
    try:
        if 'GOOGLE_API_KEY' in st.secrets:
            google_key = st.secrets['GOOGLE_API_KEY']
    except:
        pass
        
    # If not found, try environment variable
    if not google_key:
        google_key = os.getenv("GOOGLE_API_KEY")
    
    if google_key:
        result["google"] = True
    
    return result

# New function to get API keys from either Streamlit secrets or environment variables
def get_api_key(provider):
    """
    Get the API key for the specified provider from Streamlit secrets or environment variables
    
    Args:
        provider: The API provider ('openai' or 'google')
        
    Returns:
        API key as string or None if not found
    """
    key_name = f"{provider.upper()}_API_KEY"
    api_key = None
    
    # Try getting from Streamlit secrets
    try:
        if key_name in st.secrets:
            api_key = st.secrets[key_name]
    except:
        pass
    
    # If not found, try environment variable
    if not api_key:
        api_key = os.getenv(key_name)
    
    return api_key

def get_ethnicity_options():
    """
    Get a list of ethnicity options for the app
    
    Returns:
        List of ethnicity options
    """
    return [
        "Asian (East Asian)",
        "Asian (South Asian)",
        "Asian (Southeast Asian)",
        "Black/African",
        "Caucasian/White",
        "Hispanic/Latino",
        "Middle Eastern",
        "Native American",
        "Pacific Islander",
        "Mixed/Multiracial"
    ]

def get_height_options():
    """
    Get height options for the app
    
    Returns:
        List of height options
    """
    return [
        "5'0\" (152 cm)",
        "5'2\" (158 cm)",
        "5'4\" (163 cm)",
        "5'6\" (168 cm)",
        "5'8\" (173 cm)",
        "5'10\" (178 cm)",
        "6'0\" (183 cm)",
        "6'2\" (188 cm)",
        "6'4\" (193 cm)"
    ]

def get_body_type_options():
    """
    Get body type options for the app
    
    Returns:
        List of body type options
    """
    return [
        "Slim",
        "Athletic",
        "Average",
        "Curvy",
        "Plus-size"
    ]

def get_skin_tone_options():
    """
    Get skin tone options for the app
    
    Returns:
        List of skin tone options
    """
    return [
        "Very fair",
        "Fair",
        "Light",
        "Medium",
        "Olive",
        "Tan",
        "Brown",
        "Dark brown",
        "Very dark"
    ]

def get_hairstyle_options():
    """
    Get hairstyle options for the app
    
    Returns:
        List of hairstyle options
    """
    return [
        "Short straight",
        "Medium straight",
        "Long straight",
        "Short wavy",
        "Medium wavy",
        "Long wavy",
        "Short curly",
        "Medium curly",
        "Long curly",
        "Afro",
        "Braids",
        "Dreadlocks",
        "Bald",
        "Buzzcut",
        "Pixie cut",
        "Bob cut"
    ]

class ImagenHandler:
    """
    Handler for Google's Imagen API to generate images
    """
    
    def __init__(self, api_key, timeout=60, imagen_model="imagegeneration@002", gemini_image_model="gemini-2.5-flash-image-preview"):
        """
        Initialize the Imagen handler
        
        Args:
            api_key: Google API key
            timeout: Timeout for API requests in seconds
            imagen_model: Model name for image generation
            gemini_image_model: Model name for image editing
        """
        self.api_key = api_key
        self.timeout = timeout
        self.imagen_model = imagen_model
        self.gemini_image_model = gemini_image_model
        
        # Configure Google API
        genai.configure(api_key=api_key)
    
    def generate_image_with_api(self, prompt):
        """
        Generate an image using the Google Imagen API
        
        Args:
            prompt: Text prompt for image generation
            
        Returns:
            PIL Image object or None if generation failed
        """
        try:
            # Create the generation model
            model = genai.GenerativeModel(self.imagen_model)
            
            # Generate the image
            response = model.generate_content(prompt)
            
            # Process the response
            if hasattr(response, 'candidates') and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                        image_bytes = part.inline_data.data
                        return Image.open(BytesIO(image_bytes))
            
            return None
        
        except Exception as e:
            print(f"Error generating image with Google API: {str(e)}")
            return None
    
    def swap_apparel_onto_model(self, apparel_image_path, model_image_path):
        """
        Swap apparel from one image onto a model in another image
        
        Args:
            apparel_image_path: Path to image containing the apparel
            model_image_path: Path to image containing the target model
            
        Returns:
            PIL Image object with the apparel swapped onto the model
        """
        try:
            # Load the images
            apparel_img = Image.open(apparel_image_path)
            model_img = Image.open(model_image_path)
            
            # Convert images to bytes
            apparel_buffer = BytesIO()
            apparel_img.save(apparel_buffer, format="PNG")
            apparel_bytes = apparel_buffer.getvalue()
            
            model_buffer = BytesIO()
            model_img.save(model_buffer, format="PNG")
            model_bytes = model_buffer.getvalue()
            
            # Create the prompt for apparel swapping
            prompt = """
            Take the apparel item from the first image and place it on the model in the second image.
            
            MOST IMPORTANT REQUIREMENTS:
            - Preserve ALL details of the apparel including exact texture, patterns, colors, fabric details, stitching
            - Make sure the apparel fits naturally on the model
            - Maintain the model's pose and proportions
            - Keep the background and overall composition of the model image
            - The apparel should look exactly like the original, just placed on the new model
            
            Generate a high-quality professional fashion photograph.
            """
            
            # Try with Gemini for image editing
            flash_model = genai.GenerativeModel(self.gemini_image_model)
            response = flash_model.generate_content([
                prompt,
                {"mime_type": "image/png", "data": apparel_bytes},
                {"mime_type": "image/png", "data": model_bytes}
            ])
            
            # Process the response
            if hasattr(response, 'candidates') and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                        image_bytes = part.inline_data.data
                        return Image.open(BytesIO(image_bytes))
            
            raise Exception("No image was returned in the response")
        
        except Exception as e:
            print(f"Error swapping apparel onto model: {str(e)}")
            return None
            
    def swap_face_only(self, original_model_image_path, ethnicity=None, skin_tone=None, additional_features=None):
        """
        Swap only the face in a model image while keeping everything else exactly the same
        
        Args:
            original_model_image_path: Path to the original model image
            ethnicity: Desired ethnicity for the new face
            skin_tone: Desired skin tone for the new face
            additional_features: Any additional facial features to include
            
        Returns:
            PIL Image object with just the face swapped
        """
        try:
            # Load the original image
            original_img = Image.open(original_model_image_path)
            
            # Convert image to bytes
            buffer = BytesIO()
            original_img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            
            # Create a very specific prompt for face swapping only
            prompt = f"""
            Modify ONLY the face of the model in this fashion photo to be a {ethnicity if ethnicity else "different"} person with {skin_tone if skin_tone else "appropriate"} skin tone.
            
            EXTREMELY IMPORTANT REQUIREMENTS:
            - ONLY change the face and neck
            - Do NOT change the hair style or color
            - Do NOT change the body shape or posture
            - Do NOT change the clothing or any apparel details
            - Do NOT change the background or composition
            - Do NOT change the lighting, colors, or photography style
            - Do NOT change the model's pose or position
            
            The output should look EXACTLY like the input image with ONLY the face ethnicity changed.
            The colors, textures, patterns, and all details of the apparel must remain 100% identical.
            """
            
            if additional_features:
                prompt += f"\nFor the face, include these specific features: {additional_features}"
            
            # Use the Gemini image model for the face swap
            model = genai.GenerativeModel(self.gemini_image_model)
            response = model.generate_content([
                prompt,
                {"mime_type": "image/png", "data": img_bytes}
            ])
            
            # Process the response
            if hasattr(response, 'candidates') and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                        image_bytes = part.inline_data.data
                        return Image.open(BytesIO(image_bytes))
            
            raise Exception("No image was returned in the response")
            
        except Exception as e:
            print(f"Error performing face swap: {str(e)}")
            return None
            
    def swap_face_skin_hair(self, original_model_image_path, ethnicity=None, skin_tone=None, hairstyle=None, additional_features=None):
        """
        Swap the face, skin tone, and hairstyle in a model image while keeping the clothing and pose the same
        
        Args:
            original_model_image_path: Path to the original model image
            ethnicity: Desired ethnicity for the new face
            skin_tone: Desired skin tone for face and visible skin areas
            hairstyle: Desired hairstyle to apply
            additional_features: Any additional features to include
            
        Returns:
            PIL Image object with face, skin tone and hairstyle swapped
        """
        try:
            # Load the original image
            original_img = Image.open(original_model_image_path)
            
            # Convert image to bytes
            buffer = BytesIO()
            original_img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            
            # Create a specific prompt for face, skin tone, and hairstyle swapping
            prompt = f"""
            Transform the model in this fashion photo to be a {ethnicity if ethnicity else "different"} person with:
            - {skin_tone if skin_tone else "appropriate"} skin tone on face and all visible skin areas
            - {hairstyle if hairstyle else "different"} hairstyle
            
            EXTREMELY IMPORTANT REQUIREMENTS:
            - Change the face ethnicity, skin tone on ALL visible skin (arms, legs, etc.), and hairstyle
            - Do NOT change the body shape or posture
            - Do NOT change the clothing or any apparel details
            - Do NOT change the background or composition
            - Do NOT change the lighting, colors, or photography style of the overall image
            - Do NOT change the model's pose or position
            
            The output should look EXACTLY like the input image with ONLY the face ethnicity, skin tone, and hairstyle changed.
            The colors, textures, patterns, and all details of the apparel must remain 100% identical.
            """
            
            if additional_features:
                prompt += f"\nAdditional features to include: {additional_features}"
            
            # Use the Gemini image model for the transformation
            model = genai.GenerativeModel(self.gemini_image_model)
            response = model.generate_content([
                prompt,
                {"mime_type": "image/png", "data": img_bytes}
            ])
            
            # Process the response
            if hasattr(response, 'candidates') and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                        image_bytes = part.inline_data.data
                        return Image.open(BytesIO(image_bytes))
            
            raise Exception("No image was returned in the response")
            
        except Exception as e:
            print(f"Error performing face, skin tone, and hairstyle swap: {str(e)}")
            return None
            
    def generate_multiple_poses(self, base_image_path, ethnicity=None, skin_tone=None, hairstyle=None, additional_features=None):
        """
        Generate the same model in four different poses/angles while keeping the same clothing
        
        Args:
            base_image_path: Path to the original or already transformed model image
            ethnicity: Ethnicity of the model
            skin_tone: Skin tone of the model
            hairstyle: Hairstyle of the model
            additional_features: Any additional features to include
            
        Returns:
            Dictionary with four PIL Image objects with different poses
        """
        try:
            # Load the base image
            base_img = Image.open(base_image_path)
            
            # Convert image to bytes
            buffer = BytesIO()
            base_img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            
            # Define the four different poses/angles
            poses = {
                "front": "Front-facing, looking directly at the camera",
                "three_quarter": "¾ view turned to the left, showing part of the face and body at an angle",
                "profile": "Profile view to the left, showing the side of the model",
                "back": "Back view, showing the back of the model and clothing"
            }
            
            results = {}
            
            # Generate image for each pose
            for pose_key, pose_description in poses.items():
                print(f"Generating {pose_key} view...")
                
                # Create prompt for generating the specific pose
                prompt = f"""
                Transform this model into a {pose_description} pose.
                
                The model is {ethnicity if ethnicity else "as shown"} with {skin_tone if skin_tone else "the same"} skin tone
                and {hairstyle if hairstyle else "the same"} hairstyle.
                
                EXTREMELY IMPORTANT REQUIREMENTS:
                - Keep the EXACT SAME clothing/apparel, colors, patterns, and textures
                - Keep the EXACT SAME background and lighting style
                - Only change the pose/angle to a {pose_description}
                - Keep the same model with identical facial features, body type, and height
                - The final image should look like a professional fashion photo from a different angle
                
                Generate a high-quality, professional fashion photograph.
                """
                
                if additional_features:
                    prompt += f"\nAdditional specifications: {additional_features}"
                
                # Use the Gemini image model for the transformation
                model = genai.GenerativeModel(self.gemini_image_model)
                response = model.generate_content([
                    prompt,
                    {"mime_type": "image/png", "data": img_bytes}
                ])
                
                # Process the response
                if hasattr(response, 'candidates') and response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                            image_bytes = part.inline_data.data
                            results[pose_key] = Image.open(BytesIO(image_bytes))
                            break
                
                # Brief pause to avoid rate limiting
                time.sleep(2)
            
            return results
            
        except Exception as e:
            print(f"Error generating multiple poses: {str(e)}")
            return None

def get_pose_options():
    """
    Get pose/angle options for the app
    
    Returns:
        Dictionary of pose options
    """
    return {
        "front": "Front-facing",
        "three_quarter": "¾ view (turned to left)",
        "profile": "Profile (side view to left)",
        "back": "Back view"
    }