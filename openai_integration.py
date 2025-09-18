import os
import openai
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image
import requests
import tempfile
from utils import get_api_key

# Load environment variables
load_dotenv()

# Initialize OpenAI client with secure key retrieval
openai_api_key = get_api_key("openai")
client = openai.OpenAI(api_key=openai_api_key)

def generate_image_swap(
    original_image_path, 
    ethnicity, 
    height, 
    body_type, 
    skin_color, 
    hairstyle, 
    additional_features=None,
    swap_skin_and_hair=True,  # Parameter to toggle enhanced swapping
    generate_multiple_poses=False  # New parameter to generate multiple poses
):
    """
    Generate a fashion model image based on product-only images or transform existing model images.
    
    Args:
        original_image_path: Path to the original image (product or model)
        ethnicity: Target ethnicity
        height: Target height
        body_type: Target body type
        skin_color: Target skin color
        hairstyle: Target hairstyle
        additional_features: Any additional features to include
        swap_skin_and_hair: Whether to also swap skin tone and hairstyle (True) or just face (False)
        generate_multiple_poses: Whether to generate multiple poses/angles of the model
    
    Returns:
        If generate_multiple_poses=False: PIL Image object of the generated image
        If generate_multiple_poses=True: Dictionary with PIL Image objects for different poses
    """
    try:
        # Convert image to base64 for API
        with open(original_image_path, "rb") as img_file:
            img_content = img_file.read()
            img_base64 = base64.b64encode(img_content).decode("utf-8")
        
        # Determine if this is a product-only image
        is_product_only = True  # Default assumption for safety
        
        # Prepare prompt for image generation
        if is_product_only:
            prompt = f"""
            Generate a fashion model image with a {ethnicity} model wearing this exact apparel item.
            
            The model should have:
            - Height: {height}
            - Body type: {body_type}
            - Skin color: {skin_color}
            - Hairstyle: {hairstyle}
            
            MOST IMPORTANT REQUIREMENTS:
            - Keep the exact same apparel item shown in the image
            - Preserve ALL details of the apparel including exact texture, patterns, colors, fabric details, stitching
            - Make the apparel look exactly like the product image, just placed on a model
            - Create a fashion-photography style image with the model wearing the apparel
            - Use professional fashion photography lighting and composition
            - Generate a full-body or appropriate crop to showcase the apparel as it would appear in a catalog
            
            The final image should look like a professional fashion catalog photo where the apparel is identical to the original product image.
            """
            
            # If multiple poses are requested for product-only image, we'll handle that later
        else:
            if swap_skin_and_hair:
                # Enhanced transformation that includes face, skin tone, and hairstyle
                prompt = f"""
                Transform this fashion model photo to have a new ethnicity, skin tone, and hairstyle.
                
                Change to a {ethnicity} model with:
                - Height: {height} (maintain the same body proportions and pose)
                - Body type: {body_type} (maintain the same body shape)
                - Skin color: {skin_color} (apply to face AND all visible skin areas like arms, legs)
                - Hairstyle: {hairstyle} (change the hairstyle completely)
                
                Keep the following elements EXACTLY the same:
                - All clothing/apparel
                - Background and setting
                - Pose and body position
                - Lighting and photography style
                - Overall body shape and proportions
                
                Make the transformation realistic and professional for fashion photography.
                The transformation should be seamless, with the model's ethnicity, skin tone on all visible skin, and hairstyle changed,
                while every other element in the image stays exactly the same.
                """
            else:
                # Original face-only transformation
                prompt = f"""
                Transform this fashion model photo to have a new ethnicity and facial features.
                
                Change to a {ethnicity} model with:
                - Height: {height}
                - Body type: {body_type}
                - Skin color: {skin_color} (face and neck only)
                - Hairstyle: KEEP THE EXISTING HAIRSTYLE
                
                Keep the following elements EXACTLY the same:
                - All clothing/apparel
                - Background and setting
                - Pose and body position
                - Lighting and photography style
                - Hairstyle and hair color
                
                Make the transformation realistic and professional for fashion photography.
                """
        
        if additional_features:
            prompt += f"\nAdditional user instructions: {additional_features}"
            
        # For front-facing pose specific prompt
        if not is_product_only:
            prompt += "\nGenerate a FRONT-FACING view of the model looking directly at the camera."
        
        # Try GPT-4o with vision capabilities first for product-only images
        if is_product_only:
            try:
                print("Attempting image generation with GPT-4o...")
                messages = [
                    {
                        "role": "system", 
                        "content": "You are an expert fashion photographer and image editor. Generate a high-quality fashion model image based on product apparel images."
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                        ]
                    }
                ]
                
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=1000
                )
                
                # Extract image URLs from the response
                response_text = completion.choices[0].message.content
                
                # Look for image URLs in the response
                import re
                image_urls = re.findall(r'https://[^\s\)\"\']+\.(jpg|jpeg|png|gif)', response_text)
                
                if image_urls:
                    # Download the first image URL
                    image_response = requests.get(image_urls[0], stream=True)
                    if image_response.status_code == 200:
                        front_view_img = Image.open(BytesIO(image_response.content))
                        
                        # If multiple poses are not requested, return just the front view
                        if not generate_multiple_poses:
                            return front_view_img
                            
                        # For multiple poses, handle it separately for product-only images
                        # We need to use DALL-E for additional poses since GPT-4o doesn't directly support this
                        print("Front view generated, now generating additional poses with DALL-E...")
                        poses = {}
                        poses["front"] = front_view_img
                        
                        # Define the poses we need to generate
                        pose_descriptions = {
                            "three_quarter": "¾ view turned to the left, showing part of the face and body at an angle",
                            "profile": "Profile view to the left, showing the side of the model",
                            "back": "Back view, showing the back of the model and clothing"
                        }
                        
                        # Save front view temporarily
                        temp_path = os.path.join(tempfile.gettempdir(), f"front_view_{os.path.basename(original_image_path)}")
                        front_view_img.save(temp_path)
                        
                        # Generate each additional pose
                        for pose_key, pose_desc in pose_descriptions.items():
                            pose_prompt = f"""
                            Create a new view of this exact same fashion model, with identical clothing, in a {pose_desc} pose.
                            
                            The model should have:
                            - The exact same {ethnicity} model
                            - The same height: {height}
                            - The same body type: {body_type}
                            - The same skin color: {skin_color}
                            - The same hairstyle: {hairstyle}
                            
                            EXTREMELY IMPORTANT:
                            - Keep the EXACT SAME clothing/apparel with identical colors, patterns, textures
                            - Keep the EXACT SAME background and lighting
                            - Only change the camera angle/pose to {pose_desc}
                            - Make sure the model looks identical, just shown from a different angle
                            
                            Generate a high-quality, professional fashion catalog photo showing the {pose_key} view.
                            """
                            
                            if additional_features:
                                pose_prompt += f"\nAdditional specifications: {additional_features}"
                            
                            try:
                                pose_response = client.images.generate(
                                    model="dall-e-3",
                                    prompt=pose_prompt,
                                    n=1,
                                    size="1024x1024",
                                    response_format="b64_json"
                                )
                                
                                image_data = base64.b64decode(pose_response.data[0].b64_json)
                                poses[pose_key] = Image.open(BytesIO(image_data))
                                
                            except Exception as pose_error:
                                print(f"Error generating {pose_key} pose: {str(pose_error)}")
                        
                        return poses
                
                print("No image URLs found in GPT-4o response, falling back to DALL-E models")
            
            except Exception as e:
                print(f"GPT-4o generation failed: {str(e)}")
        
        # Try DALL-E 3 if we didn't succeed with GPT-4o or for non-product images
        try:
            print("Attempting image generation with DALL-E 3...")
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",
                response_format="b64_json"
            )
            
            # Convert the base64 encoded image to a PIL Image
            image_data = base64.b64decode(response.data[0].b64_json)
            front_view_img = Image.open(BytesIO(image_data))
            
            # If not generating multiple poses, return just the front view
            if not generate_multiple_poses:
                return front_view_img
                
            # For multiple poses, generate additional views
            print("Front view generated, now generating additional poses...")
            poses = {}
            poses["front"] = front_view_img
            
            # Define the poses we need to generate
            pose_descriptions = {
                "three_quarter": "¾ view turned to the left, showing part of the face and body at an angle",
                "profile": "Profile view to the left, showing the side of the model",
                "back": "Back view, showing the back of the model and clothing"
            }
            
            # Save front view temporarily
            temp_path = os.path.join(tempfile.gettempdir(), f"front_view_{os.path.basename(original_image_path)}")
            front_view_img.save(temp_path)
            
            # Generate each additional pose
            for pose_key, pose_desc in pose_descriptions.items():
                pose_prompt = f"""
                Create a new view of this exact same fashion model, with identical clothing, in a {pose_desc} pose.
                
                The model is a {ethnicity} person with {skin_color} skin tone and {hairstyle} hairstyle.
                
                EXTREMELY IMPORTANT:
                - Keep the EXACT SAME clothing/apparel with identical colors, patterns, textures
                - Keep the EXACT SAME background and lighting
                - Only change the camera angle/pose to {pose_desc}
                - Make sure the model looks identical, just shown from a different angle
                
                Generate a high-quality, professional fashion catalog photo showing the {pose_key} view.
                """
                
                if additional_features:
                    pose_prompt += f"\nAdditional specifications: {additional_features}"
                
                try:
                    pose_response = client.images.generate(
                        model="dall-e-3",
                        prompt=pose_prompt,
                        n=1,
                        size="1024x1024",
                        response_format="b64_json"
                    )
                    
                    image_data = base64.b64decode(pose_response.data[0].b64_json)
                    poses[pose_key] = Image.open(BytesIO(image_data))
                    
                except Exception as pose_error:
                    print(f"Error generating {pose_key} pose: {str(pose_error)}")
                    # Try DALL-E 2 as fallback for this pose
                    try:
                        pose_response = client.images.generate(
                            model="dall-e-2",
                            prompt=pose_prompt,
                            n=1,
                            size="1024x1024",
                            response_format="b64_json"
                        )
                        
                        image_data = base64.b64decode(pose_response.data[0].b64_json)
                        poses[pose_key] = Image.open(BytesIO(image_data))
                    except:
                        print(f"DALL-E 2 also failed for {pose_key} pose")
            
            return poses
                
        except Exception as e1:
            print(f"DALL-E 3 generation failed, trying DALL-E 2: {str(e1)}")
            
            # If DALL-E 3 fails, try DALL-E 2 as final fallback
            try:
                response = client.images.generate(
                    model="dall-e-2",
                    prompt=prompt,
                    n=1,
                    size="1024x1024",
                    response_format="b64_json"
                )
                
                image_data = base64.b64decode(response.data[0].b64_json)
                front_view_img = Image.open(BytesIO(image_data))
                
                # For DALL-E 2, we'll skip multiple pose generation as it doesn't follow instructions as well
                # Just return the front view even if multiple poses were requested
                return front_view_img if not generate_multiple_poses else {"front": front_view_img}
                
            except Exception as e2:
                print(f"DALL-E 2 generation failed: {str(e2)}")
                raise Exception(f"All OpenAI generation methods failed: {str(e1)}, {str(e2)}")
        
    except Exception as e:
        print(f"Error generating image with OpenAI: {str(e)}")
        return None