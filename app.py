import streamlit as st
import os
from PIL import Image
import time

# Import our custom modules
import utils
import image_processor
import openai_integration
import google_integration
from model_manager import ModelManager

# Set page configuration
st.set_page_config(
    page_title="B+C Virtual Photoshoot App",
    page_icon="📸",
    layout="wide"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #FF4B4B;
    }
    .sub-title {
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        color: #808080;
    }
    .stButton button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .section-title {
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .model-description {
        font-size: 0.8rem;
        color: #808080;
        margin-top: 0.3rem;
        margin-bottom: 1rem;
    }
    .model-preview {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        background-color: #f9f9f9;
    }
    .swatch-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .swatch-item {
        width: 50px;
        height: 50px;
        border-radius: 4px;
        cursor: pointer;
        border: 2px solid transparent;
    }
    .swatch-item.selected {
        border: 2px solid #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

# Initialize model manager
model_manager = ModelManager()

# App title and description
st.markdown('<p class="main-title">B+C Virtual Photoshoot App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Create professional fashion photos with customizable models.<br><b>All generated images are automatically upscaled to <span style="color:#FF4B4B">3000x5000 pixels</span> for maximum clarity and print quality.</b></p>', unsafe_allow_html=True)

# Create two columns for the main layout
left_col, right_col = st.columns([2, 3])

# Session state initialization
if "uploaded_image_path" not in st.session_state:
    st.session_state.uploaded_image_path = None
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "api_keys_checked" not in st.session_state:
    st.session_state.api_keys_checked = False
if "processing" not in st.session_state:
    st.session_state.processing = False
if "is_product_only" not in st.session_state:
    st.session_state.is_product_only = False  # Changed default to False (toggle disabled)
if "use_predefined_model" not in st.session_state:
    st.session_state.use_predefined_model = False
if "selected_gender" not in st.session_state:
    st.session_state.selected_gender = "Male"
if "selected_model_id" not in st.session_state:
    st.session_state.selected_model_id = None
if "selected_swatch" not in st.session_state:
    st.session_state.selected_swatch = None
if "apparel_type" not in st.session_state:
    st.session_state.apparel_type = "top"
if "color_changed_image" not in st.session_state:
    st.session_state.color_changed_image = None
if "use_white_background" not in st.session_state:
    st.session_state.use_white_background = False
if "preview_result" not in st.session_state:
    st.session_state.preview_result = None

# --- Preview Logic Triggered at Top Level ---
if "preview_btn_clicked" not in st.session_state:
    st.session_state.preview_btn_clicked = False
if "preview_result" not in st.session_state:
    st.session_state.preview_result = None
preview_color = None
preview_swatch_name = None
preview_swatch_path = None
if st.session_state.selected_swatch:
    preview_color = st.session_state.selected_swatch['name']
    preview_swatch_name = st.session_state.selected_swatch['name']
    preview_swatch_path = st.session_state.selected_swatch['path']
preview_apparel_type = st.session_state.get("preview_apparel_type", "t-shirt")
if st.session_state.preview_btn_clicked and preview_swatch_path:
    print(f"DEBUG: Preview button clicked. preview_swatch_path={preview_swatch_path}, preview_apparel_type={preview_apparel_type}, default_model_path=models/male_mixed_race.png")
    default_model_path = os.path.join("models", "male_mixed_race.png")
    if not os.path.exists(default_model_path):
        print("DEBUG: Default model image not found for preview.")
        st.session_state.preview_result = None
    else:
        print(f"DEBUG: About to call google_integration.change_apparel_color...")
        try:
            preview_result = google_integration.change_apparel_color(
                default_model_path,
                preview_swatch_path,
                preview_apparel_type
            )
            print(f"DEBUG: google_integration.change_apparel_color returned: {type(preview_result)} value: {preview_result}")
            if preview_result:
                st.session_state.preview_result = preview_result
                print("DEBUG: Preview image stored in session state.")
            else:
                st.session_state.preview_result = None
                print("DEBUG: Failed to generate preview image.")
        except Exception as e:
            print(f"DEBUG: Exception in google_integration.change_apparel_color: {e}")
            st.session_state.preview_result = None
    st.session_state.preview_btn_clicked = False

# Left column - Input controls
with left_col:
    # Check API keys first time
    if not st.session_state.api_keys_checked:
        api_status = utils.check_api_keys()
        st.session_state.api_keys_checked = True
        
        if not (api_status["openai"] or api_status["google"]):
            st.error("No valid API keys found. This app requires API keys to function properly.")
            st.warning("""
            ### How to Add API Keys:
            
            #### For Local Development:
            1. Create a `.env` file in the root directory of this project
            2. Add your API keys in the following format:
               ```
               OPENAI_API_KEY=your-openai-key
               GOOGLE_API_KEY=your-google-key
               ```
               
            #### For Streamlit Cloud Deployment:
            1. Go to your app settings in the Streamlit Cloud dashboard
            2. Navigate to "Secrets" section
            3. Add your API keys in the following format:
               ```
               OPENAI_API_KEY = "your-openai-key"
               GOOGLE_API_KEY = "your-google-key"
               ```
               
            At least one API key is required. The app will use available services based on which keys are provided.
            """)
        elif not api_status["openai"]:
            st.warning("OpenAI API key not found. Some features will be limited to Google AI services only.")
        elif not api_status["google"]:
            st.warning("Google API key not found. Some features will be limited to OpenAI services only.")
        else:
            st.success("API keys validated successfully!")
    
    # Upload image section
    st.markdown('<p class="section-title">1. Upload Image & Specify Type</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    # Image type toggle
    st.session_state.is_product_only = st.toggle(
        "This is a product-only image (no model)",
        value=False,
        help="Toggle ON if your image shows only the apparel item without a model. Toggle OFF if your image already shows a model wearing the apparel."
    )
    
    if st.session_state.is_product_only:
        st.info("📸 AI will generate models wearing your product while maintaining the exact texture and color.")
    else:
        st.info("📸 AI will swap the model's ethnicity while preserving the apparel and pose.")
    
    if uploaded_file is not None:
        # Save uploaded file and display
        st.session_state.uploaded_image_path = image_processor.save_uploaded_image(uploaded_file)
        st.session_state.generated_image = None  # Reset generated image
        st.session_state.color_changed_image = None  # Reset color-changed image
        st.session_state.selected_swatch = None  # Reset selected swatch
        st.success("Image uploaded successfully!")
    
    # Model selection section
    st.markdown('<p class="section-title">2. Select Generation Method</p>', unsafe_allow_html=True)
    # Using a hidden default value for ai_model
    ai_model = "Google Gemini"
    
    # Display description for Google Gemini
    st.markdown('<p class="model-description">Using Google Gemini with auto-detection of the best available model for your transformation.</p>', unsafe_allow_html=True)
    
    # Toggle for using predefined models
    st.session_state.use_predefined_model = st.toggle(
        "Use Predefined Models",
        value=True,
        help="Toggle ON to use predefined AI models instead of generating new ones each time. This improves consistency and speeds up generation."
    )
    
    # Customization section
    st.markdown('<p class="section-title">3. Customize Appearance</p>', unsafe_allow_html=True)
    
    # Show different options based on whether using predefined models or not
    if st.session_state.use_predefined_model:
        # Select predefined model gender
        st.session_state.selected_gender = st.radio("Select Gender", ["Male", "Female"])
        
        # Get available models for the selected gender
        available_models = model_manager.get_model_names_for_dropdown(st.session_state.selected_gender)
        
        # Check if any predefined models were found
        if not available_models:
            st.warning(f"No predefined {st.session_state.selected_gender.lower()} models found. Please run generate_models.py first or switch to custom generation.")
            # Add a button to run generate_models.py
            if st.button("Generate Predefined Models"):
                with st.spinner("Generating predefined models... This may take several minutes."):
                    try:
                        st.info("Starting model generation process...")
                        os.system("python generate_models.py")
                        st.success("Models generated successfully! Please refresh the page.")
                    except Exception as e:
                        st.error(f"Error generating models: {str(e)}")
        else:
            # Select specific predefined model
            selected_model_name = st.selectbox("Select Model", available_models)
            
            # Find the model ID from the selected name
            models = model_manager.get_model_list(st.session_state.selected_gender)
            for model in models:
                if model["name"] == selected_model_name:
                    st.session_state.selected_model_id = model["id"]
                    break
            
            # Show a preview of the selected model if available
            if st.session_state.selected_model_id:
                model_data = model_manager.get_model(st.session_state.selected_model_id, st.session_state.selected_gender)
                if model_data and os.path.exists(model_data["path"]):
                    st.markdown("#### Preview of Selected Model")
                    with st.container(height=250, border=True):
                        st.image(model_data["path"], width=150)
    else:
        # Standard customization options
        ethnicity = st.selectbox("Ethnicity", utils.get_ethnicity_options())
        height = st.selectbox("Height", utils.get_height_options())
        body_type = st.selectbox("Body Type", utils.get_body_type_options())
        skin_tone = st.selectbox("Skin Tone", utils.get_skin_tone_options())
        hairstyle = st.selectbox("Hairstyle", utils.get_hairstyle_options())
    
    # Advanced customization options (expandable)
    with st.expander("Advanced Customization"):
        swap_skin_and_hair = st.toggle(
            "Swap skin tone and hairstyle too",
            value=True,
            help="Toggle ON to swap the skin tone on all visible skin and change hairstyle along with the face. Toggle OFF to swap only the face and neck."
        )
        
        generate_multiple_poses = st.toggle(
            "Generate multiple angles",
            value=True,
            help="Toggle ON to generate the model in four different angles (front, ¾ view, profile, back view)"
        )
        
        st.session_state.use_white_background = st.toggle(
            "Use white background instead of light grey",
            value=False,
            help="Toggle ON to use a white background instead of the default light grey background"
        )
        
        # Image quality enhancement options
        st.markdown("#### Image Quality Options")
        
        st.session_state.preserve_max_quality = st.toggle(
            "Preserve maximum image quality",
            value=True,
            help="Toggle ON to maintain the highest possible image quality during processing (may increase processing time)"
        )
        
        st.session_state.use_ai_enhancement = st.toggle(
            "Apply AI enhancement after generation",
            value=False,
            help="Toggle ON to apply AI-based image enhancement to the final result (improves detail and sharpness)"
        )
        
        if st.session_state.use_ai_enhancement:
            st.session_state.enhancement_method = st.selectbox(
                "Enhancement Method",
                ["Default", "Super Resolution", "Advanced Sharpening", "Pixelcut (requires API key)"],
                help="Choose which AI enhancement method to apply to the final image"
            )
            
            if st.session_state.enhancement_method == "Pixelcut (requires API key)":
                pixelcut_api_key = st.text_input(
                    "Pixelcut API Key", 
                    type="password",
                    help="Enter your Pixelcut API key to use their service for image enhancement"
                )
                st.session_state.pixelcut_api_key = pixelcut_api_key if pixelcut_api_key else None
        
        additional_features = st.text_area(
            "Additional Features or Instructions",
            placeholder="E.g., specific eye color, facial features, etc.",
            help="Add any additional details or instructions for the AI model"
        )
        
        # New: Swatch image uploader for custom color reference
        uploaded_swatch_file = st.file_uploader(
            "Upload a color swatch image for apparel color reference (optional)",
            type=["jpg", "jpeg", "png"],
            key="uploaded_swatch_file"
        )
        if uploaded_swatch_file is not None:
            st.session_state.uploaded_swatch_path = image_processor.save_uploaded_image(uploaded_swatch_file)
            st.success("Swatch image uploaded successfully!")
        else:
            st.session_state.uploaded_swatch_path = None
    
    # Color swatch selection section - make it collapsible
    with st.expander("Apparel Color Options", expanded=False):
        # Get available swatches
        available_swatches = utils.get_available_swatches()
        if available_swatches:
            # Apparel type selection
            st.session_state.apparel_type = st.selectbox(
                "Apparel Type", 
                ["top", "hoodie", "t-shirt", "crop top", "sweater", "shirt", "blouse", "tank top"]
            )
            
            # Display color swatches in a grid
            st.write("Select a color swatch:")
            
            # Create a 3-column grid for swatches
            swatch_cols = st.columns(3)
            
            for i, (color_name, swatch_path) in enumerate(available_swatches.items()):
                col_index = i % 3
                with swatch_cols[col_index]:
                    # Create a container for the swatch with name below
                    st.image(swatch_path, caption=color_name, width=80)
                    if st.button(f"Select {color_name}", key=f"swatch_{color_name}"):
                        st.session_state.selected_swatch = {"name": color_name, "path": swatch_path}
                        st.success(f"Selected color: {color_name}. It will be applied automatically when you generate the transformed image.")
            
            # Show selected swatch info if one is selected
            if st.session_state.selected_swatch:
                st.info(f"Selected color '{st.session_state.selected_swatch['name']}' will be applied when you click 'Generate Transformed Image'")
        else:
            st.warning("No color swatches found in the swatches directory.")
    
    # Generate button
    generate_btn = st.button(
        "Generate Transformed Image", 
        disabled=st.session_state.uploaded_image_path is None or st.session_state.processing,
        use_container_width=True
    )
    
    if generate_btn:
        # If a swatch is selected or uploaded, apply the color change before generating the image
        swatch_path = None
        swatch_name = None
        if st.session_state.uploaded_swatch_path:
            swatch_path = st.session_state.uploaded_swatch_path
            swatch_name = "Custom Uploaded Swatch"
        elif st.session_state.selected_swatch:
            swatch_path = st.session_state.selected_swatch['path']
            swatch_name = st.session_state.selected_swatch['name']
        if swatch_path and st.session_state.uploaded_image_path:
            with st.spinner(f"Applying {swatch_name} color to apparel..."):
                try:
                    print(f"DEBUG: Applying color to apparel using swatch: {swatch_path}")
                    color_changed_img = google_integration.change_apparel_color(
                        st.session_state.uploaded_image_path,
                        swatch_path,
                        st.session_state.apparel_type
                    )
                    if color_changed_img:
                        output_path = image_processor.save_color_changed_image(
                            color_changed_img,
                            swatch_name,
                            st.session_state.uploaded_image_path
                        )
                        st.session_state.color_changed_image = output_path
                        st.session_state.source_image_for_generation = output_path
                        st.success(f"Successfully applied {swatch_name} color to the {st.session_state.apparel_type}.")
                    else:
                        st.warning("Failed to apply the selected color. Proceeding with the original image.")
                        st.session_state.color_changed_image = None
                        st.session_state.source_image_for_generation = st.session_state.uploaded_image_path
                except Exception as e:
                    print(f"DEBUG: Error in auto color change: {str(e)}")
                    st.warning(f"Could not apply the selected color due to an error. Proceeding with the original image.")
                    st.session_state.color_changed_image = None
                    st.session_state.source_image_for_generation = st.session_state.uploaded_image_path
        else:
            st.session_state.source_image_for_generation = st.session_state.uploaded_image_path
                    
        # Debug output of available images
        print(f"DEBUG - When Generate button pressed:")
        print(f"DEBUG - Uploaded image path: {st.session_state.uploaded_image_path}")
        print(f"DEBUG - Source image for generation: {st.session_state.source_image_for_generation}")
        print(f"DEBUG - Generated image path: {st.session_state.generated_image}")
        print(f"DEBUG - Color-changed image path: {st.session_state.color_changed_image}")
        st.session_state.processing = True

# Right column - Display results
with right_col:
    tabs = st.tabs(["Original Image", "Transformed Image"])
    
    # Display original image
    with tabs[0]:
        if st.session_state.uploaded_image_path is not None:
            if st.session_state.color_changed_image is not None:
                # If color was changed, show color-changed image instead
                st.image(
                    st.session_state.color_changed_image, 
                    caption="Original Image with Color Applied", 
                    use_column_width=True
                )
                st.info("Showing image with selected color applied")
            else:
                st.image(
                    st.session_state.uploaded_image_path, 
                    caption="Original Image", 
                    use_column_width=True
                )
        else:
            st.info("Please upload an image to get started.")
    
    # Display generated image
    with tabs[1]:
        if generate_btn and st.session_state.uploaded_image_path is not None:
            with st.spinner("Generating transformed image... (this may take up to 60 seconds)"):
                try:
                    # Store the results in session state
                    if "generated_images" not in st.session_state:
                        st.session_state.generated_images = {}
                        
                    if st.session_state.use_predefined_model:
                        # Make sure we have a valid model selection
                        if not st.session_state.selected_model_id:
                            st.error("No predefined model selected. Please select a model or disable 'Use Predefined Models'.")
                            st.session_state.processing = False
                        else:
                            # Use Google for predefined model apparel swapping
                            # Use color-changed image if available, otherwise use original image
                            source_image = st.session_state.source_image_for_generation
                            print(f"DEBUG: Using source image for generation: {source_image}")
                            generated_result = google_integration.generate_image_swap(
                                source_image,
                                additional_features=additional_features,
                                use_predefined_model=True,
                                predefined_model_id=st.session_state.selected_model_id,
                                predefined_model_gender=st.session_state.selected_gender,
                                swap_skin_and_hair=swap_skin_and_hair,
                                generate_multiple_poses=generate_multiple_poses,
                                use_white_background=st.session_state.use_white_background
                            )
                            
                            if generated_result is not None:
                                # Check if we got multiple poses or just one image
                                if isinstance(generated_result, dict):
                                    # Multiple poses
                                    output_paths = image_processor.save_multiple_pose_images(generated_result)
                                    st.session_state.generated_images = output_paths
                                    # Set front view as the main generated image
                                    if "front" in output_paths:
                                        st.session_state.generated_image = output_paths["front"]
                                    else:
                                        # Single image
                                        output_path = image_processor.save_result_image(generated_result)
                                        st.session_state.generated_image = output_path
                                        st.session_state.generated_images = {"front": output_path}
                                else:
                                    st.error("Failed to generate image using predefined model. The system will try to generate from scratch.")
                                    
                                    # Fallback to standard options instead of using "Custom" placeholders
                                    # Use color-changed image if available, otherwise use original image
                                    source_image = st.session_state.source_image_for_generation
                                    generated_result = google_integration.generate_image_swap(
                                        source_image,
                                        ethnicity="Mixed/Multiracial",  # Default ethnicity instead of "Custom"
                                        skin_color="Medium",  # Default skin tone instead of "Custom"
                                        hairstyle="Medium straight",  # Default hairstyle instead of "Custom"
                                        additional_features=additional_features,
                                        swap_skin_and_hair=swap_skin_and_hair,
                                        generate_multiple_poses=generate_multiple_poses,
                                        use_white_background=st.session_state.use_white_background
                                    )
                                    
                                    if generated_result is not None:
                                        # Check if we got multiple poses or just one image
                                        if isinstance(generated_result, dict):
                                            # Multiple poses
                                            output_paths = image_processor.save_multiple_pose_images(generated_result)
                                            st.session_state.generated_images = output_paths
                                            # Set front view as the main generated image
                                            if "front" in output_paths:
                                                st.session_state.generated_image = output_paths["front"]
                                        else:
                                            # Single image
                                            output_path = image_processor.save_result_image(generated_result)
                                            st.session_state.generated_image = output_path
                                            st.session_state.generated_images = {"front": output_path}
                    else:
                        # Use standard generation approach
                        if ai_model == "OpenAI GPT-4o & DALL-E":
                            # Use OpenAI for image generation
                            generated_result = openai_integration.generate_image_swap(
                                st.session_state.uploaded_image_path,
                                ethnicity,
                                height,
                                body_type,
                                skin_tone,
                                hairstyle,
                                additional_features,
                                swap_skin_and_hair=swap_skin_and_hair,
                                generate_multiple_poses=generate_multiple_poses
                            )
                        else:
                            # Use Google for image generation
                            # Use color-changed image if available, otherwise use original image
                            source_image = st.session_state.source_image_for_generation
                            generated_result = google_integration.generate_image_swap(
                                source_image,
                                ethnicity,
                                height,
                                body_type,
                                skin_tone,
                                hairstyle,
                                additional_features,
                                swap_skin_and_hair=swap_skin_and_hair,
                                generate_multiple_poses=generate_multiple_poses,
                                use_white_background=st.session_state.use_white_background
                            )
                        
                        if generated_result is not None:
                            # Check if we got multiple poses or just one image
                            if isinstance(generated_result, dict):
                                # Multiple poses
                                output_paths = image_processor.save_multiple_pose_images(generated_result)
                                st.session_state.generated_images = output_paths
                                # Set front view as the main generated image
                                if "front" in output_paths:
                                    st.session_state.generated_image = output_paths["front"]
                            else:
                                # Single image
                                output_path = image_processor.save_result_image(generated_result)
                                st.session_state.generated_image = output_path
                                st.session_state.generated_images = {"front": output_path}
                        else:
                            st.error("Failed to generate image. Please try again or switch to a different AI model.")
                    
                    # Apply AI image enhancement if enabled
                    if "use_ai_enhancement" in st.session_state and st.session_state.use_ai_enhancement:
                        from image_enhancer import enhance_image
                        
                        # Show enhancement notification
                        enhancement_method = st.session_state.get("enhancement_method", "Default")
                        with st.spinner(f"Applying {enhancement_method} image enhancement..."):
                            
                            # Process each image in the generated images
                            enhanced_images = {}
                            for pose, image_path in st.session_state.generated_images.items():
                                # Map enhancement method to the actual method
                                method_map = {
                                    "Default": "ai_upscale",
                                    "Super Resolution": "super_resolution",
                                    "Advanced Sharpening": "sharpen",
                                    "Pixelcut (requires API key)": "pixelcut"
                                }
                                
                                method = method_map.get(enhancement_method, "ai_upscale")
                                
                                # Additional parameters for specific methods
                                kwargs = {}
                                if method == "pixelcut" and hasattr(st.session_state, "pixelcut_api_key"):
                                    kwargs["api_key"] = st.session_state.pixelcut_api_key
                                    kwargs["quality"] = "high"
                                elif method == "super_resolution":
                                    kwargs["scale"] = 2
                                    kwargs["model"] = "edsr"
                                elif method == "sharpen":
                                    kwargs["amount"] = 2.0
                                    
                                # Apply enhancement
                                try:
                                    enhanced_img = enhance_image(image_path, method=method, **kwargs)
                                    # Save the enhanced image
                                    enhanced_path = image_path.replace(".png", "_enhanced.png")
                                    enhanced_img.save(enhanced_path, format="PNG", compress_level=1)
                                    # Replace the original image with the enhanced version
                                    enhanced_images[pose] = enhanced_path
                                    print(f"Enhanced {pose} image saved to {enhanced_path}")
                                except Exception as e:
                                    print(f"Error enhancing {pose} image: {str(e)}")
                                    enhanced_images[pose] = image_path  # Keep original if enhancement fails
                            
                            # Update the generated images with enhanced versions
                            st.session_state.generated_images = enhanced_images
                            # Update the main generated image (front view)
                            if "front" in enhanced_images:
                                st.session_state.generated_image = enhanced_images["front"]
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                
                st.session_state.processing = False
        
        if st.session_state.generated_image is not None or ("generated_images" in st.session_state and st.session_state.generated_images):
            # Show the different poses if available
            if "generated_images" in st.session_state and len(st.session_state.generated_images) > 1:
                st.subheader("Model in Different Angles")
                
                # Get pose labels from utils
                pose_labels = utils.get_pose_options()
                
                # Create columns for each pose
                pose_columns = st.columns(len(st.session_state.generated_images))
                
                # Display each pose in its column
                for i, (pose, path) in enumerate(sorted(st.session_state.generated_images.items())):
                    with pose_columns[i]:
                        st.image(path, caption=pose_labels.get(pose, pose.capitalize()), use_column_width=True)
                        
                        # Add download button for each pose
                        with open(path, "rb") as file:
                            st.download_button(
                                label=f"Download {pose_labels.get(pose, pose.capitalize())}",
                                data=file,
                                file_name=f"model_{pose}.png",
                                mime="image/png"
                            )
            else:
                # Display single image (front view only)
                st.image(
                    st.session_state.generated_image, 
                    caption="Transformed Image", 
                    use_column_width=True
                )
                
                # Download button for the generated image
                with open(st.session_state.generated_image, "rb") as file:
                    btn = st.download_button(
                        label="Download Transformed Image",
                        data=file,
                        file_name="transformed_model.png",
                        mime="image/png",
                        use_container_width=True
                    )
        elif st.session_state.uploaded_image_path is not None and not generate_btn and not st.session_state.processing:
            st.info("Click 'Generate Transformed Image' to see the result.")
        else:
            st.info("Upload an image and generate a transformation to see the result.")

# Footer
st.markdown("---")
st.markdown(
    "This application uses AI to transform fashion model images while preserving apparel details. "
    "The transformation process changes the model's ethnicity and facial features but maintains the "
    "clothing, background, and overall composition. You can also change the color of apparel items "
    "using the color swatches provided."
)