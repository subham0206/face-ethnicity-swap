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
</style>
""", unsafe_allow_html=True)

# Initialize model manager
model_manager = ModelManager()

# App title and description
st.markdown('<p class="main-title">B+C Virtual Photoshoot App</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Create professional fashion photos with customizable models</p>', unsafe_allow_html=True)

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

# Left column - Input controls
with left_col:
    # Check API keys first time
    if not st.session_state.api_keys_checked:
        api_status = utils.check_api_keys()
        st.session_state.api_keys_checked = True
        
        if not (api_status["openai"] or api_status["google"]):
            st.error("No valid API keys found. Please make sure your API keys are correctly set up in the .env file.")
    
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
        
        additional_features = st.text_area(
            "Additional Features or Instructions",
            placeholder="E.g., specific eye color, facial features, etc.",
            help="Add any additional details or instructions for the AI model"
        )
    
    # Generate button
    generate_btn = st.button(
        "Generate Transformed Image", 
        disabled=st.session_state.uploaded_image_path is None or st.session_state.processing,
        use_container_width=True
    )
    
    if generate_btn:
        st.session_state.processing = True

# Right column - Display results
with right_col:
    tabs = st.tabs(["Original Image", "Transformed Image"])
    
    # Display original image
    with tabs[0]:
        if st.session_state.uploaded_image_path is not None:
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
                            generated_result = google_integration.generate_image_swap(
                                st.session_state.uploaded_image_path,
                                additional_features=additional_features,
                                use_predefined_model=True,
                                predefined_model_id=st.session_state.selected_model_id,
                                predefined_model_gender=st.session_state.selected_gender,
                                swap_skin_and_hair=swap_skin_and_hair,
                                generate_multiple_poses=generate_multiple_poses
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
                                
                                # Always use Google for generation
                                generated_result = google_integration.generate_image_swap(
                                    st.session_state.uploaded_image_path,
                                    "Custom", # Placeholder
                                    "Custom", # Placeholder
                                    "Custom", # Placeholder
                                    "Custom", # Placeholder
                                    "Custom", # Placeholder
                                    additional_features,
                                    swap_skin_and_hair=swap_skin_and_hair,
                                    generate_multiple_poses=generate_multiple_poses
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
                            generated_result = google_integration.generate_image_swap(
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
    "clothing, background, and overall composition."
)