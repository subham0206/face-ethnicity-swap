#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Enhancement Module

This module provides tools for enhancing image quality through AI-powered upscaling,
sharpening, and detail enhancement. It includes both local processing options and
integration with third-party AI enhancement services.
"""

import os
import requests
import tempfile
import base64
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import time
import streamlit as st

# Try to import optional dependencies
try:
    from cv2 import dnn_superres
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class ImageEnhancer:
    """
    Class for enhancing images using various techniques
    """
    
    def __init__(self):
        """Initialize the image enhancer with available models"""
        self.temp_dir = os.path.join(tempfile.gettempdir(), "image_enhancement")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize OpenCV Super Resolution if available
        self.sr = None
        if CV2_AVAILABLE:
            try:
                self.sr = dnn_superres.DnnSuperResImpl_create()
                # We'll dynamically download models when needed
            except Exception as e:
                print(f"Error initializing OpenCV Super Resolution: {e}")
    
    def enhance_image(self, image, method="ai_upscale", scale=2, **kwargs):
        """
        Enhance an image using the specified method
        
        Args:
            image: PIL Image or path to image file
            method: Enhancement method ('ai_upscale', 'sharpen', 'super_resolution')
            scale: Upscaling factor (1-4)
            **kwargs: Additional parameters specific to each method
            
        Returns:
            Enhanced PIL Image
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)
        
        # Choose enhancement method
        if method == "ai_upscale":
            return self.upscale_image_ai(image, scale, **kwargs)
        elif method == "sharpen":
            return self.sharpen_image(image, **kwargs)
        elif method == "super_resolution":
            return self.super_resolution(image, scale, **kwargs)
        elif method == "pixelcut":
            return self.pixelcut_enhance(image, **kwargs)
        else:
            print(f"Unknown enhancement method: {method}")
            return image
    
    def upscale_image_ai(self, image, scale=2, strength=0.8):
        """
        Upscale an image using AI-based techniques
        
        Args:
            image: PIL Image to enhance
            scale: Upscaling factor (1-4)
            strength: Enhancement strength (0-1)
            
        Returns:
            Enhanced PIL Image
        """
        if CV2_AVAILABLE and self.sr:
            try:
                # Use OpenCV's super resolution for upscaling
                return self._upscale_opencv(image, scale)
            except Exception as e:
                print(f"Error with OpenCV upscaling: {e}")
        
        # Fallback to PIL-based enhancement if OpenCV fails or isn't available
        return self._upscale_pil(image, scale, strength)
    
    def _upscale_opencv(self, image, scale=2):
        """Upscale using OpenCV Super Resolution"""
        # Choose the appropriate model based on scale factor
        model_path = self._get_sr_model(scale)
        if not model_path:
            raise ValueError(f"No model available for scale factor {scale}")
        
        # Set the model
        self.sr.readModel(model_path)
        self.sr.setModel("edsr", scale)
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if img_array.shape[2] == 4:  # If RGBA, convert to RGB
            img_array = img_array[:, :, :3]
        img_array = img_array[:, :, ::-1].copy()  # RGB to BGR
        
        # Upscale image
        result = self.sr.upsample(img_array)
        
        # Convert back to PIL
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    
    def _get_sr_model(self, scale):
        """Get or download the appropriate super resolution model"""
        model_name = f"EDSR_x{scale}.pb"
        model_path = os.path.join(self.temp_dir, model_name)
        
        # Check if model exists, if not, download it
        if not os.path.exists(model_path):
            print(f"Downloading super resolution model {model_name}...")
            model_urls = {
                2: "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb",
                3: "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x3.pb",
                4: "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
            }
            
            if scale not in model_urls:
                return None
                
            try:
                response = requests.get(model_urls[scale])
                if response.status_code == 200:
                    with open(model_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Model {model_name} downloaded successfully")
                else:
                    print(f"Failed to download model, status code: {response.status_code}")
                    return None
            except Exception as e:
                print(f"Error downloading model: {e}")
                return None
                
        return model_path
    
    def _upscale_pil(self, image, scale=2, strength=0.8):
        """Upscale using PIL's built-in resizing with enhancements"""
        # Get original dimensions
        width, height = image.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # First resize using high-quality LANCZOS
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Apply a series of enhancements
        enhanced = resized
        
        # Apply sharpening
        sharpener = ImageEnhance.Sharpness(enhanced)
        sharp_factor = 1.0 + (1.5 * strength)
        enhanced = sharpener.enhance(sharp_factor)
        
        # Apply contrast enhancement
        contrast = ImageEnhance.Contrast(enhanced)
        contrast_factor = 1.0 + (0.3 * strength)
        enhanced = contrast.enhance(contrast_factor)
        
        # Apply subtle detail enhancement using unsharp mask filter
        radius = 2.0
        percent = 150 * strength
        threshold = 3
        enhanced = enhanced.filter(
            ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
        )
        
        return enhanced
    
    def sharpen_image(self, image, amount=1.8, threshold=3):
        """
        Sharpen an image using unsharp mask
        
        Args:
            image: PIL Image to sharpen
            amount: Sharpening amount (1-3)
            threshold: Threshold for sharpening effect
            
        Returns:
            Sharpened PIL Image
        """
        # First apply unsharp mask filter
        enhanced = image.filter(
            ImageFilter.UnsharpMask(radius=2.0, percent=amount * 100, threshold=threshold)
        )
        
        # Then enhance sharpness
        sharpener = ImageEnhance.Sharpness(enhanced)
        return sharpener.enhance(amount)
    
    def super_resolution(self, image, scale=2, model="edsr"):
        """
        Apply super-resolution to an image
        
        Args:
            image: PIL Image to enhance
            scale: Upscaling factor (1-4)
            model: SR model to use ('edsr', 'espcn', 'fsrcnn', 'lapsrn')
            
        Returns:
            Enhanced PIL Image
        """
        if not CV2_AVAILABLE:
            print("OpenCV not available for super resolution. Using PIL upscaling instead.")
            return self._upscale_pil(image, scale)
            
        try:
            # Choose model based on parameter
            model_name = model.lower()
            if model_name not in ["edsr", "espcn", "fsrcnn", "lapsrn"]:
                model_name = "edsr"  # Default to EDSR if invalid model specified
                
            # Get or download model
            model_path = self._get_sr_model(scale)
            if not model_path:
                return self._upscale_pil(image, scale)
                
            # Set model
            self.sr.readModel(model_path)
            self.sr.setModel("edsr", scale)
            
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            if img_array.shape[2] == 4:  # If RGBA, convert to RGB
                img_array = img_array[:, :, :3]
            img_array = img_array[:, :, ::-1].copy()  # RGB to BGR
            
            # Upscale image
            result = self.sr.upsample(img_array)
            
            # Convert back to PIL
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            return Image.fromarray(result_rgb)
        except Exception as e:
            print(f"Error in super resolution: {e}")
            return self._upscale_pil(image, scale)
    
    def pixelcut_enhance(self, image, api_key=None, quality="high"):
        """
        Enhance an image using the Pixelcut API
        
        Args:
            image: PIL Image to enhance
            api_key: Pixelcut API key (if None, will look in environment or st.secrets)
            quality: Quality level ('standard', 'high', 'ultra')
            
        Returns:
            Enhanced PIL Image or original image if enhancement fails
        """
        # Try to get API key from environment or Streamlit secrets if not provided
        if not api_key:
            try:
                api_key = os.getenv("PIXELCUT_API_KEY")
                if not api_key and 'PIXELCUT_API_KEY' in st.secrets:
                    api_key = st.secrets['PIXELCUT_API_KEY']
            except:
                pass
                
        if not api_key:
            print("No Pixelcut API key found. Skipping Pixelcut enhancement.")
            return image
            
        try:
            # Save image to buffer
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode('ascii')
            
            # Set quality parameter
            quality_level = quality.lower()
            if quality_level not in ["standard", "high", "ultra"]:
                quality_level = "high"
                
            # Prepare API request
            url = "https://api.pixelcut.ai/enhance"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "image": img_base64,
                "quality": quality_level,
                "enhance": "all"  # Options: "all", "quality", "upscale", "color", "faces"
            }
            
            # Make API request
            print(f"Sending image to Pixelcut API for {quality_level} enhancement...")
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                if 'enhanced_image' in result:
                    enhanced_data = base64.b64decode(result['enhanced_image'])
                    enhanced_image = Image.open(BytesIO(enhanced_data))
                    print(f"Image successfully enhanced with Pixelcut ({quality_level} quality)")
                    return enhanced_image
                else:
                    print(f"Error: No enhanced image in response")
            else:
                print(f"Error: Pixelcut API returned status code {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"Error enhancing image with Pixelcut: {e}")
            
        # Return original image if enhancement fails
        return image

    def ilovimg_enhance(self, image, api_key=None):
        """
        Enhance an image using the iLoveIMG API
        
        Args:
            image: PIL Image to enhance
            api_key: iLoveIMG API key (if None, will look in environment or st.secrets)
            
        Returns:
            Enhanced PIL Image or original image if enhancement fails
        """
        # Try to get API key from environment or Streamlit secrets if not provided
        if not api_key:
            try:
                api_key = os.getenv("ILOVIMG_API_KEY")
                if not api_key and 'ILOVIMG_API_KEY' in st.secrets:
                    api_key = st.secrets['ILOVIMG_API_KEY']
            except:
                pass
                
        if not api_key:
            print("No iLoveIMG API key found. Skipping iLoveIMG enhancement.")
            return image
            
        try:
            # Save image to buffer
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            
            # Prepare API request for iLoveIMG
            url = "https://api.ilovepdf.com/v1/auth"
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            
            # Get authentication token
            auth_response = requests.post(url, headers=headers)
            if auth_response.status_code != 200:
                print(f"Error authenticating with iLoveIMG: {auth_response.status_code}")
                print(auth_response.text)
                return image
                
            token = auth_response.json().get('token')
            
            # Create task
            task_url = "https://api.ilovepdf.com/v1/start/upscale"
            task_headers = {"Authorization": f"Bearer {token}"}
            task_response = requests.get(task_url, headers=task_headers)
            
            if task_response.status_code != 200:
                print(f"Error creating task: {task_response.status_code}")
                print(task_response.text)
                return image
                
            task = task_response.json()
            task_id = task.get('task')
            server = task.get('server')
            
            # Upload file
            upload_url = f"https://{server}/v1/upload"
            files = {'file': ('image.png', buffer.getvalue())}
            upload_data = {'task': task_id}
            
            upload_response = requests.post(upload_url, headers=task_headers, 
                                           files=files, data=upload_data)
            
            if upload_response.status_code != 200:
                print(f"Error uploading file: {upload_response.status_code}")
                print(upload_response.text)
                return image
                
            server_filename = upload_response.json().get('server_filename')
            
            # Process task
            process_url = f"https://{server}/v1/process"
            process_data = {
                'task': task_id,
                'tool': 'upscale',
                'files': [{'server_filename': server_filename, 'filename': 'image.png'}],
                'upscale_factor': 2,  # 2x upscaling
                'upscale_method': 'ml'  # Machine learning based upscaling
            }
            
            process_response = requests.post(process_url, headers=task_headers, json=process_data)
            
            if process_response.status_code != 200:
                print(f"Error processing task: {process_response.status_code}")
                print(process_response.text)
                return image
                
            # Download result
            download_url = f"https://{server}/v1/download/{task_id}"
            download_response = requests.get(download_url, headers=task_headers)
            
            if download_response.status_code == 200:
                enhanced_image = Image.open(BytesIO(download_response.content))
                print("Image successfully enhanced with iLoveIMG")
                return enhanced_image
            else:
                print(f"Error downloading result: {download_response.status_code}")
                print(download_response.text)
                
        except Exception as e:
            print(f"Error enhancing image with iLoveIMG: {e}")
            
        # Return original image if enhancement fails
        return image


# Create a singleton instance
enhancer = ImageEnhancer()

def enhance_image(image, method="ai_upscale", **kwargs):
    """
    Convenience function to access the image enhancer
    """
    return enhancer.enhance_image(image, method, **kwargs)