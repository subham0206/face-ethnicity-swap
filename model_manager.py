import os
from PIL import Image
import json
import glob

class ModelManager:
    """
    Handles loading and managing predefined AI models for apparel swapping.
    These models are generated once using generate_models.py and then reused
    for all apparel swapping operations to improve efficiency.
    """
    
    def __init__(self, models_dir="models"):
        """
        Initialize the model manager with the directory containing model images
        
        Args:
            models_dir: Directory containing the predefined model images
        """
        self.models_dir = models_dir
        self.models = {
            "Male": {},
            "Female": {}
        }
        self.load_available_models()
    
    def load_available_models(self):
        """Load all available predefined models from the models directory"""
        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            print(f"Created models directory: {self.models_dir}")
            return
        
        # Look for model images using the naming pattern from generate_models.py
        male_models = glob.glob(os.path.join(self.models_dir, "male_*.png"))
        female_models = glob.glob(os.path.join(self.models_dir, "female_*.png"))
        
        # Process male models
        for model_path in male_models:
            model_id = os.path.basename(model_path).replace("male_", "").replace(".png", "")
            self.models["Male"][model_id] = {
                "id": model_id,
                "path": model_path,
                "name": self._format_model_name(model_id)
            }
        
        # Process female models
        for model_path in female_models:
            model_id = os.path.basename(model_path).replace("female_", "").replace(".png", "")
            self.models["Female"][model_id] = {
                "id": model_id,
                "path": model_path,
                "name": self._format_model_name(model_id)
            }
        
        print(f"Loaded {len(self.models['Male'])} male models and {len(self.models['Female'])} female models")
    
    def _format_model_name(self, model_id):
        """Format model ID to a user-friendly name"""
        # Convert snake_case to Title Case with spaces
        name = model_id.replace("_", " ").title()
        return name
    
    def get_model_list(self, gender=None):
        """
        Get a list of available models
        
        Args:
            gender: Filter by gender ('Male' or 'Female'). If None, return all models.
        
        Returns:
            List of model dictionaries with id, name, and path
        """
        if gender and gender in self.models:
            return list(self.models[gender].values())
        elif not gender:
            # Combine all models
            all_models = []
            for gender, models in self.models.items():
                for model_id, model in models.items():
                    model_with_gender = model.copy()
                    model_with_gender["gender"] = gender
                    all_models.append(model_with_gender)
            return all_models
        else:
            return []
    
    def get_model(self, model_id, gender):
        """
        Get a specific model by ID and gender
        
        Args:
            model_id: ID of the model
            gender: Gender of the model ('Male' or 'Female')
        
        Returns:
            Dictionary with model details or None if not found
        """
        if gender in self.models and model_id in self.models[gender]:
            return self.models[gender][model_id]
        return None
    
    def get_model_image(self, model_id, gender):
        """
        Get the image for a specific model
        
        Args:
            model_id: ID of the model
            gender: Gender of the model ('Male' or 'Female')
        
        Returns:
            PIL Image object or None if not found
        """
        model = self.get_model(model_id, gender)
        if model and os.path.exists(model["path"]):
            return Image.open(model["path"])
        return None
    
    def get_model_names_for_dropdown(self, gender=None):
        """
        Get a formatted list of model names for a dropdown menu
        
        Args:
            gender: Filter by gender ('Male' or 'Female'). If None, return all models.
        
        Returns:
            List of strings in format "Gender: Name"
        """
        models = self.get_model_list(gender)
        if gender:
            return [f"{model['name']}" for model in models]
        else:
            return [f"{model['gender']}: {model['name']}" for model in models]


# Example usage
if __name__ == "__main__":
    manager = ModelManager()
    all_models = manager.get_model_list()
    print(f"Found {len(all_models)} predefined models")
    
    # Print all available models
    for model in all_models:
        print(f"- {model['gender']}: {model['name']} ({model['id']})")