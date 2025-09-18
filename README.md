# Face Ethnicity Swap

A Streamlit application that allows users to upload their images and transform their appearance to different ethnicities using AI-powered face models.

## Features

- Upload your image and see it transformed across multiple ethnicities
- Supports both male and female transformations
- Uses advanced AI models for realistic ethnicity transformations
- Simple and intuitive user interface
- Real-time processing and results

## How It Works

1. Upload an image containing a face
2. The application detects and processes the face
3. AI models transform the face to different ethnicities
4. View and compare the results

## Available Ethnicity Models

### Female Models
- African American
- Middle Eastern
- Mixed Race
- Native American
- South Asian

### Male Models
- African American
- Japanese
- Middle Eastern
- Mixed Race
- Spanish

## Technical Details

This application uses a combination of facial recognition, image processing, and AI models to create realistic ethnicity transformations. The models are pre-generated and stored in the models directory.

## Setup and Installation

1. Clone this repository
   ```
   git clone https://github.com/subham0206/face-ethnicity-swap.git
   cd face-ethnicity-swap
   ```

2. Create a virtual environment and install dependencies
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the Streamlit app
   ```
   streamlit run app.py
   ```

## Deployment

This application is deployed on Streamlit Cloud and can be accessed at: [Face Ethnicity Swap App](https://face-ethnicity-swap.streamlit.app) (link will be active after deployment)

## Privacy Note

All uploaded images are processed locally and are not stored permanently. The application does not share or transmit your images to any third-party services.