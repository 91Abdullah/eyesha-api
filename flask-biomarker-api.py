import PIL
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import base64
import io
import torch
import torchvision.transforms as transforms
import cv2
from torch.cuda import is_available
import logging
from dataclasses import dataclass
from enum import Enum
import traceback
from typing import List, Dict, Union, Optional
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights, mobilenet_v2
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define the device
DEVICE = torch.device('cuda' if is_available() else 'cpu')

class_mapping = {
    0: 'No DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferative DR'
}

# Model paths mapping
MODEL_PATHS = {
    "Age": '/mnt/d/PhD_Research/CVD Models/Age/mobilenetv3large_regressor_best.pth',
    "Gender": '/mnt/d/PhD_Research/CVD Models/Gender/mobilenetv3large_regressor_best.pth',
    "BMI": '/mnt/d/PhD_Research/CVD Models/BMI/mobilenetv3large_regressor_best.pth',
    "Diastolic Blood Pressure": '/mnt/d/PhD_Research/CVD Models/BP_OUT_CALC_AVG_DIASTOLIC_BP/mobilenetv3large_regressor_best.pth',
    "Systolic Blood Pressure": '/mnt/d/PhD_Research/CVD Models/BP_OUT_CALC_AVG_SYSTOLIC_BP/mobilenetv3large_regressor_best.pth',
    "Total Cholesterol": '/mnt/d/PhD_Research/CVD Models/Cholesterol Total/mobilenetv3large_regressor_best.pth',
    "Creatinine": '/mnt/d/PhD_Research/CVD Models/Creatinine/mobilenetv3large_regressor_best.pth',
    "Estradiol": '/mnt/d/PhD_Research/CVD Models/Estradiol/mobilenetv3large_regressor_best.pth',
    "Glucose": '/mnt/d/PhD_Research/CVD Models/Glucose/mobilenetv3large_regressor_best.pth',
    "HbA1c": '/mnt/d/PhD_Research/CVD Models/HBA 1C %/mobilenetv3large_regressor_best.pth',
    "HDL-Cholesterol": '/mnt/d/PhD_Research/CVD Models/HDL-Cholesterol/mobilenetv3large_regressor_best.pth',
    "Hematocrit": '/mnt/d/PhD_Research/CVD Models/Hematocrit/mobilenetv3large_regressor_best.pth',
    "Hemoglobin": '/mnt/d/PhD_Research/CVD Models/Hemoglobin/mobilenetv3large_regressor_best.pth',
    "Insulin": '/mnt/d/PhD_Research/CVD Models/Insulin/mobilenetv3large_regressor_best.pth',
    "LDL-Cholesterol": '/mnt/d/PhD_Research/CVD Models/LDL-Cholesterol Calc/mobilenetv3large_regressor_best.pth',
    "Red Blood Cell": '/mnt/d/PhD_Research/CVD Models/Red Blood Cell/mobilenetv3large_regressor_best.pth',
    "SHBG": '/mnt/d/PhD_Research/CVD Models/SexHormone Binding Globulin/mobilenetv3large_regressor_best.pth',
    "Testosterone": '/mnt/d/PhD_Research/CVD Models/Testosterone Total/mobilenetv3large_regressor_best.pth',
    "Triglyceride": '/mnt/d/PhD_Research/CVD Models/Triglyceride/mobilenetv3large_regressor_best.pth'
}

class ImageProcessor:
    """Handles image processing operations"""
    
    def __init__(self, target_size=(540, 540), scale=300):
        self.target_size = target_size
        self.scale = scale
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def crop_image_from_gray(self, img, tol=7):
        """
        This function from:
        https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping
        """
        if img.ndim ==2:
            mask = img>tol
            return img[np.ix_(mask.any(1),mask.any(0))]
        elif img.ndim==3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol

            check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
            if (check_shape == 0): # image is too dark so that we crop out everything,
                return img # return original image
            else:
                img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
                img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
                img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
        #         print(img1.shape,img2.shape,img3.shape)
                img = np.stack([img1,img2,img3],axis=-1)
        #         print(img.shape)
            return img


    def center_crop(self, image: PIL.Image):
        """
        Only gets center square (of rectangular images) - no resizing
        => diffently sized square images
        """
        old_width, old_heigh = image.size
        new_size = min(old_width, old_heigh)

        margin_x = (old_width - new_size) // 2
        margin_y = (old_heigh - new_size) // 2

        left   = margin_x
        right  = margin_x + new_size
        top    = margin_y
        bottom = margin_y + new_size

        return image.crop( (left, top, right, bottom) )


    def process_image_ratio_invariant(self, cv2_image, size=256, do_center_crop=True):

        image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        image = self.crop_image_from_gray(image)
        #image = cv2.resize(image, (size, size))  # this would distort eyeball shape

        if do_center_crop is False:
            return image

        cv_to_pil = transforms.ToPILImage()
        # crop the largest possible square from the center
        pil_img = cv_to_pil(image)
        pil_img = self.center_crop(pil_img)
        image   = np.array(pil_img).copy()

        # now we have quadratic, but differently sized images
        # => resize without altering the shape of the eyeball
        image = cv2.resize(image, (size, size))

        # add gaussian blur with sigma proportional to new size:
        image = cv2.addWeighted (image, 4, cv2.GaussianBlur(image, (0, 0) , size/30) , -4 ,128)

        return cv_to_pil(image)

    def preprocess_image_dr(self, image_data, size=256):
        # Load the image
        try:
            if isinstance(image_data, str):
                # Handle base64 string
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data

            # Convert to numpy array for OpenCV processing
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Ensure image is in uint8 format
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)

             # Apply the same preprocessing as during training
            image = self.process_image_ratio_invariant(image, size=size)

            # Convert to tensor and normalize
            transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            image = transform(image)
            image = image.unsqueeze(0)  # Add batch dimension
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")

    def scaleRadius(self, img, scale):
        """Scale image to a given radius"""
        center_row = img.shape[0] // 2
        x = img[int(center_row), :, :].sum(1)
        r = (x > x.mean() / 10).sum() / 2
        s = scale * 1.0 / r
        return cv2.resize(img, (0, 0), fx=s, fy=s)

    def process_fundus_image(self, img):
        """Apply fundus-specific image processing"""
        try:
            # Ensure image is in uint8 format
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            
            # Scale image to a given radius
            img = self.scaleRadius(img, self.scale)
            
            # Ensure scaled image is in uint8 format
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            
            # Subtract local mean color
            gaussian_blur = cv2.GaussianBlur(img, (0, 0), self.scale / 30)
            img = cv2.addWeighted(img, 4, gaussian_blur, -4, 128)
            
            # Ensure result is in uint8 format
            img = img.clip(0, 255).astype(np.uint8)
            
            # Remove outer 10%
            mask = np.zeros(img.shape, dtype=np.uint8)
            center = (int(img.shape[1] / 2), int(img.shape[0] / 2))
            cv2.circle(mask, center, int(self.scale * 0.9), (1, 1, 1), -1, 8, 0)
            img = img * mask + 128 * (1 - mask)
            
            # Final check for uint8
            img = img.clip(0, 255).astype(np.uint8)
            
            return img
            
        except Exception as e:
            logger.error(f"Error in process_fundus_image: {str(e)}")
            raise ValueError(f"Failed to process fundus image: {str(e)}")

    def preprocess_image(self, image_data: Union[str, bytes]) -> torch.Tensor:
        """
        Preprocess image data for model inference
        
        Args:
            image_data: Either base64 string or bytes of image
            
        Returns:
            Preprocessed image as torch tensor
        """
        try:
            if isinstance(image_data, str):
                # Handle base64 string
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data

            # Convert to numpy array for OpenCV processing
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Ensure image is in uint8 format
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            
            # Apply fundus-specific preprocessing
            processed_img = self.process_fundus_image(img)
            
            # Resize to target size
            processed_img = cv2.resize(processed_img, self.target_size)
            
            # Convert to PIL Image for PyTorch transforms
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(processed_img.astype('uint8'))
            
            # Apply normalization and convert to tensor
            image_tensor = self.transform(pil_image)
            return image_tensor.unsqueeze(0).to(DEVICE)
        
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Failed to preprocess image: {str(e)}")

class DiseasePredictor:
    def __init__(self):
        self.model = mobilenet_v2(weights=None)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(num_ftrs, 5)  # Assuming 5 classes for APTOS 2019

    def _load_model(self):
        self.model.load_state_dict(torch.load('/mnt/d/PhD_Research/DR Predictions/MobileNetv2_DR_prediciton Model and code/best_mobilenet_v2.pth', map_location=DEVICE))
        self.model.eval()
        self.model.to(DEVICE)

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Dict[str, float]:
        output = self.model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, preds = torch.max(output, 1)
        predicted_class = int(preds.item())
        predicted_class_name = class_mapping[predicted_class]

        # Print the predicted class and its probability
        print(f'Predicted class: {predicted_class} ({predicted_class_name})')
        print(f'Prediction probability: {probabilities[0][predicted_class].item() * 100:.2f}%')

        # Print probabilities for all classes
        print("\nPrediction probabilities for all classes:")
        class_probabilities = {}
        for class_idx, class_name in class_mapping.items():
            probability = probabilities[0][class_idx].item() * 100
            class_probabilities[class_name] = probability
            print(f'{class_name}: {probability:.2f}%')

        return class_probabilities

class BiomarkerPredictor:
    """Handles biomarker predictions"""
    
    def __init__(self):
        self.models = {}
        self._load_models()

    class MobileNetV3LargeRegressor(torch.nn.Module):
        def __init__(self):
            super(BiomarkerPredictor.MobileNetV3LargeRegressor, self).__init__()
            self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
            self.model.classifier = torch.nn.Identity()  # Remove the fully connected layers
            self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
            self.fc1 = torch.nn.Linear(960, 512)  # Fully connected layer with 512 units
            self.relu = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(0.5)
            self.fc2 = torch.nn.Linear(512, 1)  # Final layer for regression

        def forward(self, x):
            x = self.model.features(x)
            x = self.gap(x)
            x = x.view(x.size(0), -1)  # Flatten the tensor
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    def _create_model(self):
        """Create a MobileNetV3LargeRegressor instance"""
        return BiomarkerPredictor.MobileNetV3LargeRegressor()

    def _load_models(self):
        """Load all biomarker models"""
        logger.info("Loading biomarker models...")
        for biomarker, path in MODEL_PATHS.items():
            try:
                model = self._create_model()
                model.load_state_dict(torch.load(path, map_location=DEVICE))
                model.eval()
                model.to(DEVICE)
                self.models[biomarker] = model
                logger.info(f"Loaded model for {biomarker}")
            except Exception as e:
                logger.error(f"Failed to load model for {biomarker}: {str(e)}")

    @torch.no_grad()
    def predict(self, image: torch.Tensor, biomarker: str) -> float:
        """
        Run inference for a specific biomarker
        
        Args:
            image: Preprocessed image tensor
            biomarker: Name of biomarker to predict
            
        Returns:
            Prediction value
        """
        if biomarker not in self.models:
            raise ValueError(f"Model not loaded for biomarker: {biomarker}")

        model = self.models[biomarker]
        output = model(image)
        
        return float(output.item())

class ValidationError(Exception):
    """Custom exception for input validation errors"""
    pass

def validate_request(data: Dict) -> None:
    """Validate incoming request data"""
    if not data:
        raise ValidationError("Empty request body")
        
    if "images" not in data:
        raise ValidationError("No images provided")
        
    if "models" not in data:
        raise ValidationError("No models specified")
        
    if not isinstance(data["images"], list):
        raise ValidationError("Images must be provided as a list")
        
    if not isinstance(data["models"], list):
        raise ValidationError("Models must be provided as a list")
        
    invalid_models = set(data["models"]) - set(MODEL_PATHS.keys())
    if invalid_models:
        raise ValidationError(f"Unsupported models: {invalid_models}")

def validate_files_request(files, models):
    """Validate file upload request"""
    if not files:
        raise ValidationError("No images provided")
    
    if not models:
        raise ValidationError("No models specified")
    
    if not isinstance(models, list):
        raise ValidationError("Models must be provided as a list")
        
    invalid_models = set(models) - set(MODEL_PATHS.keys())
    if invalid_models:
        raise ValidationError(f"Unsupported models: {invalid_models}")

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint that handles both JSON and form-data"""
    try:
        # Check if the request is multipart/form-data or JSON
        if request.files:
            # Handle form-data upload
            files = request.files.getlist('images')
            
            # Handle models parameter
            models_param = request.form.get('models', '[]')
            diseases_param = request.form.get('diseases', '[]')
            try:
                # Try to parse as JSON if it looks like a JSON array
                if models_param.startswith('['):
                    models = json.loads(models_param)
                else:
                    # If not JSON, treat as single model
                    models = [models_param]
            except json.JSONDecodeError:
                # If JSON parsing fails, treat as single model
                models = [models_param]

            try:
                # Try to parse as JSON if it looks like a JSON array
                if diseases_param.startswith('['):
                    diseases = json.loads(diseases_param)
                else:
                    # If not JSON, treat as single model
                    diseases = [diseases_param]
            except json.JSONDecodeError:
                # If JSON parsing fails, treat as single model
                diseases = [diseases_param]
            
            validate_files_request(files, models)
            # validate_files_request(files, diseases)
            
            # Initialize processors
            image_processor = ImageProcessor()
            predictor = BiomarkerPredictor()
            predict_dr = DiseasePredictor()
            
            results = {"images": []}
            
            # Process each uploaded file
            for idx, file in enumerate(files):
                try:
                    # Read the file data
                    image_data = file.read()
                    
                    # Preprocess image
                    processed_image = image_processor.preprocess_image(image_data)
                    processed_dr = image_processor.preprocess_image_dr(image_data)
                    
                    # Get predictions for each requested model
                    predictions = {}
                    predictions_dr = {}

                    for disease in diseases:
                        if disease == 'Diabetic Retinopathy':
                            predictions_dr[disease] = predict_dr.predict(processed_dr)

                    for model_name in models:
                        if model_name == 'Gender':
                            predictions[model_name] = "Male" if bool(predictor.predict(processed_image, model_name)) == True else "Female"
                        elif model_name in ['Age', 'BMI', 'Diastolic Blood Pressure', 'Systolic Blood Pressure']:
                            predictions[model_name] = int(predictor.predict(processed_image, model_name))
                        else:
                            predictions[model_name] = round(predictor.predict(processed_image, model_name), 1)
                    
                    # Add to results
                    results["images"].append({
                        "image_id": f"image_{idx + 1}",
                        "filename": file.filename,
                        "predictions": predictions,
                        "predictions_dr": predictions_dr
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing image {idx + 1}: {str(e)}")
                    results["images"].append({
                        "image_id": f"image_{idx + 1}",
                        "filename": file.filename,
                        "error": str(e)
                    })
            
            return jsonify(results)
        
        else:
            # Handle JSON request (original base64 method)
            data = request.get_json()
            validate_request(data)
            
            # Initialize processors
            image_processor = ImageProcessor()
            predictor = BiomarkerPredictor()
            dr_predictor = DiseasePredictor()
            
            results = {"images": []}
            
            # Process each image
            for idx, image_data in enumerate(data["images"]):
                try:
                    # Preprocess image
                    processed_image = image_processor.preprocess_image(image_data)
                    processed_dr = image_processor.preprocess_image_dr(image_data)
                    
                    # Get predictions for each requested model
                    predictions = {}
                    predictions_dr = {}

                    for model_name in data["diseases"]:
                        if model_name == 'Diabetic Retinopathy':
                            predictions_dr[model_name] = dr_predictor.predict(processed_dr)

                    for model_name in data["models"]:
                        if model_name == 'Gender':
                            predictions[model_name] = "Male" if bool(predictor.predict(processed_image, model_name)) == True else "Female"
                        elif model_name in ['Age', 'BMI', 'Diastolic Blood Pressure', 'Systolic Blood Pressure']:
                            predictions[model_name] = int(predictor.predict(processed_image, model_name))
                        else:
                            predictions[model_name] = round(predictor.predict(processed_image, model_name), 1)
                    
                    # Add to results
                    results["images"].append({
                        "image_id": f"image_{idx + 1}",
                        "predictions": predictions,
                        "predictions_dr": predictions_dr
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing image {idx + 1}: {str(e)}")
                    results["images"].append({
                        "image_id": f"image_{idx + 1}",
                        "error": str(e)
                    })
            
            return jsonify(results)
    
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    
    except Exception as e:
        logger.error(f"Unexpected error: {traceback.format_exc()}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "https://your-production-domain.com"]}})
    app.run(debug=True, port=5001)