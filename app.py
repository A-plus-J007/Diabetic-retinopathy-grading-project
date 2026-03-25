from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import xgboost as xgb
from model_architecture import load_trained_model
from XGBclassifier import FeatureExtractor as XGBFeatureExtractor, Config as XGBConfig

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


DR_STAGES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "PDR"
}


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out)) * x

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_planes, ratio)
        self.spatial_att = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class VGG16_CBAM(nn.Module):
    def __init__(self, vgg_model, num_classes=5):
        super().__init__()
        self.features = vgg_model.features
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)
        self.cbam5 = CBAM(512)
        num_ftrs = vgg_model.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 16:
                x = self.cbam3(x)
            elif i == 23:
                x = self.cbam4(x)
            elif i == 30:
                x = self.cbam5(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.cbam3 = model.cbam3
        self.cbam4 = model.cbam4
        self.cbam5 = model.cbam5

    def forward(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 16:
                x = self.cbam3(x)
            elif i == 23:
                x = self.cbam4(x)
            elif i == 30:
                x = self.cbam5(x)
        x = torch.flatten(x, 1)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = None
xgb_model = None

def load_models():
    """Load the VGG16+CBAM model (as used in XGB training) and XGBoost classifier"""
    global feature_extractor, xgb_model

    # Get config used for XGB training (to reuse weights path and num_classes)
    try:
        cfg = XGBConfig()
        weights_path = getattr(cfg, "weights_path", "best_vgg16_cbam_dr_130epochs.pth")
        num_classes = getattr(cfg, "num_classes", 5)
    except Exception:
        weights_path = "best_vgg16_cbam_dr_130epochs.pth"
        num_classes = 5

    # Load VGG16+CBAM exactly as in the XGB pipeline
    try:
        model = load_trained_model(weights_path=weights_path, num_classes=num_classes, device=device)
        feature_extractor = XGBFeatureExtractor(model).to(device)
        feature_extractor.eval()
        print(f"VGG16+CBAM feature extractor loaded from {weights_path}.")
    except Exception as e:
        print(f"Error loading VGG16+CBAM model from {weights_path}: {e}")
        feature_extractor = None

    # Load XGBoost classifier
    xgb_path = "xgboost_dr_model.json"
    if os.path.exists(xgb_path):
        try:
            xgb_model = xgb.Booster()
            xgb_model.load_model(xgb_path)
            print("XGBoost DR classifier loaded.")
        except Exception as e:
            print(f"Error loading XGBoost model from {xgb_path}: {e}")
            xgb_model = None
    else:
        print(f"Warning: {xgb_path} not found. Please train and export the XGBoost classifier first.")
        xgb_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_features(img_path):
    """Extract features from image using VGG16+CBAM"""
    if feature_extractor is None:
        raise ValueError("Feature extractor not loaded")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to open image: {e}")
    
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feature_extractor(img).detach().cpu().numpy().flatten()
    return feat

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'}), 400
    
    if xgb_model is None:
        return jsonify({'error': 'XGBoost classifier not loaded. Please train it first.'}), 500
    
    try:
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        features = extract_features(filepath).reshape(1, -1)
        dmat = xgb.DMatrix(features)
        probs = xgb_model.predict(dmat)[0]
        prediction = int(np.argmax(probs))
        probabilities = probs
        
        
        stage_name = DR_STAGES.get(prediction, "Unknown")
        
        
        prob_dict = {DR_STAGES[i]: float(prob) for i, prob in enumerate(probabilities)}
        
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'stage': stage_name,
            'probabilities': prob_dict
        })
    
    except Exception as e:
        
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': feature_extractor is not None and xgb_model is not None
    })

if __name__ == '__main__':
    print("Loading models...")
    load_models()
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)

