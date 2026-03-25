import torch
import shap
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data_utils import DRDataset


from model_architecture import VGG16_CBAM, load_trained_model




class Config:
    """Configuration for XGBoost pipeline"""
    
    weights_path = 'best_vgg16_cbam_dr_130epochs.pth'  
    
    
    train_csv = 'D:/train_1.csv'
    test_csv = 'D:/test.csv'
    train_img_path = 'D:/train_images/train_images'
    test_img_path = 'D:/test_images/test_images'
    valid_csv = 'D:/valid.csv'
    valid_img_path = 'D:/val_images/val_images'
    
    
    num_classes = 5
    batch_size = 32
    
    
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': 5,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'min_child_weight': 1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'tree_method': 'hist',
    }
    xgb_num_rounds = 500
    xgb_early_stopping = 50
    
    
    xgb_model_path = 'xgboost_dr_model.json'
    submission_path = 'submission_xgboost.csv'
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class FeatureExtractor(nn.Module):
    """
    Extract deep features from VGG16-CBAM
    Removes final classification layer to get 1024-dim feature vectors
    """
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = model.features
        self.cbam1 = model.cbam1
        self.cbam2 = model.cbam2
        self.cbam3 = model.cbam3
        self.avgpool = model.avgpool
        
        
        self.feature_layers = nn.Sequential(*list(model.classifier.children())[:-1])
    
    def forward(self, x):
        
        for i in range(10):
            x = self.features[i](x)
        
        for i in range(10, 17):
            x = self.features[i](x)
        x = self.cbam1(x)
        
        for i in range(17, 24):
            x = self.features[i](x)
        x = self.cbam2(x)
        
        for i in range(24, len(self.features)):
            x = self.features[i](x)
        x = self.cbam3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.feature_layers(x)
        
        return x


def extract_features(model, dataloader, device):
    """
    Extract deep features from a dataloader
    
    Returns:
        features: numpy array of shape (num_samples, 1024)
        labels: numpy array of labels (or None for test set)
        image_ids: list of image IDs (for test set)
    """
    model.eval()
    all_features = []
    all_labels = []
    all_image_ids = []
    
    print("Extracting features...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Processing'):
            images, labels_or_ids = batch
            images = images.to(device)
            
           
            features = model(images)
            all_features.append(features.cpu().numpy())
            
           
            if isinstance(labels_or_ids, torch.Tensor):
                all_labels.append(labels_or_ids.cpu().numpy())
            else:
                all_image_ids.extend(labels_or_ids)
    
    features = np.vstack(all_features)
    
    if all_labels:
        labels = np.concatenate(all_labels)
        return features, labels, None
    else:
        return features, None, all_image_ids


def build_dataloaders_from_csv(config, apply_preprocessing=True, train_transform=None, val_transform=None):
    """Build train/validation/test dataloaders using the notebook `DRDataset` implementation."""
   
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    if not os.path.exists(config.train_csv):
        raise FileNotFoundError(f"Train CSV not found: {config.train_csv}")
    train_df = pd.read_csv(config.train_csv)
    if 'diagnosis' not in train_df.columns or 'id_code' not in train_df.columns:
        raise ValueError("train_csv must contain 'id_code' and 'diagnosis' columns")

    
    if getattr(config, 'valid_csv', None) and os.path.exists(config.valid_csv):
        val_df = pd.read_csv(config.valid_csv)
        if 'id_code' not in val_df.columns:
            raise ValueError("valid_csv must contain 'id_code' column")
        val_img_path = getattr(config, 'valid_img_path', None) or config.train_img_path
        train_dataset = DRDataset(train_df, config.train_img_path, transform=train_transform, apply_preprocessing=apply_preprocessing, is_test=False)
        valid_dataset = DRDataset(val_df, val_img_path, transform=val_transform, apply_preprocessing=apply_preprocessing, is_test=False)
    else:
        
        train_df_split, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['diagnosis'], random_state=42)
        train_dataset = DRDataset(train_df_split, config.train_img_path, transform=train_transform, apply_preprocessing=apply_preprocessing, is_test=False)
        valid_dataset = DRDataset(val_df, config.train_img_path, transform=val_transform, apply_preprocessing=apply_preprocessing, is_test=False)

   
    if not os.path.exists(config.test_csv):
        raise FileNotFoundError(f"Test CSV not found: {config.test_csv}")
    test_df = pd.read_csv(config.test_csv)
    if 'id_code' not in test_df.columns:
        raise ValueError("test_csv must contain 'id_code' column")
    if 'diagnosis' not in test_df.columns:
        raise ValueError("test_csv must contain 'diagnosis' column for labeled evaluation")
    test_dataset = DRDataset(test_df, config.test_img_path, transform=val_transform, apply_preprocessing=apply_preprocessing, is_test=False)

    trainloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return trainloader, validloader, testloader




def train_xgboost(X_train, y_train, X_val, y_val, config):
    """Train XGBoost classifier"""
    print("\n" + "=" * 60)
    print("Training XGBoost Classifier")
    print("=" * 60)
    
    print("Parameters:")
    for key, value in config.xgb_params.items():
        print(f"  {key}: {value}")
    
   
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    
    eval_results = {}
    model = xgb.train(
        config.xgb_params,
        dtrain,
        num_boost_round=config.xgb_num_rounds,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=config.xgb_early_stopping,
        evals_result=eval_results,
        verbose_eval=20
    )
    
    print(f"\n✓ Training complete!")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Best score: {model.best_score:.4f}")
    
    return model, eval_results


def evaluate_xgboost(model, X, y, dataset_name='Validation'):
    """Evaluate XGBoost model"""
    print("\n" + "=" * 60)
    print(f"Evaluating on {dataset_name} Set")
    print("=" * 60)
    
    
    dtest = xgb.DMatrix(X)
    probs = model.predict(dtest)                    
    y_pred = np.argmax(probs, axis=1)               
    
   
    accuracy = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred, weights='quadratic')
    cm = confusion_matrix(y, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Quadratic Kappa Score: {kappa:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=class_names))
    
    return accuracy, kappa, cm


def predict_test_set(model, X_test, image_ids, output_path):
    """Generate predictions for test set"""
    print("\n" + "=" * 60)
    print("Generating Test Set Predictions")
    print("=" * 60)
    
    dtest = xgb.DMatrix(X_test)
    probs = model.predict(dtest)
    predictions = np.argmax(probs, axis=1)          
    
    submission = pd.DataFrame({
        'id_code': image_ids,
        'diagnosis': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")
    print(f"Total predictions: {len(submission)}")
    print("\nPrediction distribution:")
    print(submission['diagnosis'].value_counts().sort_index())
    
    return submission




def main():
    """Main XGBoost pipeline"""
    config = Config()
    
    print("=" * 60)
    print("XGBoost Classifier for Diabetic Retinopathy")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Weights: {config.weights_path}")
    
   
    print("\n" + "=" * 60)
    print("Step 1: Loading VGG16-CBAM Model")
    print("=" * 60)
    
    vgg_model = load_trained_model(
        weights_path=config.weights_path,
        num_classes=config.num_classes,
        device=config.device
    )
    print("✓ Model loaded successfully!")
    
    
    print("\n" + "=" * 60)
    print("Step 2: Creating Feature Extractor")
    print("=" * 60)
    
    feature_extractor = FeatureExtractor(vgg_model)
    feature_extractor = feature_extractor.to(config.device)
    feature_extractor.eval()
    print("✓ Feature extractor ready!")
    
    
    print("\n" + "=" * 60)
    print("Step 3: Extracting Features")
    print("=" * 60)
    print("NOTE: Make sure you have defined trainloader, validloader, testloader")
    print("      before running this script!\n")
    
    
    if 'trainloader' in globals() and 'validloader' in globals() and 'testloader' in globals():
        trainloader = globals()['trainloader']
        validloader = globals()['validloader']
        testloader = globals()['testloader']
        print("Using dataloaders from the current environment.")
    else:
        print("Dataloaders not found in environment; attempting to build from CSV and image paths...")
        try:
            trainloader, validloader, testloader = build_dataloaders_from_csv(config)
            print("✓ Dataloaders built from CSV/image paths")
        except Exception as e:
            print(f"\n❌ ERROR while building dataloaders: {e}")
            print("\nPlease either define `trainloader`, `validloader`, and `testloader` in the environment (e.g. a notebook),")
            print("or ensure the CSV/image paths in `Config` are correct so this script can build them automatically.")
            return None, None, None, None, None, None, None

    
    test_df = pd.read_csv(config.test_csv)
    test_ids = test_df['id_code'].tolist()

    
    X_train, y_train, _ = extract_features(feature_extractor, trainloader, config.device)
    print(f"✓ Training features: {X_train.shape}")
    
    X_val, y_val, _ = extract_features(feature_extractor, validloader, config.device)
    print(f"✓ Validation features: {X_val.shape}")
    
    X_test, y_test, _ = extract_features(feature_extractor, testloader, config.device)
    print(f"✓ Test features: {X_test.shape}")
    
    
    print("\n" + "=" * 60)
    print("Step 4: Training XGBoost")
    print("=" * 60)
    
    xgb_model, eval_results = train_xgboost(X_train, y_train, X_val, y_val, config)
    
    
    print("\n" + "=" * 60)
    print("Step 5: Evaluation")
    print("=" * 60)
    
    train_acc, train_kappa, train_cm = evaluate_xgboost(xgb_model, X_train, y_train, 'Training')
    val_acc, val_kappa, val_cm = evaluate_xgboost(xgb_model, X_val, y_val, 'Validation')
    test_acc, test_kappa, test_cm = evaluate_xgboost(xgb_model, X_test, y_test, 'Test')
    
    
    print("\n" + "=" * 60)
    print("Step 6: Saving XGBoost Model")
    print("=" * 60)
    
    xgb_model.save_model(config.xgb_model_path)
    print(f"✓ Model saved to: {config.xgb_model_path}")
    
    
    submission = predict_test_set(xgb_model, X_test, test_ids, config.submission_path)

    

    print("\n" + "=" * 60)
    print("Step 8: SHAP Explanations")
    print("=" * 60)

    
    image_ids = test_df['id_code'].tolist()

    
    X_sample = X_test[:2]
    y_sample = y_test[:2]

    
    d_sample = xgb.DMatrix(X_sample)
    probs_sample = xgb_model.predict(d_sample)
    pred_sample = np.argmax(probs_sample, axis=1)

    
    feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

    
    print("Preparing background data with 50 random samples...")
    background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]

   
    explainer = shap.TreeExplainer(xgb_model, background)

    
    shap_values = explainer.shap_values(X_sample)

    print("\nSHAP debug info:")
    print(f"  Type of shap_values: {type(shap_values)}")
    if isinstance(shap_values, list):
      print(f"  Length of shap_values list: {len(shap_values)}")
      if len(shap_values) > 0:
        print(f"  Shape of shap_values[0]: {shap_values[0].shape}")
    else:
        print(f"  Shape of shap_values: {shap_values.shape}")
    print(f"  Shape of expected_value: {np.array(explainer.expected_value).shape}")
    print(f"  Predictions for sample: {pred_sample}")

    
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

    
    for i in range(2):
        pred = pred_sample[i]
        true = y_sample[i]

        print(f"\nExplaining Image {i+1}: {image_ids[i]}")
        print(f"True Label: {class_names[true]} ({true})")
        print(f"Predicted Label: {class_names[pred]} ({pred})")

       
        explanation = shap.Explanation(
            values = shap_values[i, :, pred],
            base_values=explainer.expected_value[pred],
            data=X_sample[i],
            feature_names=feature_names
        )

        
        shap.plots.bar(explanation, max_display=20, show=False)
        plt.title(f"SHAP Feature Contributions for Image {image_ids[i]} - Predicted Class: {class_names[pred]}")
        plot_path = f"shap_bar_{image_ids[i]}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"✓ SHAP bar plot saved to: {plot_path}")
        print("This plot shows how each feature pushes the prediction away from the base value toward the predicted class.")
        print("Positive SHAP values increase the prediction toward this class; negative values decrease it.")

    print("\nSHAP analysis complete! Check the saved plots for visual explanations.") 
   
   
    print("\n" + "=" * 60)
    print("Pipeline Complete! 🎉")
    print("=" * 60)
    print(f"Training Accuracy: {train_acc:.4f} | Kappa: {train_kappa:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f} | Kappa: {val_kappa:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} | Kappa: {test_kappa:.4f}")
    print("\nOutput files:")
    print(f"  1. {config.xgb_model_path}")
    print(f"  2. {config.submission_path}")
    print("=" * 60)

    
    return xgb_model, X_train, y_train, X_val, y_val, X_test, test_ids


if __name__ == "__main__":
    main()