"""
Supervised Learning Pipeline for Power System Fault Classification
Implements neural network-based binary classification of power line states.
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config

# Set device preference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch using device: {device}")

class PowerFlowDataset(Dataset):
    """Dataset class for power flow fault detection."""
    
    def __init__(self, data_file):
        self.dataset = pd.read_csv(data_file)
        self._validate_dataset()
        self.feature_cols = self._prepare_features()

    def _validate_dataset(self):
        """Ensure dataset has required columns."""
        required_cols = ['delta_power', 'label', 'line_name']
        missing_cols = [col for col in required_cols if col not in self.dataset.columns]
        if missing_cols:
            raise ValueError(f"Dataset missing required columns: {missing_cols}")

    def _prepare_features(self):
        """Prepare feature columns for model input."""
        feature_cols = [col for col in self.dataset.columns if col.startswith('delta_')]
        
        # Ensure minimum feature count for BatchNorm compatibility
        if len(feature_cols) == 1:
            self.dataset['auxiliary_feature'] = 0.0
            feature_cols.append('auxiliary_feature')
            
        return feature_cols

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Extract features and handle missing values
        features = self.dataset.loc[idx, self.feature_cols]
        features = pd.to_numeric(features, errors='coerce').fillna(0.0)
        
        # Convert to tensors
        feature_tensor = torch.tensor(features.values, dtype=torch.float32)
        label = int(self.dataset.loc[idx, 'label'])
        line_name = self.dataset.loc[idx, 'line_name']
        
        return feature_tensor, label, line_name

class FaultClassifier(nn.Module):
    """Neural network for power system fault classification."""
    
    def __init__(self, input_size):
        super().__init__()
        
        # Ensure minimum input size for stable training
        if input_size < 2:
            input_size = 2
            
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.network(x)

class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.SupervisedConfig.LEARNING_RATE
        )

    def train_epoch(self, dataloader):
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0
        
        for features, labels, _ in dataloader:
            features, labels = features.to(self.device), labels.to(self.device)
            
            # Forward pass
            predictions = self.model(features)
            loss = self.loss_fn(predictions, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def evaluate(self, dataloader, return_detailed=False):
        """Evaluate model performance."""
        self.model.eval()
        all_predictions, all_labels, all_lines = [], [], []
        
        with torch.no_grad():
            for features, labels, lines in dataloader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                predictions = self.model(features)
                predicted_classes = predictions.argmax(1)
                
                # Move back to CPU for analysis
                all_predictions.extend(predicted_classes.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_lines.extend(lines)

        # Calculate metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"Accuracy: {accuracy*100:.1f}% | F1-Score: {f1:.4f}")
        
        if return_detailed:
            return all_labels, all_predictions, all_lines, f1
        return accuracy, f1

def create_confusion_matrix(y_true, y_pred, save_path):
    """Generate and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    fig, ax = plt.subplots(figsize=config.PlotConfig.CONFUSION_MATRIX_SIZE)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=['Normal', 'Fault']
    )
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    
    accuracy = np.sum(np.array(y_pred) == np.array(y_true)) / len(y_true)
    ax.set_title(f"Model Performance (Accuracy: {accuracy*100:.1f}%)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PlotConfig.DPI)
    plt.close()

def run_supervised_pipeline(epochs=None):
    """Execute complete supervised learning pipeline."""
    if epochs is None:
        epochs = config.SupervisedConfig.EPOCHS
        
    start_time = time.time()
    
    # Load and prepare data
    print("Loading dataset...")
    dataset = PowerFlowDataset(str(config.FilePaths.LABELED_DATASET))
    
    # Create train/test split
    train_size = int(config.SupervisedConfig.TRAIN_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.SupervisedConfig.BATCH_SIZE, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=len(test_dataset), 
        shuffle=False
    )
    
    # Initialize model and trainer
    input_size = len(dataset.feature_cols)
    model = FaultClassifier(input_size)
    trainer = ModelTrainer(model, device)
    
    print(f"Model initialized with {input_size} input features")
    print(f"Training on {train_size} samples, testing on {test_size} samples")
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.4f}")
            trainer.evaluate(test_loader)
    
    # Final evaluation with detailed results
    print("\nFinal evaluation:")
    y_true, y_pred, line_names, final_f1 = trainer.evaluate(
        test_loader, return_detailed=True
    )
    
    # Save results for analysis
    print("Saving model and results...")
    
    # Model state
    torch.save(model.state_dict(), str(config.FilePaths.SUPERVISED_MODEL))
    
    # Prediction results
    np.save(str(config.FilePaths.SUPERVISED_TRUE), y_true)
    np.save(str(config.FilePaths.SUPERVISED_PRED), y_pred)
    
    # Detailed test results
    results_df = pd.DataFrame({
        'line_name': line_names,
        'true_label': y_true,
        'pred_label': y_pred
    })
    results_df.to_csv(str(config.FilePaths.SUPERVISED_RESULTS), index=False)
    
    # Generate confusion matrix
    create_confusion_matrix(
        y_true, y_pred, 
        str(config.FilePaths.CONFUSION_MATRIX)
    )
    
    # Save performance metrics
    training_time = time.time() - start_time
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    metrics = {
        "training_time_sec": training_time,
        "f1_score": final_f1,
        "model_parameters": model_params
    }
    np.save(str(config.FilePaths.SUPERVISED_METRICS), metrics)
    
    print(f"Training completed in {training_time:.1f} seconds")
    print(f"Model saved with {model_params:,} trainable parameters")
    print("All results saved to output directory")

if __name__ == "__main__":
    run_supervised_pipeline()