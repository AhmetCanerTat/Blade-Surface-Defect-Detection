import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import torch
import os
import pandas as pd
from torchinfo import summary

def log_before_training(num_epochs, lr_rate, batch_size, criterion, model):

    log_model_params(num_epochs, lr_rate, batch_size, criterion)
    log_model_summary(model)

def log_after_training(model, test_loader, device, run_id):
    log_model_with_signature(model, test_loader, device, run_id)
    
def log_after_evaluation(report, unique_labels):
    log_classification_table(report, unique_labels)
    log_confusion_matrix()

   
    
def log_model_params(num_epochs, lr_rate, batch_size, criterion):
    params ={
        "epoch":num_epochs,
        "lr_rate":lr_rate,
        "batch_size":batch_size,
        "loss_function":str(criterion),
    }
    
    mlflow.log_params(params)
    
def log_model_summary(model):
    """Log model summary as an artifact."""
    os.makedirs("artifacts", exist_ok=True)

    model_summary_path = os.path.join("artifacts", "model_summary.txt")
    with open(model_summary_path, "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact(model_summary_path)
    
def log_model_with_signature(model,test_loader,device,run_id):
    model.eval()
    with torch.no_grad():
        # Get a sample batch from test loader for input example
        sample_batch = next(iter(test_loader))
        sample_input, _ = sample_batch
        sample_input = sample_input.to(device)
        
        # Get model prediction for signature
        sample_output = model(sample_input)
        
        # Convert to numpy for MLflow (use CPU tensors)
        input_example = sample_input[:1].cpu().numpy()  # Take only first sample
        model_output = sample_output[:1].cpu().numpy()  # Take only first prediction
        
        # Infer signature
        signature = infer_signature(input_example, model_output)
    
    # Log model with signature and input example
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=f"EfficientNetV2S_BladeDefect_{run_id[:8]}"  # Optional: register model
    )
    
def log_classification_table(report, unique_labels, artifacts_dir="artifacts"):
    """Log classification metrics as a table to MLflow."""
    macro = "macro avg"
    weighted = "weighted avg"
    
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Create table data
    table_data = []
    for label in unique_labels:
        table_data.append({
            'Class': label,
            'Precision': round(report[label]['precision'], 3),
            'Recall': round(report[label]['recall'], 3),
            'F1-Score': round(report[label]['f1-score'], 3),
            'Support': report[label]['support']
        })
    
    # Add macro and weighted averages
    table_data.append({
        'Class': macro,
        'Precision': round(report[macro]['precision'], 3),
        'Recall': round(report[macro]['recall'], 3),
        'F1-Score': round(report[macro]['f1-score'], 3),
        'Support': report[macro]['support']
    })
    
    # Add accuracy row
    table_data.append({
        'Class': 'accuracy',
        'Precision': '',  # Not applicable for accuracy
        'Recall': '',     # Not applicable for accuracy
        'F1-Score': round(report['accuracy'], 3),
        'Support': report[macro]['support']  # Total support
    })
    
    table_data.append({
        'Class': weighted,
        'Precision': round(report[weighted]['precision'], 3),
        'Recall': round(report[weighted]['recall'], 3),
        'F1-Score': round(report[weighted]['f1-score'], 3),
        'Support': report[weighted]['support']
    })
    
    # Create DataFrame and log as table
    df = pd.DataFrame(table_data)
    
    # Log as MLflow table
    mlflow.log_table(data=df, artifact_file="artifacts/classification_report.json")
    
    # Also save as CSV artifact
    classification_csv_path = os.path.join(artifacts_dir, "classification_report.csv")
    df.to_csv(classification_csv_path, index=False)
    mlflow.log_artifact(classification_csv_path)
    
def log_confusion_matrix():
    mlflow.log_artifact("artifacts/confusion_matrix.png")
    