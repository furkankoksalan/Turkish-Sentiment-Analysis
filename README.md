# Turkish Sentiment Analysis Model Comparison

A comprehensive comparison framework for Turkish BERT models on sentiment analysis tasks with detailed performance metrics and visualizations.

##  Overview

This project compares multiple Turkish BERT models for sentiment analysis using the Turkish Sentiment Analysis Dataset. It provides detailed performance metrics, visualizations, and statistical comparisons to help identify the best model for Turkish sentiment classification tasks.

##  Features

- **Multi-model comparison**: Evaluate multiple Turkish BERT models simultaneously
- **Comprehensive metrics**: Accuracy, F1-score, Precision, and Recall
- **Rich visualizations**: Bar charts, heatmaps, scatter plots, and trend analysis
- **Early stopping**: Prevent overfitting with configurable patience
- **Export results**: Save comparison results to CSV format
- **Detailed reporting**: Generate comprehensive performance reports

##  Models Tested

- `dbmdz/bert-base-turkish-cased`
- `savasy/bert-base-turkish-sentiment-cased`


##  Requirements

```txt
datasets==2.14.0
pandas==1.5.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
transformers==4.30.0
torch==2.0.1
scikit-learn==1.3.0
```

##  Quick Start

```python
# Import required libraries
import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Run the comparison
python sentiment_comparison.py
```

##  Usage

### Basic Usage

```python
# Load and run the comparison script
from sentiment_comparison import run_model_comparison

# Define models to compare
models = [
    "dbmdz/bert-base-turkish-cased",
    "savasy/bert-base-turkish-sentiment-cased"
]

# Run comparison
results = run_model_comparison(models)
```

### Custom Configuration

```python
# Customize training parameters
training_config = {
    'learning_rate': 2e-5,
    'num_epochs': 8,
    'batch_size': 16,
    'weight_decay': 0.01,
    'early_stopping_patience': 3
}

results = run_model_comparison(models, config=training_config)
```


##  Visualizations

The script creates 4 types of visualizations:

1. **Bar Chart**: Side-by-side comparison of all metrics
2. **Heatmap**: Color-coded performance matrix
3. **Scatter Plot**: Accuracy vs F1 Score relationship
4. **Line Plot**: Metric trends across models

##  Configuration

### Training Parameters

```python
TrainingArguments(
    learning_rate=2e-5,
    num_train_epochs=8,
    per_device_train_batch_size=16,
    weight_decay=0.01,
    warmup_steps=500,
    early_stopping_patience=3
)
```

### Model Parameters

```python
# Tokenization settings
max_length=128
truncation=True
padding='max_length'

# Classification settings
num_labels=2
labels=["Negative", "Positive"]
```

##  Workflow

1. **Data Loading**: Load Turkish sentiment dataset
2. **Preprocessing**: Tokenize and prepare data for each model
3. **Training**: Fine-tune models with early stopping
4. **Evaluation**: Calculate comprehensive metrics
5. **Visualization**: Generate comparison charts
6. **Export**: Save results to CSV

##  Metrics Explained

- **Accuracy**: Overall correct predictions percentage
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

##  Common Issues

### Memory Issues
```python
# Reduce batch size if encountering CUDA OOM
per_device_train_batch_size=8
per_device_eval_batch_size=8
```

### Slow Training
```python
# Use fewer epochs for quick testing
num_train_epochs=4
early_stopping_patience=2
```

### Model Loading Errors
```bash
# Clear cache if models fail to load
rm -rf ~/.cache/huggingface/
```

##  Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


##  Acknowledgments

- Turkish Sentiment Analysis Dataset by [winvoker](https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset)
- Turkish BERT models by [dbmdz](https://huggingface.co/dbmdz) and [savasy](https://huggingface.co/savasy)
- Hugging Face Transformers library

##  Results Summary

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|---------|
| dbmdz/bert-base-turkish-cased | 0.9613 | 0.9609 | 0.9608 | 0.9613 |
| savasy/bert-base-turkish-sentiment-cased | 0.9602 | 0.9597 | 0.9596 | 0.9602 |

##  Future Improvements

### Short-term Goals
- [ ] Add more Turkish BERT models
- [ ] Implement cross-validation
- [ ] Add confusion matrix visualization
- [ ] Support for custom datasets
- [ ] Hyperparameter optimization

### Long-term Goals
- [ ] Enterprise deployment options
- [ ] Custom model fine-tuning
- [ ] Advanced security features
- [ ] Multi-language support
- [ ] Integration with BI tools

##  Contact

- Email: furkan1234koksalan@gmail.com
- LinkedIn:([Furkan Koksalan](https://www.linkedin.com/in/furkan-k%C3%B6ksalan-253515286/))

---
