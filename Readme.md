# Medical NLP Pipeline

This project processes medical text using a Natural Language Processing (NLP) model. It extracts key medical information such as symptoms, diagnosis, treatment, and prognosis. Additionally, it performs sentiment and intent analysis, and generates SOAP notes.

## Features
- **Medical Text Processing**: Extracts patient symptoms, diagnosis, treatment, and prognosis.
- **Sentiment & Intent Analysis**: Identifies the sentiment and intent of medical discussions.
- **SOAP Note Generation**: Structures medical data into Subjective, Objective, Assessment, and Plan format.

## Setup & Installation

### Run on Google Colab

1. Open Google Colab: [Google Colab](https://colab.research.google.com/drive/1uOVCcOMqnBURJnv6sPcc9l9HKg89iscW#scrollTo=QTT0SHe9Svq6)
2. Upload your notebook (`.ipynb`) file or create a new one.
3. Install required dependencies by running:

   ```python
   !pip install transformers datasets torch
   ```

4. (Optional) Authenticate with Hugging Face for model access:

   ```python
   from huggingface_hub import notebook_login
   notebook_login()
   ```

## Model & Pipeline

- **Model**: `distilbert-base-uncased`
- **Library**: Hugging Face Transformers

## Running the Pipeline

Run the following script inside your Colab notebook:

```python
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load model and tokenizer
device = "cpu"
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

print(f"Device set to use {device}")
```

## Expected Output

When executed correctly, the output should be similar to:

```json
{
    "Patient_Name": "Janet Jones",
    "Symptoms": ["stiffness", "pain", "discomfort"],
    "Diagnosis": "Whiplash injury",
    "Treatment": ["painkillers", "sessions", "physiotherapy"],
    "Current_Status": "Occasional backache",
    "Prognosis": "Full recovery expected within six months"
}
```

## Troubleshooting

- If you see a warning about uninitialized weights:
  - It means the classifier layers need training. You can fine-tune the model for better predictions.
- If the Hugging Face token is missing:
  - Generate a token from [Hugging Face](https://huggingface.co/settings/tokens) and authenticate in Colab.

---

Enjoy building your Medical NLP system! 🚀
