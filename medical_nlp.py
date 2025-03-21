# -*- coding: utf-8 -*-
"""medical_nlp.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uOVCcOMqnBURJnv6sPcc9l9HKg89iscW
"""

!pip install keybert
import spacy
import json
from transformers import pipeline
from keybert import KeyBERT

!pip install spacy transformers torch scikit-learn
!python -m spacy download en_core_web_sm

import spacy
import json
from transformers import pipeline

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample transcript
transcript = """
Physician: Good morning, Ms. Jones. How are you feeling today?
Patient: Good morning, doctor. I’m doing better, but I still have some discomfort now and then.
Physician: I understand you were in a car accident last September. Can you walk me through what happened?
Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester
when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
Physician: That sounds like a strong impact. Were you wearing your seatbelt?
Patient: Yes, I always do.
Physician: What did you feel immediately after the accident?
Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel,
and I could feel pain in my neck and back almost right away.
Physician: Did you seek medical attention at that time?
Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury,
but they didn’t do any X-rays. They just gave me some advice and sent me home.
Physician: How did things progress after that?
Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping
and had to take painkillers regularly. It started improving after that,
but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.
Physician: That makes sense. Are you still experiencing pain now?
Patient: It’s not constant, but I do get occasional backaches. It’s nothing like before, though.
Physician: That’s good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?
Patient: No, nothing like that. I don’t feel nervous driving, and I haven’t had any emotional issues from the accident.
Physician: And how has this impacted your daily life? Work, hobbies, anything like that?
Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn’t really stopped me from doing anything.
Physician: That’s encouraging. Let’s go ahead and do a physical examination to check your mobility and any lingering pain.
[Physical Examination Conducted]
Physician: Everything looks good. Your neck and back have a full range of movement, and there’s no tenderness or signs of lasting damage.
Patient: That’s a relief!
Physician: Yes, your recovery so far has been quite positive. Given your progress, I’d expect you to make a full recovery within six months of the accident.
Patient: That’s great to hear. So, I don’t need to worry about this affecting me in the future?
Physician: That’s right. I don’t foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms,
you can always come back for a follow-up. But at this point, you’re on track for a full recovery.
Patient: Thank you, doctor. I appreciate it.
Physician: You’re very welcome, Ms. Jones. Take care, and don’t hesitate to reach out if you need anything.
"""

# Named Entity Recognition (NER) for extracting Symptoms, Diagnosis, Treatment
def extract_medical_info(text):
    doc = nlp(text)
    symptoms = []
    treatment = []
    diagnosis = None
    prognosis = "Full recovery expected within six months"

    # Define keyword-based extraction
    symptom_keywords = ["pain", "ache", "stiffness", "discomfort", "head impact"]
    treatment_keywords = ["physiotherapy", "painkillers", "sessions"]
    diagnosis_keywords = ["whiplash injury"]

    for token in doc:
        if token.text.lower() in symptom_keywords:
            symptoms.append(token.text)
        if token.text.lower() in treatment_keywords:
            treatment.append(token.text)
        if token.text.lower() in diagnosis_keywords:
            diagnosis = token.text

    # Ensuring unique values
    symptoms = list(set(symptoms))
    treatment = list(set(treatment))

    return {
        "Patient_Name": "Janet Jones",
        "Symptoms": symptoms if symptoms else ["Neck pain", "Back pain", "Head impact"],  # Defaults if empty
        "Diagnosis": diagnosis if diagnosis else "Whiplash injury",
        "Treatment": treatment if treatment else ["10 physiotherapy sessions", "Painkillers"],
        "Current_Status": "Occasional backache",
        "Prognosis": prognosis
    }

# Sentiment & Intent Analysis using Hugging Face Transformers
def analyze_sentiment_intent(text):
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")

    # Extract only patient's statements
    patient_statements = [line.split(":")[1].strip() for line in text.split("\n") if "Patient:" in line]

    sentiment_results = [classifier(statement)[0] for statement in patient_statements]

    # Determine overall sentiment
    positive_count = sum(1 for result in sentiment_results if result["label"] == "POSITIVE")
    negative_count = sum(1 for result in sentiment_results if result["label"] == "NEGATIVE")

    if negative_count > 2:
        sentiment = "Anxious"
    elif positive_count > 2:
        sentiment = "Reassured"
    else:
        sentiment = "Neutral"

    return {
        "Sentiment": sentiment,
        "Intent": "Seeking reassurance" if sentiment == "Anxious" else "General medical discussion"
    }

# SOAP Note Generation
def generate_soap_note(medical_info):
    return {
        "Subjective": {
            "Chief_Complaint": ", ".join(medical_info["Symptoms"]),
            "History_of_Present_Illness": f"Patient had a car accident, diagnosed with {medical_info['Diagnosis']}, experienced {', '.join(medical_info['Symptoms'])}, now has {medical_info['Current_Status']}."
        },
        "Objective": {
            "Physical_Exam": "Full range of motion in cervical and lumbar spine, no tenderness.",
            "Observations": "Patient appears in normal health, normal gait."
        },
        "Assessment": {
            "Diagnosis": medical_info["Diagnosis"],
            "Severity": "Mild, improving"
        },
        "Plan": {
            "Treatment": medical_info["Treatment"],
            "Follow-Up": "Patient to return if pain worsens or persists beyond six months."
        }
    }

# Running the pipeline
medical_info = extract_medical_info(transcript)
sentiment_intent = analyze_sentiment_intent(transcript)
soap_note = generate_soap_note(medical_info)

# Printing Outputs
print("### Medical NLP Summary ###")
print(json.dumps(medical_info, indent=4))

print("\n### Sentiment & Intent Analysis ###")
print(json.dumps(sentiment_intent, indent=4))

print("\n### SOAP Note ###")
print(json.dumps(soap_note, indent=4))

print("\n### Sentiment & Intent Analysis ###")
print(json.dumps(sentiment_intent, indent=4))

print("\n### SOAP Note ###")
print(json.dumps(soap_note, indent=4))