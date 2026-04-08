import os
import pickle
import numpy as np
import docx
import PyPDF2
from colorama import Fore
from NLPpipeline import extract_all_features, score_document

scriptFolder = os.path.dirname(os.path.abspath(__file__))
regressorFileName = "Predict.sav"
regressorFilePath = scriptFolder + "/" + regressorFileName
bestRegressor = pickle.load(open(regressorFilePath, "rb"))
# -----------------------------
# Utilities
# -----------------------------
def read_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif ext == ".docx":
        doc = docx.Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
    elif ext == ".pdf":
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        raise ValueError("Unsupported file format! Use .txt, .docx, or .pdf")
    return text.strip()


if __name__ == "__main__":

    print("\n=== Admission Evaluation System ===")

    # -------- NLP INPUT --------
    sop_path = input("\nEnter the SOP file path (.txt, .docx, .pdf): ").strip()
    lor_path = input("\nEnter the Letter of Recommendation file path (.txt, .docx, .pdf): ").strip()


    sop_text = read_file(sop_path)
    lor_text = read_file(lor_path)


    gre = float(input("GRE Score / 340: "))
    toefl = float(input("TOEFL Score / 120: "))
    cgpa = float(input("CGPA / 10: "))
    university_rating = float(input("University Rating / 5: "))
    research = float(input("Research Experience (0 or 1): "))
    print("\n======SOP Document Evaluation:======")
    sop_score = score_document(extract_all_features(sop_text))
    print("\n======LOR Document Evaluation:======")
    lor_score = score_document(extract_all_features(lor_text))

    print("\n--- NLP Evaluation ---")
    print(f"SOP Score: {sop_score} / 5")
    print(f"LOR Score: {lor_score} / 5")
    # Feature order MUST match ML training
    X = np.array([
        gre,
        toefl,
        university_rating,
        sop_score,
        lor_score,
        cgpa,
        research
    ]).reshape(1, -1)

    prediction = bestRegressor.predict(X)[0]
    prediction = round(prediction, 2)
    print("\n\nWith the Entered values: \nGRE Score: ", Fore.LIGHTBLUE_EX ,gre,Fore.RESET," /340\nTOEFL Score: ",Fore.LIGHTBLUE_EX , toefl,Fore.RESET,
          "/120\nUniversity Rating: ", Fore.LIGHTBLUE_EX ,university_rating,Fore.RESET,
          "/5\nCGPA Score: ",Fore.LIGHTBLUE_EX , cgpa,Fore.RESET, "/10", "\nResearch Experience: ", Fore.LIGHTBLUE_EX ,research,Fore.RESET)
    print(Fore.GREEN, "\n\nThe model has predicted: ", Fore.RESET, prediction*100, "%")
    if prediction > 0.5:
        print(Fore.BLUE, "You have a high chance to be accepted for being accepted in graduate degrees", Fore.RESET)
    else:
        print(Fore.RED, "You have a low chance to be accepted for being accepted in graduate degrees", Fore.RESET)