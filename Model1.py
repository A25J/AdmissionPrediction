import os
import pickle
import warnings
from colorama import Fore

warnings.filterwarnings("ignore")
scriptFolder = os.path.dirname(os.path.abspath(__file__))
regressorFileName = "PredictKfolds.sav"
regressorFilePath = scriptFolder + "/" + regressorFileName
bestRegressor = pickle.load(open(regressorFilePath, "rb"))
i = 401
take = True
while take:
    gre = input("GRE Score /340: ")
    if gre.lower()=='end':
        take=False
        break
    gre = int(gre)
    toefl = int(input("TOEFL score /120: "))
    rating = int(input("University Rating: "))
    sop = float(input("SOP score (1-5): "))
    lor = float(input("LoR score (1-5): "))
    CGPA = float(input("CGPA score /10: "))
    Research = int(input("Research experience (0 or 1): "))
    prediction = bestRegressor.predict([[i,gre, toefl, rating, sop, lor, CGPA, Research]])
    #prediction = round(prediction, 4)
    print("Prediction using the model with cross validation")
    print("--------------------------------------------------")
    print("GRE: ",gre, "TOEFL Score: ",toefl, "University Rating: ",rating,
          "CGPA: ",CGPA,"SoP: ",sop, "loR: ", lor, "Reasearch: ", Research)
    print(Fore.GREEN,"The model has predicted: ", Fore.RESET, prediction)
    if prediction > 0.5:
        print(Fore.BLUE, "You have a high chance to be accepted for being accepted graduate degrees",Fore.RESET)
    else:
        print(Fore.RED, "You have a low chance to be accepted for being accepted in graduate degrees", Fore.RESET)
    i+=1

