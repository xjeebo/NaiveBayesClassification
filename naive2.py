import pandas as pd
import numpy as np
from variables import *
from naiveFunctions import check_pclass,check_sex, check_sibsp, check_parch, check_embarked, get_classification

file = pd.read_csv("TitanicRecords_training_set.csv")
Pclass = file["Pclass"]
Sex = file["Sex"]
SibSp = file["SibSp"]
Parch = file["Parch"]
Survived = file["Survived"]
Embarked = file["Embarked"]

# Acquire a ratio of those who lived and died
for i in range(0,len(Survived)):
    if Survived[i] == 1:
        LivedCnt += 1
    else:
        DeathCnt += 1

R_lived = LivedCnt/len(Survived)
R_death = DeathCnt/len(Survived)

# probabiility based on Ticket Class, Pclass
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
for i in range(0, len(Pclass)):
        if Pclass[i] == 1 and Survived[i] == 1:
            Pclass1LivedCnt += 1
        elif Pclass[i] == 1 and Survived[i] == 0:
            Pclass1DiedCnt += 1
        
        if Pclass[i] == 2 and Survived[i] == 1:
            Pclass2LivedCnt += 1
        elif Pclass[i] == 2 and Survived[i] == 0:
            Pclass2DiedCnt += 1

        if Pclass[i] == 3 and Survived[i] == 1:
            Pclass3LivedCnt += 1
        elif Pclass[i] == 3 and Survived[i] == 0:
            Pclass3DiedCnt += 1

# Porbability Calculation (Pclass)
R_Pclass1Lived =  Pclass1LivedCnt/LivedCnt
R_Pclass1Died = Pclass1DiedCnt/DeathCnt
R_Pclass2Lived =  Pclass2LivedCnt/LivedCnt
R_Pclass2Died = Pclass2DiedCnt/DeathCnt
R_Pclass3Lived =  Pclass3LivedCnt/LivedCnt
R_Pclass3Died = Pclass3DiedCnt/DeathCnt

# Probability based on Sex
for i in range(0, len(Sex)):
        if Sex[i] == "male" and Survived[i] == 1:
            maleLivedCnt += 1
        elif Sex[i] == "male" and Survived[i] == 0:
            maleDiedCnt += 1

        if Sex[i] == "female" and Survived[i] == 1:
            femaleLivedCnt += 1
        elif Sex[i] == "female" and Survived[i] == 0:
            femaleDiedCnt += 1

# probability calculations (sex)
R_maleLived =  maleLivedCnt/LivedCnt
R_maleDied = maleDiedCnt/DeathCnt
R_femaleLived =  femaleLivedCnt/LivedCnt
R_femaleDied = femaleDiedCnt/DeathCnt

# Probability based on if the person had a sibling or spouse with them
for i in range(0,len(SibSp)):
    # they didnt have a sibling or spouse
    if SibSp[i] == 0 and Survived[i] == 1:
        No_SibSpLivedCnt += 1
    elif SibSp[i] == 0 and Survived[i] == 0:   
        No_SibSpDiedCnt += 1

    # they had atleast one sibling or spouse
    if SibSp[i] > 0 and Survived[i] == 1:
        Yes_SibSpLivedCnt += 1
    if SibSp[i] > 0 and Survived[i] == 0:   
        Yes_SibSpDiedCnt += 1
    
# probability calculations (SbSp)
R_NoSibSpLived = No_SibSpLivedCnt/LivedCnt
R_NoSibSpDied = No_SibSpDiedCnt/DeathCnt
R_YesSibSpLived = Yes_SibSpLivedCnt/LivedCnt
R_YesSibSpDied = Yes_SibSpDiedCnt/DeathCnt


# Probability based on if the person had a parent or child with them
for i in range(0,len(Parch)):
    # they didnt have a parent or child
    if Parch[i] == 0 and Survived[i] == 1:
        No_ParchLivedCnt += 1
    elif Parch[i] == 0 and Survived[i] == 0:   
        No_ParchDiedCnt += 1

    # they had atleast one parent or child
    if Parch[i] > 0 and Survived[i] == 1:
        Yes_ParchLivedCnt += 1
    if Parch[i] > 0 and Survived[i] == 0:   
        Yes_ParchDiedCnt += 1
    
# probability calculations (Parch)
R_NoParchLived = No_ParchLivedCnt/LivedCnt
R_NoParchDied = No_ParchDiedCnt/DeathCnt
R_YesParchLived = Yes_ParchLivedCnt/LivedCnt
R_YesParchDied = Yes_ParchDiedCnt/DeathCnt


# probabiility based on port of embarkation
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
for i in range(0, len(Pclass)):
        if Embarked[i] == 'C' and Survived[i] == 1:
            CEmLivedCnt += 1
        elif Embarked[i] == 'C' and Survived[i] == 0:
            CEmLivedCnt += 1
        
        if Embarked[i] == 'Q' and Survived[i] == 1:
            QEmLivedCnt += 1
        elif Embarked[i] == 'Q' and Survived[i] == 0:
            QEmDiedCnt += 1

        if Embarked[i] == 'S' and Survived[i] == 1:
            SEmLivedCnt += 1
        elif Embarked[i] == 'S' and Survived[i] == 0:
            SEmDiedCnt += 1

# Porbability Calculation (Embarked)
R_CEmLived =  CEmLivedCnt/LivedCnt
R_CEmDied = CEmDiedCnt/DeathCnt
R_QEmLived =  QEmLivedCnt/LivedCnt
R_QEmDied = QEmDiedCnt/DeathCnt
R_SEmLived =  SEmLivedCnt/LivedCnt
R_SEmDied = SEmDiedCnt/DeathCnt

def naive_bayes():
    global R_lived,R_death,X,X_bar,Y,Y_bar,Z,Z_bar,K,K_bar
    global J,J_bar
    global classification, outputClassification

    file = pd.read_csv("TitanicRecords_training_set.csv")
    name = file["Name"]
    Pclass = file["Pclass"]
    Sex = file["Sex"]
    SibSp = file["SibSp"]
    Parch = file["Parch"]
    Embarked = file["Embarked"]

    fo = open("Output_File.csv", "w") # output file based on Naive Bayes Classification
    fo.write("Name,\t,Pclass,Sex,SibSp,Parch,Embarked,Survived Classification Made By Naive Bayes\n")
    for i in range(0, len(name)):
        fo.write((str(name[i])+","))
        X,X_bar = check_pclass(Pclass[i],R_Pclass1Lived,R_Pclass1Died,R_Pclass2Lived,R_Pclass2Died,R_Pclass3Lived,R_Pclass3Died)
        fo.write((str(Pclass[i])+","))
        Y,Y_bar = check_sex(Sex[i],R_maleLived,R_maleDied,R_femaleLived,R_femaleDied)
        fo.write((str(Sex[i])+","))
        Z,Z_bar = check_sibsp(SibSp[i],R_YesSibSpLived,R_YesSibSpDied,R_NoSibSpLived,R_NoSibSpDied)
        fo.write((str(SibSp[i])+","))
        K,K_bar = check_parch(Parch[i],R_NoParchLived,R_NoParchDied,R_YesParchLived,R_YesParchDied)
        fo.write((str(Parch[i])+","))
        J,J_bar = check_embarked(Embarked[i],R_CEmLived, R_CEmDied,R_QEmLived,R_QEmDied,R_SEmLived,R_SEmDied)
        fo.write((str(Embarked[i])+","))
        classification = get_classification(X,X_bar,Y,Y_bar,Z,Z_bar,K,K_bar,J,J_bar,R_lived,R_death)
        outputClassification.append(classification)
        fo.write(str(classification)+"\n")

val = 0
naive_bayes()
for i in range(0,len(Survived)):
    if Survived[i] == outputClassification[i]:
        val+=1
val = 100*(val/len(Survived))
print("Naive Bayes Classification Accuracy: "+str(val)+"%")





