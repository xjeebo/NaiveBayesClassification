import pandas as pd
import numpy as np

def check_pclass(Pclass,R_Pclass1Lived,R_Pclass1Died,R_Pclass2Lived,R_Pclass2Died,R_Pclass3Lived,R_Pclass3Died):
    if Pclass == 1:
        X = R_Pclass1Lived
        X_bar = R_Pclass1Died

    elif Pclass == 2:
        X = R_Pclass2Lived
        X_bar = R_Pclass2Died

    elif Pclass == 3:
        X = R_Pclass3Lived
        X_bar = R_Pclass3Died
        
    return X,X_bar   

def check_sex(Sex,R_maleLived,R_maleDied,R_femaleLived,R_femaleDied):
    if Sex == "male":
        Y = R_maleLived
        Y_bar = R_maleDied
    
    elif Sex == "female":
        Y = R_femaleLived
        Y_bar = R_femaleDied

    return Y,Y_bar

def check_sibsp(SibSp,R_YesSibSpLived,R_YesSibSpDied,R_NoSibSpLived,R_NoSibSpDied):
    if SibSp == 0:
        Z = R_NoSibSpLived
        Z_bar = R_NoSibSpDied

    elif SibSp > 0:
        Z = R_YesSibSpLived
        Z_bar = R_YesSibSpDied

    return Z,Z_bar

def check_parch(Parch,R_NoParchLived,R_NoParchDied,R_YesParchLived,R_YesParchDied):
    if Parch == 0:   
        K = R_NoParchLived
        K_bar = R_NoParchLived

    elif Parch > 0:
        K = R_YesParchLived
        K_bar = R_NoParchDied

    return K,K_bar

def check_embarked(Embarked,R_CEmLived, R_CEmDied,R_QEmLived,R_QEmDied,R_SEmLived,R_SEmDied):
    global J,J_bar
    if Embarked == 'C':
        J = R_CEmLived
        J_bar = R_CEmDied
    elif Embarked == 'Q':
        J = R_QEmLived
        J_bar = R_QEmDied
    elif Embarked == 'S':
        J = R_SEmLived
        J_bar = R_SEmDied

    return J,J_bar



def get_classification(X,X_bar,Y,Y_bar,Z,Z_bar,K,K_bar,J,J_bar,R_lived,R_death):
    # probability values for playing and not playing
    R_lived = X*Y*Z*K*J*R_lived
    R_death = X_bar*Y_bar*Z_bar*K_bar*J_bar*R_death
    # adding the probabilities so we can calculate a percentage
    P_X = R_lived+R_death
    # now calculating the probabilities 
    R_lived = R_lived/P_X
    R_death = R_death/P_X
    if R_lived > R_death:
        return 1
    else:
        return 0








       
