import pandas as pd
import numpy as np
from variables import *
from naiveFunctions import check_humidity,check_outlook,check_temperature,check_windy,get_classification

file = pd.read_csv("Training_Dataset.csv")
outlook = file["outlook"]
temperaturelevel = file["temperature"]
humiditylevel = file["humidity"]
windyornot = file["windy"]
playornot = file["play"]

# count of plays/noplay of when they played and didnt
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
for i in playornot:
    if i == "yes":
        PlayCnt += 1
    else:
        NoPlayCnt += 1  
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
P_play = PlayCnt/len(playornot)
P_noplay = NoPlayCnt/len(playornot)

# probabiility played or not based on outlook
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
for i in range(0, len(outlook)):
    if outlook[i] == "sunny" and playornot[i] == "yes":
        playSunnyCnt += 1
    elif outlook[i] == "sunny" and playornot[i] == "no":
        noplaySunnyCnt += 1

    if outlook[i] == "overcast" and playornot[i] == "yes":
        playOvercastCnt += 1
    elif outlook[i] == "overcast" and playornot[i] == "no":
        noplayOvercastCnt += 1

    if outlook[i] == "rainy" and playornot[i] == "yes":
        playRainCnt += 1
    elif outlook[i] == "rainy" and playornot[i] == "no":
        noplayRainCnt += 1

# Probability calculations (outlook)
P_playSunny = playSunnyCnt/PlayCnt
P_noplaySunny = noplaySunnyCnt/NoPlayCnt
P_playOvercast = playOvercastCnt/PlayCnt
P_noplayOvercast = noplayOvercastCnt/NoPlayCnt
P_playRainy = playRainCnt/PlayCnt
P_noplayRainy = noplayRainCnt/NoPlayCnt

# probabiility played or not based on temperature
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
for i in range(0, len(outlook)):
    if temperaturelevel[i] == "hot" and playornot[i] == "yes":
        playHotCnt += 1
    elif temperaturelevel[i] == "hot" and playornot[i] == "no":
        noplayHotCnt += 1

    if temperaturelevel[i] == "mild" and playornot[i] == "yes":
        playMildCnt += 1
    elif temperaturelevel[i] == "mild" and playornot[i] == "no":
        noplayMildCnt += 1

    if temperaturelevel[i] == "cool" and playornot[i] == "yes":
        playCoolCnt += 1
    elif temperaturelevel[i] == "cool" and playornot[i] == "no":
        noplayCoolCnt += 1
        
# Probability calculations (temperature)
P_playHot = playHotCnt/PlayCnt
P_noplayHot = noplayHotCnt/NoPlayCnt
P_playMild = playMildCnt/PlayCnt
P_noplayMild = noplayMildCnt/NoPlayCnt
P_playCool = playCoolCnt/PlayCnt
P_noplayCool = noplayCoolCnt/NoPlayCnt


# probabiility played or not based on humidity
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
for i in range(0, len(humiditylevel)):
    if humiditylevel[i] == "high" and playornot[i] == "yes":
        playHighCnt += 1
    elif humiditylevel[i] == "high" and playornot[i] == "no":
        noplayHighCnt += 1

    if humiditylevel[i] == "normal" and playornot[i] == "yes":
        playNormalCnt += 1
    elif humiditylevel[i] == "normal" and playornot[i] == "no":
        noplayNormalCnt += 1

# Probability calculations (humidity)
P_playHigh = playHighCnt/PlayCnt
P_noplayHigh = noplayHighCnt/NoPlayCnt
P_playNormal = playNormalCnt/PlayCnt
P_noplayNormal = noplayNormalCnt/NoPlayCnt


# probabiility played or not based on wind
#-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
for i in range(0, len(windyornot)):
    if windyornot[i] == True and playornot[i] == "yes":
        playWindTCnt += 1
    elif windyornot[i] == True and playornot[i] == "no":
        noplayWindTCnt += 1

    if windyornot[i] == False and playornot[i] == "yes":
        playWindFCnt += 1
    elif windyornot[i] == False and playornot[i] == "no":
        noplayWindFCnt += 1

# Probability calculations (wind)
P_playWindT = playWindTCnt/PlayCnt
P_noplayWindT = noplayWindTCnt/NoPlayCnt
P_playWindF = playWindFCnt/PlayCnt
P_noplayWindF = noplayWindFCnt/NoPlayCnt


#-~--~-~--~-~--~-~--~-~--~-~--~-~--~-~--~-~--~-~--~-~--~-~--~-~--~-~--~
# we will not try to predict the "play" column on the training set 
# it will output a .csv file so we can make comparisons
def naive_bayes():
    global P_play,P_noplay,X,X_bar,Y,Y_bar,Z,Z_bar,K,K_bar
    global P_playSunny,P_noplaySunny, decision, playornot 
    accuracy = 0
    # Opening the dataset without the "play" column
    file = pd.read_csv("Test_Dataset.csv")
    outlook = file["outlook"]
    temperaturelevel = file["temperature"]
    humiditylevel = file["humidity"]
    windyornot = file["windy"]

    fo = open("Output_File.csv", "w") # output file based on Naive Bayes Classification
    fo.write("outlook,temperature,humidity,windy,Decision Made By Naive Bayes\n")
    for i in range(0,len(outlook)):
        X,X_bar = check_outlook(outlook[i],P_playSunny,P_noplaySunny,P_playOvercast,P_noplayOvercast,P_playRainy,P_noplayRainy)
        fo.write(str(outlook[i])+",")
        Y,Y_bar = check_temperature(temperaturelevel[i],P_playHot,P_noplayHot,P_playMild,P_noplayMild,P_playCool,P_noplayCool)
        fo.write(str(temperaturelevel[i])+",")        
        Z,Z_bar = check_humidity(str(humiditylevel[i]),P_playHigh,P_noplayHigh,P_playNormal,P_noplayNormal)
        fo.write(str(humiditylevel[i])+",")        
        K,K_bar = check_windy(str(windyornot[i]),P_playWindT,P_noplayWindT,P_playWindF,P_noplayWindF)
        fo.write(str(windyornot[i])+",")
        decision = get_classification(decision,X,X_bar,Y,Y_bar,Z,Z_bar,K,K_bar,P_play,P_noplay)
        fo.write(str(decision)+"\n")

        if decision == playornot[i]:
            accuracy+=1

    print("Naive Bayes Classification Accuracy: "+str(100*(accuracy/len(playornot)))+'%')
#-~--~-~--~-~--~-~--~-~--~-~--~-~--~-~--~-~--~-~--~-~--~-~--~-~--~-~--~

# main
naive_bayes()



