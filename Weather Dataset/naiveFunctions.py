import pandas as pd
import numpy as np
from variables import *
# Based on input calculate the probability of playing or not
def check_outlook(outlook,P_playSunny,P_noplaySunny,P_playOvercast,P_noplayOvercast,P_playRainy,P_noplayRainy):
    if outlook == 'sunny':
        X = P_playSunny
        X_bar = P_noplaySunny
        return X,X_bar
    elif outlook == 'overcast':
        X = P_playOvercast
        X_bar = P_noplayOvercast
        return X,X_bar
    elif outlook == 'rainy':
        X = P_playRainy
        X_bar = P_noplayRainy
        return X,X_bar

def check_temperature(temperature,P_playHot,P_noplayHot,P_playMild,P_noplayMild,P_playCool,P_noplayCool):
    if temperature == 'hot':
        Y = P_playHot
        Y_bar = P_noplayHot
        return Y,Y_bar
    elif temperature == 'mild':
        Y = P_playMild
        Y_bar = P_noplayMild
        return Y,Y_bar
    elif temperature == 'cool':
        Y = P_playCool
        Y_bar = P_noplayCool
        return Y,Y_bar

def check_humidity(humidity,P_playHigh,P_noplayHigh,P_playNormal,P_noplayNormal):
    if humidity == 'high':  
        Z = P_playHigh
        Z_bar = P_noplayHigh
        return Z,Z_bar
    elif humidity == 'normal':
        Z = P_playNormal
        Z_bar = P_noplayNormal
        return Z,Z_bar

def check_windy(windy,P_playWindT,P_noplayWindT,P_playWindF,P_noplayWindF):
    if windy == "True":  
        K = P_playWindT
        K_bar = P_noplayWindT
        return K,K_bar
    elif windy == "False":
        K = P_playWindF
        K_bar = P_noplayWindF
        return K,K_bar

def get_classification(decision,X,X_bar,Y,Y_bar,Z,Z_bar,K,K_bar,P_play,P_noplay):
    # probability values for playing and not playing
    P_play = X*Y*Z*K*P_play
    P_noplay = X_bar*Y_bar*Z_bar*K_bar*P_noplay
    # adding the probabilities so we can calculate a percentage
    P_X = P_play+P_noplay
    # now calculating the probabilities 
    P_play = P_play/P_X
    P_noplay = P_noplay/P_X
    if P_play > P_noplay:
        return "yes"
    else:
        return "no"

