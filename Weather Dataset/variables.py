# Amount of Play and No Play days
PlayCnt = 0
NoPlayCnt = 0
# probability to play or not in total
P_play = 0
P_noplay = 0

# probability to play or not in total based on outlook
P_playSunny = 0
P_noplaySunny = 0
P_playOvercast = 0
P_noplayOvercast = 0
P_playRainy = 0
P_noplayRainy = 0

# probability to play or not in total based on temperature
P_playHot = 0
P_noplayHot = 0
P_playMild = 0
P_noplayMild = 0
P_playCool = 0
P_noplayCool = 0

# probability to play or not in total based on humidity
P_playHigh = 0
P_noplayHigh = 0
P_playNormal = 0
P_noplayNormal = 0

# probability to play or not in total based on wind
P_playWindT = 0
P_noplayWindT = 0
P_playWindF = 0
P_noplayWindF = 0

# Counts for each variable factor
playSunnyCnt = 0
noplaySunnyCnt = 0
playOvercastCnt = 0
noplayOvercastCnt = 0
playRainCnt = 0
noplayRainCnt = 0
playHotCnt = 0
noplayHotCnt = 0
playMildCnt = 0
noplayMildCnt = 0
playCoolCnt = 0
noplayCoolCnt = 0
playHighCnt = 0
noplayHighCnt = 0
playNormalCnt = 0
noplayNormalCnt = 0
playWindTCnt = 0
noplayWindTCnt = 0
playWindFCnt = 0
noplayWindFCnt = 0

# product of the sum of each factor for the total days 
P_X = 0

# variables for getting the percentage values for whichever input is collected
X = 0
X_bar = 0
Y = 0
Y_bar = 0
Z = 0
Z_bar = 0
K = 0
K_bar = 0

# decisions which are wrote to the output file based on the naive bayes classification
decision = ""
