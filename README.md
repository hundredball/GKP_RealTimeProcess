# GKP_RealTimeProcess
This is used for classifying received data in real time. (Demo)


## run user.py
1. sudo -s  #to access USB socket, changing into adminastrator.
2. source activate py3  #open your anaconda environment.
3. python user.py --add streamer_lsl --num 'subjectID'-'session'.

## Select model
1. Put param_(date).txt and EEGNet_ReLU_(date).pt in this directory, which are trained by TrainModel.
2. In openbci/cyton.py, change the values of 'm' and 'param' to select a specific model. 
