import os

here = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(here + '/ML_Project/pyScripts/BarModels/logs/logsRF_Main_Run_FullScript'):
    os.makedirs(here + '/ML_Project/pyScripts/BarModels/logs/logsRF_Main_Run_FullScript')
    print('Directory created')
