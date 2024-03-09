#%%
import pandas as pd
import numpy as np
import os 
import sys

diseases = ['Circulatory', 'Respiratory', 'Digestive', 'Diabetes','Diabetes Uncontrolled', 'Injury', 'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other']
ids = [
    (list(range(390, 460)) + [785]),
    (list(range(460, 520)) + [786]),
    (list(range(520, 580)) + [787]),
    ([250.00, 250.01]),
    np.round(np.arange(250.02, 251.00, 0.01), 2).tolist(),
    (list(range(800, 1000))),
    (list(range(710, 740))),
    (list(range(580, 630)) + [788]),
    (list(range(140, 240))),  
    (list(range(790, 800)) + 
    list(range(240, 250)) + 
    list(range(251, 280)) + 
    list(range(680, 710)) + 
    list(range(780, 785)) + 
    list(range(290, 320)) + 
    list(range(280, 290)) + 
    list(range(320, 360)) + 
    list(range(630, 680)) + 
    list(range(360, 390)) + 
    list(range(740, 760)) +
    list(range(1,140)))]


def replaceNumEmergency(value):
    if value == 0:
        return str(value)
    elif (value > 0) & (value < 5):
        return '<5'
    else:
        return '>=5'

def timeInHosp(value):
    if (value >= 1) & (value <= 4):
        return '1-4'
    elif (value > 4) & (value <= 8):
        return '5-8'
    else:
        return '>8'

def numProcedures(value):
    if value == 0:
        return str(value)
    elif (value >= 1) & (value <= 3):
        return '1-3'
    else:
        return '4-6' 

def inPatiant(value):
    if value == 0:
        return str(value)
    elif (value >= 1) & (value <= 5):
        return '1-5'
    else:
        return '>5'     

def numDiag(value):
    if (value >= 1) & (value <= 4):
        return '1-4'
    elif (value > 4) & (value <= 8):
        return '5-8'
    else:
        return '>=9'
    
def raceChange(value):
    if value == '?':
        return 'Other'
    else:
        return value

def outPatiant(value):
    if value == 0:
        return str(value)
    elif (value >= 1) & (value <= 5):
        return '1-5'
    else:
        return '>5'  

def changeCol(value):
    if value == 'Ch':
        return True
    else:
        return False
    
def diabMed(value):
    if value == 'Yes':
        return True
    else:
        return False
    
def ageFunc(value):
    if value == '[0-10)':
        return 'Children'
    elif value == '[10-20)':
        return 'Adolescents'
    elif value in ['[70-80)','[80-90)','[90-100)']:
        return 'Older_adults'
    else:
        return 'Adults'
    
# Define functions for each column
functions = {
    'number_emergency': replaceNumEmergency,
    'time_in_hospital': timeInHosp,
    'num_procedures': numProcedures,
    'number_inpatient': inPatiant,
    'number_diagnoses': numDiag,
    'race' : raceChange,
    'number_outpatient' : outPatiant,
    'age': ageFunc
}
bool_functions = {
    'change': changeCol,
    'diabetesMed' : diabMed
}