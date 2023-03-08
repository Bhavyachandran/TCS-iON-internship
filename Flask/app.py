import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
# making list columns
column = [ 'MarriedID','FromDiversityJobFairID','Termd', 'Position', 'State', 'Zip','Sex', 'MaritalDesc', 'CitizenDesc', 'HispanicLatino', 'RaceDesc',
         'TermReason', 'EmploymentStatus','Department', 'ManagerName', 'RecruitmentSource', 'PerformanceScore', 'EngagementSurvey', 'EmpSatisfaction',
       'SpecialProjectsCount', 'DaysLateLast30','Absences', 'Experience', 'Age']
column_ids = [ 'MarriedID', 'FromDiversityJobFairID', 'Termd', 'Position', 'State', 'Zip','Sex', 'MaritalDesc', 'CitizenDesc', 'HispanicLatino', 'RaceDesc',
         'TermReason', 'EmploymentStatus','Department', 'ManagerName', 'RecruitmentSource', 'PerformanceScore', 'EngagementSurvey', 'EmpSatisfaction',
       'SpecialProjectsCount', 'DaysLateLast30','Absences', 'Experience', 'Age']

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('index.html', len = len(column))
@app.route('/predict',methods =['POST'])
def predict():
     '''
    For rendering results on HTML GUI
    '''
    #features entered by user are collected and passed to model created for prediction
     int_features = [float(x) for x in request.form.values()]
     print('request.form.values()')
     print(np.array(int_features))
     final_features = [np.array(int_features)]

     output =[]

     final_features = {'MarriedID': final_features[0][0], 'FromDiversityJobFairID': final_features[0][1],'Termd': final_features[0][2], 'Position': final_features[0][3], 'State': final_features[0][10], 'Zip': final_features[0][11],
       'Sex': final_features[0][4], 'MaritalDesc': final_features[0][5], 'CitizenDesc': final_features[0][6], 'HispanicLatino': final_features[0][7], 
       'RaceDesc': final_features[0][8],'TermReason': final_features[0][9], 'EmploymentStatus': final_features[0][10],'Department': final_features[0][11], 
       'ManagerName': final_features[0][20], 'RecruitmentSource': final_features[0][21],'PerformanceScore': final_features[0][22], 'EngagementSurvey': final_features[0][23], 
       'EmpSatisfaction': final_features[0][12],'SpecialProjectsCount': final_features[0][13], 'DaysLateLast30': final_features[0][14],'Absences': final_features[0][15], 
       'Experience': final_features[0][16], 'Age': final_features[0][17]}
     final_features = pd.DataFrame(data=final_features, index=[0])
     prediction = model.predict(final_features)

     output = print('Predicted salary :', prediction)

     # the predicted value is returned to the html
     return render_template('prediction.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run()
