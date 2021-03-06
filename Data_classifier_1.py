import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns

student_dataset = pd.read_csv('DS_DATESET.csv', header = 0)

to_drop = ['First Name', 'Last Name', 'State', 'Zip Code', 'DOB [DD/MM/YYYY]', 'Age', 'Gender', 'Email Address', 'Contact Number', 'Emergency Contact Number', 'University Name', 'Degree', 'Course Type', 'Current Employment Status', 'Certifications/Achievement/ Research papers', 'Link to updated Resume (Google/ One Drive link preferred)', 'link to Linkedin profile']

student_dataset.drop(to_drop, inplace=True, axis = 1)

student_dataset['Label_encoded'] = student_dataset['Label'].map( {'eligible':1, 'ineligible':0} )
eligiblity = student_dataset['Label_encoded'].tolist()
student_dataset['OOP_encoded'] = student_dataset['Have you studied OOP Concepts'].map({'Yes':1, 'No':0})


technology = student_dataset['Areas of interest'].value_counts().keys().tolist()
no_of_students = student_dataset['Areas of interest'].value_counts().tolist()

student_dataset['Year_of_study_encoded'] = LabelEncoder().fit_transform(student_dataset['Which-year are you studying in?'])
student_dataset['CGPA_encoded'] = LabelEncoder().fit_transform(student_dataset['CGPA/ percentage'])
student_dataset['Major_encoded'] = LabelEncoder().fit_transform(student_dataset['Major/Area of Study'])
student_dataset['Languages_encoded'] = LabelEncoder().fit_transform(student_dataset['Programming Language Known other than Java (one major)'])

student_dataset['Areas_of_Interest_encoded'] = LabelEncoder().fit_transform(student_dataset['Areas of interest'])
student_dataset['Written_communication_skills_encoded'] = LabelEncoder().fit_transform(student_dataset['Rate your written communication skills [1-10]'])
student_dataset['Verbal_communication_skills_encoded'] = LabelEncoder().fit_transform(student_dataset['Rate your verbal communication skills [1-10]'])
student_dataset['DBMS_encoded'] = LabelEncoder().fit_transform(student_dataset['Have you worked on MySQL or Oracle database'])
student_dataset['Core_Java_encoded'] = LabelEncoder().fit_transform(student_dataset['Have you worked core Java'])



'''plt.xlabel('Technology')
plt.ylabel('Number of students')
plt.scatter(technology, no_of_students).axes.get_xaxis().set_visible(False)
for i_x, i_y in zip(technology, no_of_students):
    plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y))
plt.show()'''

#Logistic Regression

feature_cols = ['Year_of_study_encoded', 'CGPA_encoded', 'Major_encoded', 'Languages_encoded', 'Areas_of_Interest_encoded', 'DBMS_encoded', 'Written_communication_skills_encoded', 'Verbal_communication_skills_encoded', 'Core_Java_encoded', 'OOP_encoded']
X = student_dataset[feature_cols] # Features
y = student_dataset["Label_encoded"] # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# instantiate the model (using the default parameters)
logreg = LogisticRegression(max_iter=1000)
# fit the model with data
logreg.fit(X_train,y_train)
#
y_pred=logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print('F1:', metrics.f1_score(y_test, y_pred))
#