# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder

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

# SVM Classifier

feature_cols = ['Year_of_study_encoded', 'CGPA_encoded', 'Major_encoded', 'Languages_encoded', 'Areas_of_Interest_encoded', 'DBMS_encoded', 'Written_communication_skills_encoded', 'Verbal_communication_skills_encoded', 'Core_Java_encoded', 'OOP_encoded']
X = student_dataset[feature_cols] # Features
y = student_dataset["Label_encoded"] # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1:",metrics.f1_score(y_test, y_pred))