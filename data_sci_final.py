import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

dataset_path = input("Enter the path of the Dataset (excluding the name of the csv file): ")
student_dataset = pd.read_csv(dataset_path + "/DS_DATESET.csv", header = 0)

to_drop = ['First Name', 'Last Name', 'State', 'Zip Code', 'DOB [DD/MM/YYYY]', 'Age', 'Gender', 'Email Address', 'Contact Number', 'Emergency Contact Number', 'University Name', 'Degree', 'Course Type', 'Current Employment Status', 'Certifications/Achievement/ Research papers', 'Link to updated Resume (Google/ One Drive link preferred)', 'link to Linkedin profile']

student_dataset.drop(to_drop, inplace=True, axis = 1)

# Data Visualisation model
with PdfPages('data_visualization.pdf') as pdf:
        # a. The number of students applied to different technologies.
        technology = student_dataset['Areas of interest'].value_counts().keys().tolist()
        no_of_students = student_dataset['Areas of interest'].value_counts().tolist()
        technology_keys = [label.replace(' ', '\n') for label in technology]
        df = pd.DataFrame({'technology classification': no_of_students})

        plt.figure(1)
        colors = tuple(sns.color_palette())
        ax = df['technology classification'].plot(kind='barh', figsize=(15,13), color=colors, fontsize=13)
        ax.set_alpha(0.8)
        ax.set_title("Classification of Students based on Technology", fontsize=18)
        ax.set_xlabel("Number of Students", fontsize=16)
        #ax.set_ylabel("Number of Students", fontsize=16)
        ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700])
        ax.set_yticklabels(technology_keys, rotation = 0)

        # set individual bar lables using above list
        for i in ax.patches:
            # get_width pulls left or right; get_y pushes up or down
            ax.text(i.get_width() + 5, i.get_y() + 0.5, str(round((i.get_width()), 2)), fontsize=13, color='dimgrey')

        # invert for largest on top
        ax.invert_yaxis()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        #b. Interns who knew python and have applied for data science
        ds = student_dataset[student_dataset['Areas of interest'].str.contains('Data Science ')]
        ds_number = len(ds)
        ds_python = len([elem for elem in ds['Programming Language Known other than Java (one major)'] if elem == 'Python'])
        no_of_python_students = [ds_number-ds_python, ds_python]
        plt.figure(2)
        #y_pos = np.arange(len(no_of_python_students))
        bars = ["Don't know python", "Know Python"]

        # Create pie and choose color
        p, tx, autotexts = plt.pie(no_of_python_students, labels=bars, colors=['coral', '#4ec97b'], autopct="", shadow=False, startangle = 0)

        for i, a in enumerate(autotexts):
                percent = (no_of_python_students[i]/ds_number)*100
                a.set_text("{} ({:.2f} %)".format(no_of_python_students[i], percent))

        plt.title("Interns have applied for data science", fontsize=18)
        plt.axis('equal')
        #ax.set_ylabel("Number of students", fontsize=18)
        #ax.set_yticks([0, 100, 200, 300, 400, 500, 600, 700])
        #ax.set_xticklabels(bars, rotation=0, fontsize=11)
        #ax.get_legend().remove()
        # set individual bar lables using above list
        #for i in ax.patches:
            # get_x pulls left or right; get_height pushes up or down
        #    ax.text(i.get_x() + 0.18, i.get_height() + 1, str(i.get_height()), fontsize=13, color='dimgrey')

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
        # Show graphic

        #c. The different ways students learned about this program.

        social_media = student_dataset['How Did You Hear About This Internship?'].value_counts().keys().tolist()
        no_of_students_social_media = student_dataset['How Did You Hear About This Internship?'].value_counts().tolist()
        social_media_keys = [label.replace(' ', '\n') for label in social_media]
        plt.figure(3)
        #colors = tuple(sns.color_palette())
        df = pd.DataFrame({'media classification': no_of_students_social_media})
        ax = df['media classification'].plot(kind='barh', figsize=(13,7), color=colors, fontsize=13)
        ax.set_alpha(0.8)
        ax.set_title("Classification of Students based on Social Media", fontsize=18)
        ax.set_xlabel("Number of Students", fontsize=16)
        #ax.set_ylabel("Number of Students", fontsize=16)
        ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200, 1400])
        ax.set_yticklabels(social_media_keys, rotation = 0)

        # set individual bar lables using above list
        for i in ax.patches:
            # get_width pulls left or right; get_y pushes up or down
            ax.text(i.get_width() + 5, i.get_y() + 0.35, str(round((i.get_width()), 2)), fontsize=13, color='dimgrey')

        # invert for largest on top
        ax.invert_yaxis()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        #d. Students who are in the fourth year and have a CGPA greater than 8.0.

        be = student_dataset[student_dataset['Which-year are you studying in?'].str.contains('Fourth-year')]
        be_8 = len([elem for elem in be['CGPA/ percentage'] if elem >= 8.0])

        no_of_BE_students = [be_8, len(be)-be_8]
        plt.figure(4)
        y_pos = np.arange(len(no_of_BE_students))
        bars = ["Having CGPA >= 8", "Having CGPA < 8"]

        # Create pie and choose color
        p, tx, autotexts = plt.pie(no_of_BE_students, labels=bars, colors=['#8acdee', '#3f42f4'], autopct="", shadow=False, startangle=0)

        for i, a in enumerate(autotexts):
                percent = (no_of_BE_students[i] / len(be)) * 100
                a.set_text("{} ({:.2f} %)".format(no_of_BE_students[i], percent))

        plt.title("Classification of BE Students", fontsize=18)
        plt.axis('equal')

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        #e. Students who applied for Digital Marketing with verbal and written communication score greater than 8.

        dm = student_dataset[student_dataset['Areas of interest'].str.contains('Digital Marketing ')]
        dm_8 = len([elem for elem in zip(dm['Rate your written communication skills [1-10]'], dm['Rate your verbal communication skills [1-10]']) if (elem[0] >= 8 and elem[1] >= 8)])

        plt.figure(5)
        # Create bars and choose color
        dm_students = [dm_8, len(dm)-dm_8]
        #y_pos = np.arange(len(dm_students))
        bars = ["Written and Verbal scores \n Above or Equal to 8", "Written and Verbal scores \n Below 8"]

        # Create pie and choose color
        p, tx, autotexts = plt.pie(dm_students, labels=bars, colors=['#f6cf70', '#f6709f'], autopct="", shadow=False, startangle=0)

        for i, a in enumerate(autotexts):
                percent = (dm_students[i] / len(dm)) * 100
                a.set_text("{} ({:.2f} %)".format(dm_students[i], percent))

        plt.title("Digital Marketing Students classification", fontsize=16)
        plt.axis('equal')

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()


        #f. Year-wise and area of study wise classification of students.

        year_wise_keys = student_dataset['Which-year are you studying in?'].value_counts().keys().tolist()
        #year_wise_students = student_dataset['Which-year are you studying in?'].value_counts().tolist()
        plt.figure(6)
        year_wise = student_dataset['Which-year are you studying in?'].value_counts().plot(kind='bar', figsize=(10,7), color="coral", fontsize=13)
        year_wise.set_alpha(0.8)
        year_wise.set_title("Year wise classification of students", fontsize=18)
        year_wise.set_ylabel("Number of students", fontsize=18)
        year_wise.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000])
        year_wise.set_xticklabels(year_wise_keys, rotation=0, fontsize=11)
        # set individual bar lables using above list
        for i in year_wise.patches:
            # get_x pulls left or right; get_height pushes up or down
            year_wise.text(i.get_x() + 0.175, i.get_height() + 15, str(i.get_height()), fontsize=13, color='dimgrey', )

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        major_wise_keys = student_dataset['Major/Area of Study'].value_counts().keys().tolist()
        #year_wise_students = student_dataset['Which-year are you studying in?'].value_counts().tolist()
        plt.figure(7)
        major_wise = student_dataset['Major/Area of Study'].value_counts().plot(kind='bar', figsize=(10,7),color="dodgerblue", fontsize=13);
        major_wise.set_alpha(0.8)
        major_wise.set_title("Area of study wise classification of students", fontsize=18)
        major_wise.set_ylabel("Number of students", fontsize=18)
        major_wise.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000])
        major_wise.set_xticklabels(major_wise_keys, rotation=0, fontsize=11)
        # set individual bar lables using above list
        for i in major_wise.patches:
            # get_x pulls left or right; get_height pushes up or down
            major_wise.text(i.get_x() + 0.175, i.get_height() + 20, str(i.get_height()), fontsize=13, color='dimgrey', )

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        #g. City and college wise classification of students.

        city_wise_keys = student_dataset['City'].value_counts().keys().tolist()
        #year_wise_students = student_dataset['Which-year are you studying in?'].value_counts().tolist()
        plt.figure(8)
        city_wise = student_dataset['City'].value_counts().plot(kind='bar', figsize=(10,7),color='#9467bd', fontsize=13);
        city_wise.set_alpha(0.8)
        city_wise.set_title("City wise classification of students", fontsize=18)
        city_wise.set_ylabel("Number of students", fontsize=18)
        city_wise.set_yticks([0, 500, 1000, 1500, 2000])
        city_wise.set_xticklabels(city_wise_keys, rotation=0, fontsize=11)
        # set individual bar lables using above list
        for i in city_wise.patches:
            # get_x pulls left or right; get_height pushes up or down
            city_wise.text(i.get_x() + 0.08, i.get_height() + 5, str(i.get_height()), fontsize=13, color='dimgrey')

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        qualitative_colors = sns.color_palette("Set3")

        college_wise_keys_org = student_dataset['College name'].value_counts().keys().tolist()
        college_wise = student_dataset['College name'].value_counts().tolist()
        college_wise_keys = [label.replace(' ', '\n') for label in college_wise_keys_org]
        df = pd.DataFrame({'college wise':college_wise})
        plt.figure(9)
        ax = df['college wise'].plot(kind='bar', figsize=(30, 13), color=qualitative_colors,fontsize=13);
        ax.set_alpha(0.8)
        ax.set_title("College wise classification of students", fontsize=18)
        ax.set_ylabel("Number of students", fontsize=18)
        ax.set_yticks([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000])
        ax.set_xticklabels(college_wise_keys, rotation=0, fontsize=11)
        # set individual bar lables using above list
        for i in ax.patches:
                # get_x pulls left or right; get_height pushes up or down
                ax.text(i.get_x(), i.get_height() + 5, str(i.get_height()), fontsize=13, color='dimgrey')

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        #Plot the relationship between the CGPA and the target variable.

        ineligible = student_dataset[student_dataset['Label'].str.contains('ineligible')]
        #ineligible_56 = [elem for elem in ineligible['CGPA/ percentage'] if (5 <= elem < 6)]
        #ineligible_67 = [elem for elem in ineligible['CGPA/ percentage'] if (6 <= elem < 7)]
        ineligible_78 = len([elem for elem in ineligible['CGPA/ percentage'] if (7 <= elem < 8)])
        ineligible_89 = len([elem for elem in ineligible['CGPA/ percentage'] if (8 <= elem < 9)])
        ineligible_910 = len([elem for elem in ineligible['CGPA/ percentage'] if (9 <= elem <= 10)])
        #cgpa = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]


        #eligible_56 = [elem for elem in eligible['CGPA/ percentage'] if (5 <= elem < 6)]
        #eligible_67 = [elem for elem in eligible['CGPA/ percentage'] if (6 <= elem < 7)]
        student_dataset_78 = len([elem for elem in student_dataset['CGPA/ percentage'] if (7 <= elem < 8)])
        student_dataset_89 = len([elem for elem in student_dataset['CGPA/ percentage'] if (8 <= elem < 9)])
        student_dataset_910 = len([elem for elem in student_dataset['CGPA/ percentage'] if (9 <= elem <= 10)])

        plt.figure(10)
        df = pd.DataFrame({'eligible': [student_dataset_78 - ineligible_78, student_dataset_89 - ineligible_89, student_dataset_910 - ineligible_910], 'ineligible': [ineligible_78, ineligible_89, ineligible_910]})
        ax = df.plot(kind='bar', figsize=(10,7), color=['#00a367', 'coral'], fontsize=13)
        ax.set_alpha(0.8)
        ax.set_title("CGPA Classification", fontsize=18)
        ax.set_ylabel("Number of Students", fontsize=16)
        ax.set_xlabel("Range of CGPA", fontsize=16)
        ax.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500])
        ax.set_xticklabels(["7 - 8", "8 - 9", "9 - 10"], rotation=0, fontsize=11)

        # set individual bar lables using above list
        for i in ax.patches:
            # get_x pulls left or right; get_height pushes up or down
            ax.text(i.get_x() + 0.05, i.get_height() + 25, str(round((i.get_height()), 2)), fontsize=11, color='dimgrey', rotation=0)

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        #i. Plot the relationship between the Area of Interest and the target variable.

        #technology = student_dataset['Areas of interest'].value_counts().keys().tolist() which is already done
        #no_of_students = student_dataset['Areas of interest'].value_counts().tolist()
        no_of_ineligible_students = ineligible['Areas of interest'].value_counts().tolist()
        no_of_eligible_students = [int(i-j) for i,j in zip(no_of_students, no_of_ineligible_students)]

        plt.figure(11)

        df = pd.DataFrame({'eligible': no_of_eligible_students, 'ineligible': no_of_ineligible_students})
        ax = df[['eligible', 'ineligible']].plot(kind='barh', figsize=(17.5,7), color=['dodgerblue', '#ffa700'], fontsize=13);
        ax.set_alpha(0.8)
        ax.set_title("Eligibilty for the Applied Technology", fontsize=18)
        ax.set_xlabel("Number of Students", fontsize=16)
        #ax.set_ylabel("Number of Students", fontsize=16)
        ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
        ax.set_yticklabels(technology)

        # set individual bar lables using above list
        for i in ax.patches:
            # get_width pulls left or right; get_y pushes up or down
            ax.text(i.get_width()+ 5, i.get_y()+.18, str(round((i.get_width()), 2)), fontsize=11, color='dimgrey')

        # invert for largest on top
        ax.invert_yaxis()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        #j. Plot the relationship between the year of study, major, and the target variable.

        no_of_students = student_dataset['Which-year are you studying in?'].value_counts().tolist()
        no_of_ineligible_students = ineligible['Which-year are you studying in?'].value_counts().tolist()
        no_of_eligible_students = [int(i-j) for i,j in zip(no_of_students, no_of_ineligible_students)]

        plt.figure(12)

        df = pd.DataFrame({'eligible': no_of_eligible_students, 'ineligible': no_of_ineligible_students})
        ax = df.plot(kind='bar', figsize=(10,7), color=['#84eef6', '#F6546A'], fontsize=13);
        ax.set_alpha(0.8)
        ax.set_title("Year wise eligibility Classification", fontsize=18)
        ax.set_ylabel("Number of Students", fontsize=16)
        ax.set_xlabel("Year of Students", fontsize=16)
        ax.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000])
        ax.set_xticklabels(year_wise_keys, rotation=0, fontsize=11)

        # set individual bar lables using above list
        for i in ax.patches:
            # get_x pulls left or right; get_height pushes up or down
            ax.text(i.get_x() + 0.05, i.get_height() + 25, str(round((i.get_height()), 2)), fontsize=11, color='dimgrey', rotation=0)

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        no_of_students = student_dataset['Major/Area of Study'].value_counts().tolist()
        no_of_ineligible_students = ineligible['Major/Area of Study'].value_counts().tolist()
        no_of_eligible_students = [int(i-j) for i,j in zip(no_of_students, no_of_ineligible_students)]

        plt.figure(13)

        df = pd.DataFrame({'eligible': no_of_eligible_students, 'ineligible': no_of_ineligible_students})
        ax = df.plot(kind='bar', figsize=(10, 7), color=['#aa84f6', '#f4be58'], fontsize=13)
        ax.set_alpha(0.8)
        ax.set_title("Area of Study/Major wise eligibility Classification", fontsize=18)
        ax.set_ylabel("Number of Students", fontsize=16)
        ax.set_xlabel("Area of Study", fontsize=16)
        ax.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
        ax.set_xticklabels(major_wise_keys, rotation=0, fontsize=11)

        # set individual bar lables using above list
        for i in ax.patches:
            # get_x pulls left or right; get_height pushes up or down
            ax.text(i.get_x()+.05, i.get_height() + 50, str(round((i.get_height()), 2)), fontsize=11, color='dimgrey', rotation=0)

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

# Data Classification model

student_dataset['Label_encoded'] = student_dataset['Label'].map( {'eligible':1, 'ineligible':0} )
eligiblity = student_dataset['Label_encoded'].tolist()
student_dataset['OOP_encoded'] = student_dataset['Have you studied OOP Concepts'].map({'Yes':1, 'No':0})


#technology = student_dataset['Areas of interest'].value_counts().keys().tolist()
#no_of_students = student_dataset['Areas of interest'].value_counts().tolist()

student_dataset['Year_of_study_encoded'] = LabelEncoder().fit_transform(student_dataset['Which-year are you studying in?'])
student_dataset['CGPA_encoded'] = LabelEncoder().fit_transform(student_dataset['CGPA/ percentage'])
student_dataset['Major_encoded'] = LabelEncoder().fit_transform(student_dataset['Major/Area of Study'])
student_dataset['Languages_encoded'] = LabelEncoder().fit_transform(student_dataset['Programming Language Known other than Java (one major)'])
student_dataset['Areas_of_Interest_encoded'] = LabelEncoder().fit_transform(student_dataset['Areas of interest'])
student_dataset['Written_communication_skills_encoded'] = LabelEncoder().fit_transform(student_dataset['Rate your written communication skills [1-10]'])
student_dataset['Verbal_communication_skills_encoded'] = LabelEncoder().fit_transform(student_dataset['Rate your verbal communication skills [1-10]'])
student_dataset['DBMS_encoded'] = LabelEncoder().fit_transform(student_dataset['Have you worked on MySQL or Oracle database'])
student_dataset['Core_Java_encoded'] = LabelEncoder().fit_transform(student_dataset['Have you worked core Java'])

# Random Forest Classifier

feature_cols = ['Year_of_study_encoded', 'CGPA_encoded', 'Major_encoded', 'Languages_encoded', 'Areas_of_Interest_encoded', 'DBMS_encoded', 'Written_communication_skills_encoded', 'Verbal_communication_skills_encoded', 'Core_Java_encoded', 'OOP_encoded']
X = student_dataset[feature_cols] # Features
y = student_dataset["Label_encoded"] # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=1)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1:",metrics.f1_score(y_test, y_pred))