import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns

student_dataset = pd.read_csv('DS_DATESET.csv', header = 0)

to_drop = ['First Name', 'Last Name', 'State', 'Zip Code', 'DOB [DD/MM/YYYY]', 'Age', 'Gender', 'Email Address', 'Contact Number', 'Emergency Contact Number', 'University Name', 'Degree', 'Course Type', 'Current Employment Status', 'Certifications/Achievement/ Research papers', 'Link to updated Resume (Google/ One Drive link preferred)', 'link to Linkedin profile']

student_dataset.drop(to_drop, inplace=True, axis = 1)

#a. The number of students applied to different technologies.

technology = student_dataset['Areas of interest'].value_counts().keys().tolist()
no_of_students = student_dataset['Areas of interest'].value_counts().tolist()

plt.figure(1)
plt.ylabel('Number of students')
plt.title('Technology')
plt.scatter(technology, no_of_students).axes.get_xaxis().set_visible(False)
for i_x, i_y in zip(technology, no_of_students):
    plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y))
plt.show()

#b. Interns who knew python and have applied for data science
ds = student_dataset[student_dataset['Areas of interest'].str.contains('Data Science ')]
ds_number = len(ds)
ds_python = len([elem for elem in ds['Programming Language Known other than Java (one major)'] if elem == 'Python'])
no_of_python_students = [ds_number-ds_python, ds_python]
plt.figure(2)
y_pos = np.arange(len(no_of_python_students))
bars = ["Don't know python", "Know Python"]

# Create bars and choose color
plt.bar(y_pos, no_of_python_students , color=(0.5, 0.1, 0.5, 0.6))

# Add title and axis names
plt.title('Interns who applied for data Science')
plt.ylabel('Number of Students')

# Limits for the Y axis
plt.ylim(0, 650)

# Create names
plt.xticks(y_pos, bars)

# Show graphic
plt.show()

#c. The different ways students learned about this program.

social_media = student_dataset['How Did You Hear About This Internship?'].value_counts().keys().tolist()
no_of_students_social_media = student_dataset['How Did You Hear About This Internship?'].value_counts().tolist()

plt.figure(3)
plt.ylabel('Number of students')
plt.title('Social Media')
plt.scatter(social_media, no_of_students_social_media).axes.get_xaxis().set_visible(False)
for i_x, i_y in zip(social_media, no_of_students_social_media):
    plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y))
plt.show()

#d. Students who are in the fourth year and have a CGPA greater than 8.0.

be = student_dataset[student_dataset['Which-year are you studying in?'].str.contains('Fourth-year')]
be_8 = len([elem for elem in be['CGPA/ percentage'] if elem >= 8.0])

plt.figure(4)
# Create bars and choose color
plt.barh(1, be_8, color='g')
fig = plt.gcf()
fig.set_size_inches(12,0.3)
plt.yticks([])
plt.title('Interns who are in fourth year and have got CGPA above 8')
plt.xlabel('Number of Students are: ' + str(be_8))

plt.show()

#e. Students who applied for Digital Marketing with verbal and written communication score greater than 8.

dm = student_dataset[student_dataset['Areas of interest'].str.contains('Digital Marketing ')]
dm_8 = len([elem for elem in zip(dm['Rate your written communication skills [1-10]'], dm['Rate your verbal communication skills [1-10]']) if (elem[0] >= 8 and elem[1] >= 8)])

plt.figure(5)
# Create bars and choose color
plt.barh(1, dm_8, color='red')
fig = plt.gcf()
fig.set_size_inches(12,0.3)
plt.yticks([])
plt.title('Interns who applied for digital marketing and have got above 8 in verbal and written communication')
plt.xlabel('Number of Students are: ' + str(dm_8))
plt.show()

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
    year_wise.text(i.get_x() + 0.05, i.get_height() + 1, str(i.get_height()), fontsize=15, color='dimgrey', )

plt.show()

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
    major_wise.text(i.get_x() + 0.15, i.get_height() + 1, str(i.get_height()), fontsize=15, color='dimgrey', )

plt.show()

#g. City and college wise classification of students.

city_wise_keys = student_dataset['City'].value_counts().keys().tolist()
#year_wise_students = student_dataset['Which-year are you studying in?'].value_counts().tolist()
plt.figure(8)
city_wise = student_dataset['City'].value_counts().plot(kind='bar', figsize=(10,7),color='#9467bd', fontsize=13);
city_wise.set_alpha(0.8)
city_wise.set_title("City wise classification of students", fontsize=18)
city_wise.set_ylabel("Number of students", fontsize=18)
city_wise.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000])
city_wise.set_xticklabels(city_wise_keys, rotation=0, fontsize=11)
# set individual bar lables using above list
for i in city_wise.patches:
    # get_x pulls left or right; get_height pushes up or down
    city_wise.text(i.get_x() + 0.05, i.get_height() + 1, str(i.get_height()), fontsize=15, color='dimgrey')

plt.show()

college_wise_keys = student_dataset['College name'].value_counts().keys().tolist()
#year_wise_students = student_dataset['Which-year are you studying in?'].value_counts().tolist()
plt.figure(9)
college_wise = student_dataset['College name'].value_counts().plot(kind='bar', figsize=(10,7), color='#ffa700', fontsize=13)
college_wise.set_alpha(0.8)
college_wise.set_title("City wise classification of students", fontsize=18)
college_wise.set_ylabel("Number of students", fontsize=18)
college_wise.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000])
college_wise.set_xticklabels(college_wise_keys, rotation=0, fontsize=11)

# set individual bar lables using above list
for i in college_wise.patches:
    # get_x pulls left or right; get_height pushes up or down
    college_wise.text(i.get_x() + 0.05, i.get_height() + 1, str(i.get_height()), fontsize=15, color='dimgrey', )

plt.show()

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
df = pd.DataFrame({'eligible_cgpa': [student_dataset_78 - ineligible_78, student_dataset_89 - ineligible_89, student_dataset_910 - ineligible_910], 'ineligible_cgpa': [ineligible_78, ineligible_89, ineligible_910]})
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
    ax.text(i.get_x()+.01, i.get_height()+100, str(round((i.get_height()), 2)), fontsize=11, color='dimgrey', rotation=0)

plt.show()

#i. Plot the relationship between the Area of Interest and the target variable.

#technology = student_dataset['Areas of interest'].value_counts().keys().tolist() which is already done
#no_of_students = student_dataset['Areas of interest'].value_counts().tolist()
no_of_ineligible_students = ineligible['Areas of interest'].value_counts().tolist()
no_of_eligible_students = [int(i-j) for i,j in zip(no_of_students, no_of_ineligible_students)]
'''no_of_eligible_students = []
for i in range(len(no_of_eligible_students)):
    a = no_of_students - no_of_ineligible_students
    no_of_eligible_students.append(a)'''

plt.figure(11)

df = pd.DataFrame({'eligible_tech': no_of_eligible_students, 'ineligible_tech': no_of_ineligible_students})
ax = df[['eligible_tech', 'ineligible_tech']].plot(kind='barh', figsize=(10,7), color=['dodgerblue', '#ffa700'], fontsize=13);
ax.set_alpha(0.8)
ax.set_title("Eligibilty for the Applied Technology", fontsize=18)
ax.set_xlabel("Number of Students", fontsize=16)
#ax.set_ylabel("Number of Students", fontsize=16)
ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600])
ax.set_yticklabels(technology)

# set individual bar lables using above list
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+ 5, i.get_y()+.18, str(round((i.get_width()), 2)), fontsize=11, color='dimgrey')

# invert for largest on top
ax.invert_yaxis()
plt.show()

#j. Plot the relationship between the year of study, major, and the target variable.

no_of_students = student_dataset['Which-year are you studying in?'].value_counts().tolist()
no_of_ineligible_students = ineligible['Which-year are you studying in?'].value_counts().tolist()
no_of_eligible_students = [int(i-j) for i,j in zip(no_of_students, no_of_ineligible_students)]

plt.figure(12)

df = pd.DataFrame({'eligible_tech': no_of_eligible_students, 'ineligible_tech': no_of_ineligible_students})
ax = df.plot(kind='bar', figsize=(10,7), color=['#84eef6', '#F6546A'], fontsize=13);
ax.set_alpha(0.8)
ax.set_title("Year wise eligibility Classification", fontsize=18)
ax.set_ylabel("Number of Students", fontsize=16)
ax.set_xlabel("Year of Students", fontsize=16)
ax.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500])
ax.set_xticklabels(year_wise_keys, rotation=0, fontsize=11)

# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+.01, i.get_height()+100, str(round((i.get_height()), 2)), fontsize=11, color='dimgrey', rotation=0)

plt.show()

no_of_students = student_dataset['Major/Area of Study'].value_counts().tolist()
no_of_ineligible_students = ineligible['Major/Area of Study'].value_counts().tolist()
no_of_eligible_students = [int(i-j) for i,j in zip(no_of_students, no_of_ineligible_students)]

plt.figure(13)

df = pd.DataFrame({'eligible_major': no_of_eligible_students, 'ineligible_major': no_of_ineligible_students})
ax = df.plot(kind='bar', figsize=(15, 15), color=['#aa84f6', '#f4be58'], fontsize=13)
ax.set_alpha(0.8)
ax.set_title("Year wise eligibility Classification", fontsize=18)
ax.set_ylabel("Number of Students", fontsize=16)
ax.set_xlabel("Area of Study", fontsize=16)
ax.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
ax.set_xticklabels(major_wise_keys, rotation=0, fontsize=11)

# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+.01, i.get_height()+100, str(round((i.get_height()), 2)), fontsize=11, color='dimgrey', rotation=0)

plt.show()
