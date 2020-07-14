import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt
import seaborn as sns

student_dataset = pd.read_csv('DS_DATESET.csv', header = 0)

to_drop = ['First Name', 'Last Name', 'State', 'Zip Code', 'DOB [DD/MM/YYYY]', 'Age', 'Gender', 'Email Address', 'Contact Number', 'Emergency Contact Number', 'University Name', 'Degree', 'Course Type', 'Current Employment Status', 'Have you worked on MySQL or Oracle database', 'Have you studied OOP Concepts', 'Certifications/Achievement/ Research papers', 'Link to updated Resume (Google/ One Drive link preferred)', 'link to Linkedin profile']

student_dataset.drop(to_drop, inplace=True, axis = 1)

#The number of students applied to different technologies
Technology = []
for i in student_dataset['Areas of interest']:
    if i not in Technology:
        Technology.append(i)

no_of_students_in_tech = []

for j in Technology:
    c = 0
    for i in student_dataset['Areas of interest']:
        if j == i:
            c += 1
    no_of_students_in_tech.append(c)

for i in no_of_students_in_tech:
    c = c + i

total_no_of_students = c
plt.xlabel('Technology')
plt.ylabel('Number of students')
plt.scatter(Technology, no_of_students_in_tech).axes.get_xaxis().set_visible(False)
for i_x, i_y in zip(Technology, no_of_students_in_tech):
    plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y))
plt.show()

#The number of students applied for Data Science who knew ‘’Python” and who didn’t.
c1 = 0
c2 = 0
DS = zip(student_dataset['Areas of interest'], student_dataset['Programming Language Known other than Java (one major)'])

for i, j in DS:
    if i == 'Data Science ':
        c1 += 1
        if j == 'Python':
            c2 += 1

print("Number of students who have applied for DS and knew python are", c2)
print("Number of students who have applied for DS and did not know python are", c1 - c2)

#The different ways students learned about this program.
Social_Media = []
for i in student_dataset['How Did You Hear About This Internship?']:
    if i not in Social_Media:
        Social_Media.append(i)

student_social_media = []

for j in Social_Media:
    c = 0
    for i in student_dataset['How Did You Hear About This Internship?']:
        if j == i:
            c += 1
    student_social_media.append(c)

for i in zip(Social_Media, student_social_media):
    print(i)

#Students who are in the fourth year and have a CGPA greater than 8.0.

year_and_cgpa = zip(student_dataset['Which-year are you studying in?'], student_dataset['CGPA/ percentage'])
c = 0
for i,j in year_and_cgpa:
    if i == 'Fourth-year' and j > 8.0:
        c += 1
print('Students who are in the fourth year and have a CGPA greater than 8.0 ' ,c)

#Students who applied for Digital Marketing with verbal and written communication score greater than 8.

interest_written_verbal = zip(student_dataset['Areas of interest'], student_dataset['Rate your written communication skills [1-10]'], student_dataset['Rate your verbal communication skills [1-10]'])
c = 0
for i,j,k in interest_written_verbal:
    if i == 'Digital Marketing ' and j > 8 and k > 8:
        c += 1

print('Students who applied for Digital Marketing with verbal and written communication score greater than 8 ', c)

#Year-wise and area of study wise classification of students.

year_of_students = []
for i in student_dataset['Which-year are you studying in?']:
    if i not in year_of_students:
        year_of_students.append(i)

area_of_study = []
for i in student_dataset['Major/Area of Study']:
    if i not in area_of_study:
        area_of_study.append(i)

year = student_dataset['Which-year are you studying in?']
major = student_dataset['Major/Area of Study']
year_and_number_of_students = []
major_and_number_of_students= []

for i in year_of_students:
    c = 0
    for j in year:
        if i == j:
            c += 1
    year_and_number_of_students.append(c)

for a in year_and_number_of_students:
    print(a)

plt.xlabel('Year of the students')
plt.ylabel('Number of students')
plt.scatter(year_of_students, year_and_number_of_students)
for i_x, i_y in zip(year_of_students, year_and_number_of_students):
    plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y))
plt.show()

for i in area_of_study:
    c = 0
    for j in major:
        if i == j:
            c += 1
    major_and_number_of_students.append(c)

#for a in major_and_number_of_students:
#    print(a)

plt.xlabel('Area of study')
plt.ylabel('Number of students')
plt.scatter(area_of_study, major_and_number_of_students)
for i_x, i_y in zip(area_of_study, major_and_number_of_students):
    plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y))
plt.show()

#City and college wise classification of students.

city_of_students = []
for i in student_dataset['City']:
    if i not in city_of_students:
        city_of_students.append(i)

college_of_students = []
for i in student_dataset['College name']:
    if i not in college_of_students:
        college_of_students.append(i)

city = student_dataset['City']
college = student_dataset['College name']
city_and_number_of_students = []
college_and_number_of_students= []

for i in city_of_students:
    c = 0
    for j in city:
        if i == j:
            c += 1
    city_and_number_of_students.append([i, c])

for a in city_and_number_of_students:
    print(a)

for i in college_of_students:
    c = 0
    for j in college:
        if i == j:
            c += 1
    college_and_number_of_students.append([i, c])

for a in college_and_number_of_students:
    print(a)

#Plot the relationship between the CGPA and the target variable.

sns.countplot(student_dataset['CGPA/ percentage'],label="Sum")

plt.show()

