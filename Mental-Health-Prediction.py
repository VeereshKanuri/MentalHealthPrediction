#!/usr/bin/env python
# coding: utf-8

# # About data 
# 
# - Open Sourcing Mental Illness is a non-profit, corporation dedicated to raising awareness, educating, and providing resources to support mental wellness in the tech and open source communities. OSMI began in 2013, with Ed Finkler speaking at tech conferences about his personal experiences as a web developer and open source advocate with a mental health disorder. The response was overwhelming, and thus OSMI was born.
# 
# 
# - Every year, OSMI came out with a new survey to see how employees want to get mental health treatment in tech companies around the world and this is the survey from 2014
# 
# - This survey is filled by respondents who suffer from mental health disorders (diagnose or un-diagnosed by medical, even it's just a feeling) in tech companies and see if any factors can affect the employee to get treatment or not.

# # Domain Knowledge
# 
# - Mental health touches every aspect of our lives, especially workplace performance. Key performance indicators such as productivity, creativity, and social engagement can all take a hit if an employee's mental health is suffering. Prioritizing mental health in the workplace can help your workers flourish and reach their full potential, which is what businesses need to thrive and grow.
# 
# 
# 
# - Business leaders say mental health issues can have a negative impact on their operations in the following ways:
#    Revenue decreased (40%)
#    Profitability decreased (39%)
#    Loss of customers (30%)
#    Diminished output (26%)
#    Reduced competitiveness (20%)
# 
# 
# - go through this article to get more info https://www.paychex.com/articles/human-resources/workplace-mental-health-effects
#     
#     
# ### Employer plays important role in this 
# 
# - Mental Health First Aid Training: Offering Mental Health First Aid training to employees can be extremely beneficial. This training equips employees with the skills to recognize signs of mental health or substance use concerns in their colleagues and provide appropriate support. This early intervention can help prevent issues from escalating and create a more supportive environment.
# 
# 
# - Robust Benefit Packages: Offering comprehensive benefit packages that include Employee Assistance Programs (EAPs), wellness programs, health and disability insurance, and flexible working arrangements demonstrates a commitment to employees' well-being. These benefits provide support and resources for employees facing mental health challenges.
# 
# 
# - Promotion of Work-Life Balance: Implementing flexible working schedules or time-off policies allows employees to manage their mental health while maintaining their work responsibilities. A healthy work-life balance can significantly contribute to overall mental well-being.
#     
#     
# - Research shows that employees who go through Mental Health First Aid have an increased awareness of mental health among themselves and their co-workers
# 
# 
# ### Benfit 
# - Nearly 86% of employees report improved work performance and lower rates of absenteeism after receiving treatment for depression, according to an April 2018 article in the Journal of Occupational and Environmental Medicine. This means big gains in retention and productivity for employers. By providing employees access to mental health benefits, the company can begin to create a culture of understanding and compassion at the tech company. And having employees who feel cared for and happy isn’t just good, it’s good business.

# ## understanding description:
# 
# 1 **Timestamp**: The date and time when the survey response was recorded.
# 
# 2 **Age**: The age of the employee who participated in the survey.
# 
# 3 **Gender**: The gender identity of the employee. This could include options like "Male," "Female," "Non-binary," and more.
# 
# 4 **Country**: The country where the employee is located.
# 
# 5 **State**: The state or region within the country where the employee is located.
# 
# 6 **Self_Employed**: Indicates whether the employee is self-employed or not.
# 
# 7 **Family_History**: Indicates whether the employee has a family history of mental health issues.
# 
# 8 **Treatment**: Indicates whether the employee sought treatment for mental health issues (Yes/No).
# 
# 9 **Work_Interfere**: How much the employee's work is affected by their mental health issue.
# 
# 10 **No_Employees**: The number of employees in the company or organization.
# 
# 11 **Remote_Work**: Do you work remotely (outside of an office) at least 50% of the time
# 
# 12 **Tech_Company**: Is your employer primarily a tech company/organization?
# 
# 13 **Benefits**: Does your employer provide mental health benefits?
# 
# 14 **Care_Options**: Do you know the options for mental health care your employer provides?
# 
# 15 **Wellness_Program**: Has your employer ever discussed mental health as part of an employee wellness program?
# 
# 16 **Seek_Help**: Does your employer provide resources to learn more about mental health issues and how to seek help?
# 
# 17 **Anonymity**: Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment?
# 
# 18 **Leave**: How easy is it for you to take medical leave for a mental health condition?
# 
# 19 **Mental_Health_Consequence**: Do you think that discussing a mental health issue with your employer would have negative consequences?
# 
# 20 **Phys_Health_Consequence**: Do you think that discussing a physical health issue with your employer would have negative consequences?
# 
# 21 **Coworkers**: Would you be willing to discuss a mental health issue with your coworkers?
# 
# 22 **Supervisor**: Would you be willing to discuss a mental health issue with your direct supervisor(s)?
# 
# 23 **Mental_Health_Interview**: Would you bring up a mental health issue with a potential employer in an interview?
# 
# 24 **Phys_Health_Interview**: Would you bring up a physical health issue with a potential employer in an interview?
# 
# 25 **Mental_vs_Physical**:Do you feel that your employer takes mental health as seriously as physical health?
# 
# 26**Obs_Consequence**: Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?
# 
# 27 **Comments**: Additional comments or notes provided by the employee.
# 

# In[1]:


pip install xgboost


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings 
warnings.filterwarnings('ignore')


# In[3]:


pd.set_option('display.max_columns',None)
df= pd.read_csv('Mental_health_data.csv')


# # Data Understanding

# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.columns


# In[ ]:





# In[8]:


df.columns=df.columns.str.lower()


# In[9]:


df.info()


# ### observation
# 
# - there are 26 columns in total
# - we can see that except age column all other columns are of object type
# - country,self employed,work_interfere, comments column contains null values
# - comments column contains many null values approx 80 percent because it surveyor kept that as optional

# ### observation
# 
# - column timestamp is useless we can drop that column because it tells when employee filled the form

# In[10]:


df.drop(columns='timestamp',inplace=True)


# In[11]:


df.isnull().sum()


# ### INSIGHT

# - we can observe that there are null values present in cloumns like state,self_employed,work_interfere,comments

# In[12]:


(df.isnull().sum()/df.shape[0])*100


# ### observation

# - we can see 86 percent null values are there in comments colums
# - so it is useless. we can drop it

# # Data Preprocessing

# In[13]:


df.drop(columns='comments',inplace=True)


# In[14]:


df.columns


# In[15]:


for i in df.columns:
    print("--------------------------------")
    print(i,df[i].unique())
    print("--------------------------------")
    


# ### observation
# 
# - we can see that in age column age is 99999999999 and  negative also . it's so silly to see that 
# - let's data get cleaned

# In[16]:


pd.set_option('display.float_format', '{:.0f}'.format)
df.describe()


# ### observation
# - min employee age is 15 which is legal age to work 
# - let's keep max employee age is 75 

# In[17]:


df.loc[df.age<15,'age']=15
df.loc[df.age>75,'age']=75


# In[18]:


df[df['age']<15]


# In[19]:


df.shape


# ### observation

# - we can observe that in gender column there is some noise let's clean it 

# In[20]:


df['gender'].unique()


# - definetly surveyor used text box rather than drop down box for this columns that's why these many classifications
# - Male,m,Male-ish,maile,mal,Male(CIS),Cis Male,msle,Man,mail,cis male,cis man,malr - these all can be considered as males
# - female,Female,Feamke,Cis Female,f,F,cis-female/femme,Feamle(cis),femail-these all can be considered as females

# In[21]:


df['gender'].replace(['Male ', 'male', 'M', 'm', 'Male', 'Cis Male',
                     'Man', 'cis male', 'Mail', 'Male-ish', 'Male (CIS)',
                      'Cis Man', 'msle', 'Malr', 'Mal', 'maile', 'Make',],'male',inplace=True)
df['gender'].replace(['Female ', 'female', 'F', 'f', 'Woman', 'Female',
                     'femail', 'Cis Female', 'cis-female/femme', 'Femake', 'Female (cis)',
                     'woman',], 'Female', inplace = True)

df["gender"].replace(['Female (trans)', 'queer/she/they', 'non-binary',
                     'fluid', 'queer', 'Androgyne', 'Trans-female', 'male leaning androgynous',
                      'Agender', 'A little about you', 'Nah', 'All',
                      'ostensibly male, unsure what that really means',
                      'Genderqueer', 'Enby', 'p', 'Neuter', 'something kinda male?',
                      'Guy (-ish) ^_^', 'Trans woman',], 'Other', inplace = True)


# In[22]:


df['gender'].unique()


# ### observation
# - other include transgender,lesbian etc

# In[23]:


df['gender'].value_counts()


# ### observation
# 
# - we can see that males are approximately 4 times the female. as males are more we can't make wrong assumptions like males are more likely have mental health issues.
# 
# 
# - that doesn't make sense alternatively we can assume that males are dominant in tech than females

# In[24]:


df['country'].value_counts()


# In[25]:


df['state'].value_counts()


# ### observation 
# 
# - we can see max repeated  country is usa
# - we can see most of the states also from usa
# - so there is no use of that columns . let's drop those columns because it will be really misleading to conclude that a certain country or state faces more mental health problems beacuse around  60 percent of people from usa  

# In[26]:


df.drop(columns=['country','state'],inplace =True)


# In[27]:


df.isnull().sum()


# # Exploratory data analysis

# ### each column in a dataset is a servey question. let's look at some intresting questions 

# In[28]:


df.columns


# ### firstly let's focus on target variable (treatment)
# 
# - 1. Q) have you sought treatment for mental health ? 

# In[29]:


df['treatment'].value_counts()


# In[30]:


treatment_counts=df['treatment'].value_counts(normalize=True)


# In[31]:


plt.pie(treatment_counts,labels=treatment_counts.index,autopct='%1.1f%%')


# ### observation 
# 
# - nearly 50 percent employees seek for treatment
# - employees who are seeking for treatement must be treated 

# - 2. Q) Are you a self employed?

# In[32]:


df['self_employed'].value_counts()


# In[33]:


self_employed_counts=df['self_employed'].value_counts()


# In[34]:


sns.barplot(x=self_employed_counts.index,y=self_employed_counts.values,palette = 'Blues_r')


# ### observation
# 
# - most of the people(>1000) belongs to not self_employed class which is working class
# 
# - very few belongs to self_employed

# - 3.Q) will there be a difference in seeking treatement for self_employed and not self_employed ?

# In[35]:


sns.countplot(df['self_employed'], hue = df['treatment'], palette = 'Greys')
plt.title('Employement Type of the Employees who are seeking Treatment',  fontsize=12, fontweight='bold')
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.legend(fontsize=12)


# ### observation
# 
# - even though there is difference in people of self_employed and not self_employed but there no much difference in seeking treatment 
# 
# - in both self_employed and not self_employed 50 percent of people seek for treatment

# - 4.Q) Do your family have mental mental health issues ?

# In[36]:


df['family_history'].value_counts()


# In[37]:


family_health=df['family_history'].value_counts()


# In[38]:


plt.pie(family_health,labels=family_health.index,autopct="%1.1f%%")


# ### observation
# 
# - we can see 60 percent of the employee doesn't have a family history of mental health issues
# - 39 percent of employees have a family history of mental health issues

# - 5.Q) Does a family history of mental health issues affect whether an employee seeks treatment?

# In[39]:


sns.countplot(x=df['family_history'],hue=df['treatment'],palette='OrRd')


# ### observation
# 
# - we can observe that family history of mental health issues affects respective employee to seek treatment
# 
# - Family history is a significant risk factor for many mental health disorders.
#   Thus, this is an important factor that has to be taken under consideration as it influences the behaviour of the employees to a significant extent.

# - Q.6) what is your age ?

# In[40]:


df['age'].hist()


# ### observation
# 
# - The majority of employees who filled the survey form fall within the age range of 28 to 32
# 

# - 7.Q) Does age of employee affects whether to seek treatment or not ?

# In[41]:


sns.boxplot(x=df['age'],y=df['treatment'])


# ### observation
# 
# - we can see that age of employee doesn't affect to seek treatment or not 

# - 8.Q)If you have a mental health condition, do you feel that it interferes with your work?

# In[42]:


df['work_interfere'].value_counts()


# In[43]:


work_interfere=df['work_interfere'].value_counts()


# In[44]:


sns.barplot(x=work_interfere.index,y=work_interfere.values,palette='gist_heat')


# In[45]:


sns.countplot(df['work_interfere'], hue = df['treatment'], palette = 'Accent_r')


# ### observation
# 
# * About 78% of respondents have experienced interference at work with a ratio of rarely, sometimes, and frequently.
# * Mental health conditions sometimes become an interfere while working about 45%. The plots prove that almost 80% want to get treatment. But it's surprising to know even mental health never has interfered at work, there is a little group that still want to get treatment before it become a job stress. It can be triggered by the requirements of the job do not match the capabilities, resources or needs of the worker
# * **If you are running a tech organization , you should consider providing resources for employees seeking treatment and it will help in boosting employee experience and will definitely increase their productivity.**

# - Q.9) No of employees in your workplace or organization?

# In[46]:


no_employees=df['no_employees'].value_counts()


# In[47]:


plt.figure(figsize=(10,5))
sns.barplot(x=no_employees.index,y=no_employees.values)


# ### observation
# 
# - we can see that 6-25,26-100,more than 1000 are equally have same weightage 
# 
# - where as 100-500,1-5,500-1000 are comparitively less
# 

# - Q.10) does size of organization affects likelihood of employees seek of treatement or not?

# In[48]:


sns.countplot(x=df['no_employees'],hue=df['treatment'],palette='gist_ncar')
plt.tight_layout()


# ### observation
# 
# 
# - from above we cannot observe much difference or any relation saying if number of employees are more liklihood of seeking treatment is also more.

# - Q. 11) Does your employer provide mental health benfits?

# In[49]:


df['benefits'].value_counts()


# In[50]:


sns.countplot(df['benefits'],palette='winter_r')


# ### observation
# 
# - we can observe that 37 percent of employees are saying yes and 32 percent of employees are saying don't know benfits and 29 percent are saying no 

# - Q.12) does providing or not providing benfits affects likelihood of seeking treatment or not ?

# In[51]:


sns.countplot(df['benefits'],hue=df['treatment'],palette='icefire_r')


# ### observation
# 
# - employees in organization or company which provide health benfits are more likely to seek treatment
# - employeees in organization or company which doesn't provide benfits are also more likely to seek treatment which  almost 45 percent 

# - Q. 13) does size of organization determines whether it provides benfits or not?

# In[52]:


sns.countplot(df['no_employees'],hue=df['benefits'],palette='YlOrBr')
plt.tight_layout()


# ### observation
# 
# - we can see that companies with large size provide more health benfits and employees in organization which provide more health benfits are more likely to seek treatment
# 
# - we can conclude that employees in large organization are more likely to seek treatment 

# - Q.13)Do you work remotely (outside of an office) at least 50% of the time?

# In[53]:


df['remote_work'].value_counts()


# In[54]:


sns.countplot(df['remote_work'],palette='gnuplot2')


# - Q.14) does working or not working remotely determines whether employee seek treatment or not?

# In[55]:


sns.countplot(df['remote_work'],hue=df['treatment'],palette='Dark2')


# ### observation
# 
# - we can see that there is no much difference in above for both the comparisons(employees who work 50 percent remotely and who doesn't)

# - Q.15) Is your employer primarily a tech company/organization?

# In[56]:


tech_nontech=df['tech_company'].value_counts()


# In[57]:


plt.pie(tech_nontech,labels=['Tech','Non-Tech'],autopct='%1.1f%%')


# ### observation
# 
# - 81 percent of employees who filled survey belongs to tech organization
# 
# - 18 percent of employees belongs to non-tech organization

# - Q.) Does working in tech/non-tech determines whether employee likely to seek treatment or not?

# In[58]:


sns.countplot(df['tech_company'],hue=df['treatment'],palette='Set1_r')


# ### observation
# 
# - From the observation, it can be seen that employees working in both tech and non-tech sectors are equally likely to seek treatment 
# 
# - we cannot conclude anything from this 

# - Q. 17) Has your employer ever discussed mental health as part of an employee wellness program?

# In[59]:


df['wellness_program'].value_counts()


# In[60]:


sns.countplot(df['wellness_program'],palette='afmhot_r')


# ### observation
# 
# - from the bar plot we can see that most of the employees are saying employer never discussed mental health as part of an employee wellness program
# 
# - very few of them are  saying never discussed mental health as part of an employee wellness program
# 
# - remaining are saying they don't even know about wellness program

# - Q. 18) does discussing wellness program determines whether employee seek treatment or not?

# In[61]:


sns.countplot(df['wellness_program'],hue=df['treatment'],palette='RdGy')


# ### observation
# 
# - Despite over 60 percent of employees indicating that their employer never discussed mental health as part of a wellness program, among this group, 50 percent still express a desire to seek treatment
# 
# - company need to fulfill its duty and provide it soon so that employees can work more efficiently

# - Q.21) Does your employer provide resources to learn more about mental health issues and how to seek help?

# In[62]:


df['seek_help'].value_counts()


# In[63]:


sns.countplot(df['seek_help'],palette='rainbow_r')


# ### observation
# 
# - most of employees are saying that our employer doesn't provide resources to learn more about mental health issues and how to seek help
# 
# - very few employees are saying yes
# 
# - others say they dodn't know 

# In[64]:


sns.countplot(df['seek_help'],hue=df['treatment'],palette='twilight_shifted_r')


# ### observation
# 
# - almost 50 percent of employees in every category seek help
# 
# - Companies should provide resources to help employees learn more about mental health issues and how to seek help, as this can benefit up to 50 percent of employees

# - Q.22) Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment?

# In[65]:


df['anonymity'].value_counts()


# In[66]:


sns.countplot(df['anonymity'],palette='viridis_r')


# ### observation
# 
# -  Most people don't know whether their anonymity is protected if they decided to take seek treatment.
# -  30 percent of people are sure that thier anonymity is protected
# -  very few people are sure that thier anonymity is not protected 

# - Q.23)Does the protection of anonymity influence whether employees are more likely to seek treatment ?

# In[67]:


sns.countplot(df['anonymity'],hue=df['treatment'],palette='PuBu_r')


# ### observation
# 
# - from the above observation  it is evident that employees whose anonymity is protected are more likely to seek treatment
# 
# - Possible reasoning for this may be that the employee feels that the company has protected his/her privacy and can be trusted with knowing the mental health condition of it's workers
# 
# - On the other hand, employees who are unsure whether their anonymity is protected or not are less likely to receive treatment.
# 

# - Q.24) How easy is it for you to take medical leave for a mental health condition?

# In[68]:


df['leave'].value_counts()


# In[69]:


plt.figure(figsize=(10,5))
sns.countplot(df['leave'],palette= 'prism_r')
plt.tight_layout()


# ### observation
# 
# - from the above plot we can observe that most of the employees don't know whether it is easy to take medical leave or not 
# 
# - remaing most says some what easy and very easy 
# 
# - few says that some what difficult and very difficult

# - Q. 25) does ease of taking medical leave determines whether the employees likely to seek treatment or not?

# In[70]:


plt.figure(figsize=(10,5))
sns.countplot(df['leave'],hue=df['treatment'],palette='inferno')


# ###  observation
# 
# - While close to 50% of the people answered that they did not know about it, suprisingly around 45% of those people sought help for their condition.
# 
# - A small percent of people ( around 8% ) said that it was very difficult for them to get leave for mental health and out of those, 75% of them sought for help.
# 
# - Employees who said it was 'somewhat easy' or 'very easy' to get leave had almost 50% people seeking medical help.

# - Q.26)  Do you think that discussing a mental health issue with your employer would have negative consequences?

# In[71]:


value_counts_normalized=df['mental_health_consequence'].value_counts(normalize=True)


# In[72]:


for value, percentage in value_counts_normalized.items():
    print(f'{value}: {percentage:1.1%}')


# In[73]:


sns.countplot(df['mental_health_consequence'],palette='cividis')


# ### observation
# 
# - 39 percent of the employees said no which means that discussing a mental health issue with thier employer wouldn't have negative consequences
# - 38 percent of employees are unsure 
# - 23 percent of employees said yes that means discussing a mental health issue with thier employer would have negative consequences

# In[74]:


sns.countplot(df['mental_health_consequence'],hue=df['treatment'],palette='cividis')


# ### observation
# 
# - we can observe that out of the people who answered No, there were only around 40% of the people who actually sought after help, whereas in both the other categories, it is more than 50%

# - Q.28) Do you think that discussing a physical health issue with your employer would have negative consequences?

# In[75]:


value_counts_normalized=df['phys_health_consequence'].value_counts(normalize=True)


# In[76]:


for value,percentage in value_counts_normalized.items():
    print(f'{value}: {percentage:.1%}')


# In[77]:


sns.countplot(df['phys_health_consequence'],palette='hsv_r')


# ### observation
# 
# - we can observe almost 73.5 percent of poeple are saying no that means discussing a physical health issue with their employer wouldn't have negative consequences
# 
# - we can observe only 2.7 percent of employees says yes 
# 
# - remaining 21 percent says don't know
# 
# - There is a starking difference between the reponses for the same question regarding mental and physical health. More than 70% of the employees believe that their physical health does not create a negative impact on their employer and only 5% of them believes that it does

# In[78]:


sns.countplot(df['phys_health_consequence'],hue=df['treatment'],palette='twilight_shifted_r')


# ### observation
# 
# - While it maybe incorrect for us to draw any conclusions about whether they seek mental help on the basis of their physical condition, because it is more or less same for all the three categories, we must keep in mind about how differently mental and physical health are treated as a whole.

# - Q. 28) Would you be willing to discuss a mental health issue with your coworkers?

# In[79]:


df['coworkers'].value_counts()


# In[80]:


sns.countplot(df['coworkers'],palette='gist_earth_r')


# ### observation 
# 
# -  It is good sign that most people have atleast some people(coworkers) to talk to about the mental health issues.

# In[81]:


sns.countplot(df['coworkers'],hue=df['treatment'],palette='gist_earth_r')


# ### observation
# 
# - there is almost equal ration of seek or not seek treatment in three categories

# - Q.29) Would you be willing to discuss a mental health issue with your direct supervisor(s)?

# In[82]:


df['supervisor'].value_counts()


# In[83]:


sns.countplot(df['supervisor'])


# ### observation
# 
# - most of them says yes unlike in the coworkers
# 
# - They may feel that supervisors are better equipped to offer support and guidance in handling sensitive issues like mental health.

# In[84]:


sns.countplot(df['supervisor'],hue=df['treatment'],palette='gist_earth')


# ### observation 
# 
# - there is almost equal ration of seek or not seek treatment in three categories

# - Q.30)Would you bring up a mental health issue with a potential employer in an interview?

# In[85]:


df['mental_health_interview'].value_counts()


# In[86]:


sns.countplot(df['mental_health_interview'])


# ### observation
# 
# - we can observe that most of the employees said no beacuase they are not intrested to bring up a mental health issue with a potential employer in an interview.
# - very few said may be 

# In[87]:


sns.countplot(df['mental_health_interview'],hue=df['treatment'],palette='mako')


# ### observation
# 
# - there is almost equal ration of seek or not seek treatment in three categories

# - Q.31)Would you bring up a physical health issue with a potential employer in an interview?

# In[88]:


sns.countplot(df['phys_health_interview'])


# # observation 
# 
# - there is difference in discussing mental health in interview and physical health in interview as we can see most of employees said may be.
# - in case of mental health they said no 
# - While a majority of the people are still dubious about discussing their physical health condition with the future employer, however, close to 17% believe that there is no issue in discussing their physical health conditions.
# - Around 50% of the people still remain confused about whether it is a good option to discuss their condition or not

# - Q.32)Do you feel that your employer takes mental health as seriously as physical health?

# In[89]:


sns.countplot(df['mental_vs_physical'],palette='afmhot')


# In[90]:


sns.countplot(df['mental_vs_physical'],hue=df['treatment'],palette='gist_earth_r')


# ### observation
# 
# - Employees who think their company doesn't take mental health seriously or who are not sure, are more inclined to seek treatment than the other two categories

# - Q.33)  Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?

# In[91]:


sns.countplot(df['obs_consequence'],palette='gist_rainbow')


# In[92]:


sns.countplot(df['obs_consequence'],palette='YlOrRd_r',hue=df['treatment'])


# ### observation
# 
# -  Almost 85% of people never heard of or observed co-workers having negative consequences for having mental health issues.
# - Out of remaining people,who observed negative consequences for co-workers, 10% of them are seeking help.

# ## We are done with our EDA, We have got some valuable insights for the companies on grabing the best interests of employees.This will be useful for Human resource teams to decide on to introduce new wellness programs,resources etc for employees. 

# ### Data Preprocessing

# In[93]:


df.info()


# ### Replacing null values
# 
# - we can observe there are almost 20 percent null values in column work interfere 
# 
# - in case of self employed null values are very less(~ 2 percent)
# 
# - let's replace null values with mode in case of self employed as there are very less null values
# 
# - let's go by logical in case of work interfere

# In[94]:


df['self_employed']=df['self_employed'].fillna(df['self_employed'].mode()[0])


# ##### df[df['work_interfere'].isna()]['treatment'].value_counts()

# In[ ]:





# ### observation
# 
# - We can observe a notable trend among employees who haven't filled out the 'work_interfere' column—they are more likely to have not received treatment
# 
# -  It's reasonable to infer that if these employees had provided a response, it would likely align with 'never.' This inference is supported by our EDA findings, which indicated that employees who reported 'never' are also most likely to not receive treatment

# In[96]:


df['work_interfere']=df['work_interfere'].fillna('Never')


# In[97]:


df['work_interfere']


# In[98]:


df.info()


# ## we are done with eda and data preprocessing steps . lets go and build ML model 

# In[99]:


df.columns


# * **except age column all other columns are categorical** 
# 
# * **to build a ml model we have convert this categorical to numerical** 
# 
# * **to convert we have to use any one of encoding methods that best suits these columns**

# In[100]:


for i in df.columns:
    print("--------------------------------")
    print(i,df[i].unique())
    print("--------------------------------")


# * **All these columns except age column consist of nominal categories. OHE preserves the distinctness of each category without imposing any ordinal relationship**
# 
# * **But one hot encoding creates many columns that may leads to overfitting**
# 
# * **let's use label encoder**

# In[101]:


df['treatment'].value_counts()


# In[102]:


object_cols = ['gender', 'self_employed', 'family_history', 'treatment',
       'work_interfere', 'no_employees', 'remote_work', 'tech_company',
       'benefits', 'care_options', 'wellness_program', 'seek_help',
       'anonymity', 'leave', 'mental_health_consequence',
       'phys_health_consequence', 'coworkers', 'supervisor',
       'mental_health_interview', 'phys_health_interview',
       'mental_vs_physical', 'obs_consequence']


# In[103]:


from sklearn.preprocessing import LabelEncoder


# In[104]:


LE=LabelEncoder()


# In[105]:


for col in object_cols:
    LE.fit(df[col])
    
    df[col]=LE.fit_transform(df[col])
    


# In[106]:


df.head()


# ### standardization
# 
# * **let's standardize age column**
# 
# * **beacause in age column values range blw 15 to 65**

# In[107]:


from sklearn.preprocessing import StandardScaler


# In[108]:


scaler=StandardScaler()


# In[109]:


df['age']=scaler.fit_transform(df[['age']])


# In[110]:


df.head()


# ### Train Test split

# In[111]:


from sklearn.model_selection import train_test_split


# In[112]:


x = df.drop('treatment', axis = 1)
y = df['treatment']


# In[113]:


x_train,x_test, y_train, y_test = train_test_split(x,y,stratify = y,
                                                    test_size = 0.15,
                                                   random_state = 42)


# ### Model buliding and Evaluation

# -   **TN: Employee's Mental Health predict with No Treatment and the actual is No Treatment**
# -   **TP: Employee's Mental Health predict with Get Treatment and the actual is Get Treatment**
# -   **FP: Employee's Mental Health predict with Get Treatment and the actual is No Treatment**
# -   **FN: Employee's Mental Health predict with No Treatment and the actual is Get Treatment**

# ### observation
# 
# - False Positives (FP): Predicting that an employee will seek treatment (Get Treatment) when they actually won't (No Treatment).
#    - Implication: This might lead to unnecessary allocation of resources or interventions for employees who don't need them.
# - False Negatives (FN): Predicting that an employee won't seek treatment (No Treatment) when they actually will (Get Treatment).
# 
#    -    Implication: This could potentially result in employees who need treatment not receiving the necessary support, which could have negative consequences for their well-being.

# * **whenever both fp and fn are important then we use f1 score as performance metric**

# ### Logistic Regression

# In[114]:


from sklearn.linear_model import LogisticRegression


# In[115]:


LR=LogisticRegression(penalty='l1',solver='liblinear')


# In[116]:


LR.fit(x_train,y_train)


# In[117]:


y_pred=LR.predict(x_test)


# In[118]:


from sklearn.metrics import f1_score


# In[119]:


f1_score(y_test,y_pred)


# In[120]:


from sklearn.model_selection import cross_val_score


# In[121]:


f1score=cross_val_score(LR,x_train,y_train,scoring='f1',cv=10)


# In[122]:


np.mean(f1score)


# ### Decision Tree

# - for small datasets it is better to use entropy as a purity check 

# In[123]:


from sklearn.tree import DecisionTreeClassifier


# In[124]:


DT=DecisionTreeClassifier(random_state=42,max_depth=4)


# In[125]:


DT.fit(x_train,y_train)


# In[126]:


y_pred=DT.predict(x_test)


# In[127]:


f1_score(y_test,y_pred)


# In[128]:


from sklearn import tree


# In[129]:


plt.figure(figsize=(10,5))
tree.plot_tree(DT,filled=True)


# In[130]:


f1score=cross_val_score(DT,x_train,y_train,scoring='f1',cv=10)


# In[131]:


np.mean(f1score)


# ### SVM

# In[132]:


from sklearn.svm import SVC


# In[133]:


classifier=SVC()


# In[134]:


classifier.fit(x_train,y_train)


# In[135]:


y_pred=classifier.predict(x_test)


# In[136]:


f1_score(y_test,y_pred)


# In[137]:


f1score=cross_val_score(DT,x_train,y_train,scoring='f1',cv=10)


# In[138]:


np.mean(f1score)


# ### Random Forest

# In[139]:


from sklearn.ensemble import  RandomForestClassifier


# In[140]:


rfc=RandomForestClassifier(random_state=42)


# In[141]:


rfc.fit(x_train,y_train)


# In[142]:


y_pred=rfc.predict(x_test)


# In[143]:


f1_score(y_test,y_pred)


# In[144]:


f1score=cross_val_score(rfc,x_train,y_train,scoring='f1',cv=10) 


# In[145]:


np.mean(f1score)


# ### Boosting Models

# In[146]:


from sklearn.ensemble import AdaBoostClassifierh


# In[147]:


abc=AdaBoostClassifier()


# In[148]:


abc.fit(x_train,y_train)


# In[149]:


y_pred=abc.predict(x_test)


# In[150]:


f1_score(y_test,y_pred)


# In[151]:


f1score=cross_val_score(abc,x_train,y_train,scoring='f1',cv=10)


# In[152]:


np.mean(f1score) 


# In[153]:


from sklearn.ensemble import GradientBoostingClassifier


# In[154]:


gbc=GradientBoostingClassifier(random_state=42,subsample=0.8)


# In[155]:


gbc.fit(x_train,y_train)


# In[156]:


y_pred=gbc.predict(x_test)


# In[157]:


f1_score(y_test,y_pred)


# In[158]:


f1score=cross_val_score(gbc,x_train,y_train,scoring='f1',cv=10)


# In[159]:


np.mean(f1score)


# In[160]:


from xgboost import XGBClassifier


# In[161]:


xgb_clf = XGBClassifier(verbosity = 0)


# In[162]:


xgb_clf.fit(x_train,y_train)


# In[163]:


y_pred=xgb_clf.predict(x_test)


# In[164]:


f1_score(y_test,y_pred)


# In[165]:


f1score=cross_val_score(xgb_clf,x_train,y_train,scoring='f1',cv=10)


# In[166]:


np.mean(f1score)


# ### shortlisting best models

# - SVM
# - RANDOM FOREST
# - GRADIENT BOOSTING
# - XGBOOST

# ### HYPERPARAMETER TUNING OF ABOVE MODELS 

# ### Tuning svc model

# In[167]:


from sklearn.model_selection import GridSearchCV
param_distribs = {
        'kernel': ['linear', 'rbf','polynomial'],
        'C': [0.01,0.01,0.1,0.15,0.2,0.25,0.5,0.75,1,2,10,100],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    }
svm_clf = SVC()
grid_cv = GridSearchCV(svm_clf , param_grid = param_distribs,
                    \we          cv=5,scoring='f1',
                              verbose=1)
grid_cv.fit(x_train,y_train)


# In[168]:


grid_cv.best_estimator_


# In[169]:


grid_cv.best_estimator_.fit(x_train,y_train)


# In[170]:


y_pred=grid_cv.best_estimator_.predict(x_test)


# In[171]:


f1score= f1_score(y_test,y_pred)


# In[172]:


f1score


# In[173]:


f1score=cross_val_score(grid_cv.best_estimator_,x_train,y_train,scoring='f1',cv=10)


# In[174]:


np.mean(f1score)


# ### Tuning Random Forest 

# In[175]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators':[3,10,30,50,100],'max_features':[2,4,6,8],'max_depth' : [1,2,3,4]}
]



forest_clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(forest_clf, param_grid, cv=5,
                           scoring='f1',
                           return_train_score=True)
grid_search.fit(x_train, y_train)


# In[176]:


grid_search.best_estimator_


# In[177]:


grid_search.best_estimator_.fit(x_train,y_train)


# In[178]:


y_pred=grid_search.best_estimator_.predict(x_test)


# In[179]:


f1score= f1_score(y_test,y_pred)


# In[180]:


f1score


# In[181]:


f1score=cross_val_score(grid_search.best_estimator_,x_train,y_train,scoring='f1',cv=10)


# In[182]:


np.mean(f1score)


# ### Tuning Gradient Boosting model

# In[183]:


param_grid = [
    {
        'n_estimators': [3, 10, 30, 50, 100],
        'max_features': [2, 4, 6, 8, 10],
        'max_depth': [1, 2, 3, 4] ,
        'subsample': [0.25, 0.5, 0.75]
    }
]



gdb_clf2 = GradientBoostingClassifier(random_state=42)
grid_search2 = GridSearchCV(gdb_clf2, param_grid, cv=5,
                           scoring='f1',
                           return_train_score=True)
grid_search2.fit(x_train, y_train)


# In[184]:


grid_search2.best_estimator_


# In[185]:


grid_search2.best_estimator_.fit(x_train,y_train)


# In[186]:


y_pred=grid_search2.best_estimator_.predict(x_test)


# In[187]:


f1score= f1_score(y_test,y_pred)


# In[188]:


f1score


# In[189]:


f1score=cross_val_score(grid_search2.best_estimator_,x_train,y_train,scoring='f1',cv=10)


# In[190]:


np.mean(f1score)


# ### Tuning Xg Boosting 

# In[191]:


param_grid = [
    {'n_estimators':[3,10,30,50,100],
    'eta' : [0.01,0.025, 0.05, 0.1],
    'max_features':[2,4,6,8],
    'max_depth' : [1,2,3,4],
    'subsample': [0.5,0.75],
    'booster':['gblinear','gbtree']}
]

xgb_clf = XGBClassifier(verbosity = 0)
grid_search3 = GridSearchCV(xgb_clf, param_grid, cv=5,
                           scoring='f1',
                           return_train_score=True)
grid_search3.fit(x_train, y_train)


# In[192]:


grid_search3.best_estimator_


# In[193]:


grid_search3.best_estimator_.fit(x_train,y_train)


# In[194]:


y_pred=grid_search3.best_estimator_.predict(x_test)


# In[195]:


f1score=f1_score(y_test,y_pred)


# In[196]:


f1score


# In[197]:


f1score=cross_val_score(grid_search.best_estimator_,x_train,y_train,scoring='f1',cv=10)


# In[198]:


np.mean(f1score)


# ### I finally found xg booster is giving high F1-score among shortlisted models
