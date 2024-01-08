# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:21:09 2018

@author: Dinesh
"""
''' I searched for a month and weekday sort function to do certain work as part of my project work.
I did get only sort function which sorts months and weekdays by alphabetical order but I need them in chronological 
order. I did not find any function which gives the results I wanted, so I am writing below functions
to get the solution I looked for. '''

month_dict = {'Jan':1,'January':1,'jan':1,'january':1,'JAN':1,'JANUARY':1, 'Feb':2,'February':2,'feb':2,'february':2,'FEB':2,'FEBRUARY':2, 
                  'Mar':3,'March':3,'mar':3,'march':3,'MAR':3,'MARCH':3, 'Apr':4,'April':4,'apr':4,'april':4,'APR':4,'APRIL':4, 
                  'May':5,'may':5,'MAY':5, 'Jun':6,'June':6,'jun':6,'june':6,'JUN':6,'JUNE':6, 
                  'Jul':7,'July':7,'jul':7,'july':7,'JUL':7,'JULY':7, 'Aug':8,'August':8,'aug':8,'august':8,'AUG':8,'AUGUST':8, 
                  'Sep':9,'September':9,'sep':9,'september':9,'SEP':9,'SEPTEMBER':9, 'Oct':10,'October':10,'oct':10,'october':10,'OCT':10,'OCTOBER':10, 
                  'Nov':11,'November':11,'nov':11,'november':11,'NOV':11,'NOVEMBER':11, 'Dec':12,'December':12,'dec':12,'december':12,'DEC':12,'DECEMBER':12}

WeekDay_dict = {'Mon':1,'Monday':1,'mon':1,'monday':1,'MON':1,'MONDAY':1, 'Tue':2,'Tuesday':2,'tue':2,'tuesday':2,'TUE':2,'TUESDAY':2, 
                  'Wed':3,'Wednesday':3,'wed':3,'wednesday':3,'WED':3,'WEDNESDAY':3, 'Thu':4,'Thursday':4,'thu':4,'thursday':4,'THU':4,'THURSDAY':4, 
                  'Fri':5,'Friday':5,'fri':5,'friday':5,'FRI':5,'FRIDAY':5, 'Sat':6,'Saturday':6,'sat':6,'saturday':6,'SAT':6,'SATURDAY':6, 
                  'Sun':7,'Sunday':7,'sun':7,'sunday':7,'SUN':7,'SUNDAY':7}

def Month_Sorted_Month(Months):
    value_list = [month_dict[x] for x in Months]
    new_dict = dict(zip(Months, value_list))
    sorted_dict = sorted(new_dict.items(), key=lambda x: x[1])
    sorted_month = [x[0] for x in sorted_dict]
    return(sorted_month)

def Month_Number(Months):
    value_list = [month_dict[x] for x in Months]
    new_dict = dict(zip(Months, value_list))
    #sorted_dict = sorted(new_dict.items(), key=lambda x: x[1])
    #sorted_month_Number = [x[1] for x in sorted_dict]
    return(list(new_dict.values()))

def Weekday_Sorted_Week(Weekdays):
    value_list = [WeekDay_dict[x] for x in Weekdays]
    new_dict = dict(zip(Weekdays, value_list))
    sorted_dict = sorted(new_dict.items(), key=lambda x: x[1])
    sorted_weekday = [x[0] for x in sorted_dict]
    return(sorted_weekday)

def Weekday_Number(Weekdays):
    value_list = [WeekDay_dict[x] for x in Weekdays]
    new_dict = dict(zip(Weekdays, value_list))
    #sorted_dict = sorted(new_dict.items(), key=lambda x: x[1])
    #sorted_weekday_Number = [x[1] for x in sorted_dict]
    return(list(new_dict.values())) # -*- coding: utf-8 -*-

