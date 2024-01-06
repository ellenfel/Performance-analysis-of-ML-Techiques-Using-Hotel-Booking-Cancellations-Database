# DATA ACQUISITION

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel, SelectPercentile
from sklearn.metrics import f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

pd.set_option('display.max_columns', None)

df = pd.read_csv("data/hotel_bookings.csv")

#df2 = pd.read_csv("data/hotel_booking.csv") personal info

df.head(10)

df.info()

"""

**Data types:**

* Categorical - hotel, is_canceled, customer_type, is_repeated_guest, meal, * 
* country, market_segment, distribution_channel, reserved_room_type, * 
* assigned_room_type, deposit_type, agent, company, reservation_status, *

* Numerical - lead_time, stays_in_weekend_nights, stays_in_week_nights, * 
* adults, children, babies, previous_cancellations, booking_changes, * 
* previous_bookings_not_canceled, days_in_waiting_list, adr, * 
* required_car_parking_spaces, total_of_special_requests, * 

* Ordinal - arrival_date_year, arrival_date_month, arrival_date_week_number, * 
* arrival_date_day_of_month,  reservation_status_date

"""

df.describe()

"""

* The following columns; *
* previous_cancellations, previous_bookings_not_canceled, booking_changes, *
* days_in_waiting_list, required_car_parking_spaces, total_of_special_requests * 
* have only a maximum value. This shows that these features contribute to the *
* decision of cancellation only in very few cases. *

"""

"""
**Assumptions about impact of features:**
* High:
hotel, lead_time, arrival_date_year, arrival_date_month, 
stays_in_weekend_nights, stays_in_week_nights, is_repeated_guest, 
previous_cancellations, previous_bookings_not_canceled, reserved_room_type, 
assigned_room_type, deposit_type, days_in_waiting_list, customer_type

* Medium:
children, babies, distribution_channel, booking_changes, adr

* Low:
arrival_date_week_number, arrival_date_day_of_month,country, meal, adults, 
market_segment, agent, company, required_car_parking_spaces, 
total_of_special_requests, reservation_status, reservation_status_date
"""

"""
**Assumptions about cancellation:**
1. The type of hotel decides the cancelation rate with higher cancellations 
in city hotels as compared to resort hotels due to variety of facilities 
available in resort hotels.

2. The earlier the booking made, higher the chances of cancellation.

3. Customers who have bookings for longer durations have lesser chance of 
cancelling their booking. 

4. As more children or babies are involved in the booking, higher chances of 
cancellation.

5. Old guest (is_repeated_guest=1) is less likely to cancel current booking.

6. If there are high previous cancellations, possibility of cancellation of 
current booking is also high.

7. If room assigned is not the same as reserved room type, 
customer might positively cancel the booking.

8. Higher the number of changes made to the booking, lesser is the chance of 
cancellation due to the investment of time in curating the booking as per 
one's requirement.

9. Bookings that are refundable or for which deposits were not made at the 
time of booking stand a high chance of cancelation.

10. If the number of days in waiting list is significant, customer might make 
some other booking due to uncertainty of confirmation of current booking.
"""

"""**Target variable: 
    is_canceled"""
    
    
    
# EXPLORATORY DATA ANALYSIS

""" ***UNIVARIATE ANALYSIS (Checking the validity of assumptions)***"""
    
is_can = len(df[df['is_canceled']==1])
print("Percentage cancelation= ", is_can/len(df))
df['reservation_status'].value_counts(normalize=True)*100


# highest positive correlations : lead_time followed by previous_cancellations
# highest negative correlations : total_of_special_requests, required_car_parking_spaces
corr= df.corr(method='pearson')['is_canceled'][:]
corr
 
df.insert(2,"is_cancelled_str",df['is_canceled'].astype(str)) # for hue

#df['is_canceled'] = df['is_canceled'].astype(str)
sns.countplot(data=df, x='hotel', hue='is_canceled')
resort_canceled = df[(df['hotel']=='Resort Hotel') & (df['is_canceled']==1)]
city_canceled = df[(df['hotel']=='City Hotel') & (df['is_canceled']==1)]
print('Cancelations in resort hotel= ', (len(resort_canceled))/(len(df[df['hotel']=='Resort Hotel'])))
print('Cancelations in city hotel= ', (len(city_canceled))/(len(df[df['hotel']=='City Hotel'])))  
#Our 1st assumption, city hotels have higher cancelation rate than resort hotels, is valid.
# what is coralation?
# cant quite make heads and tails of the above code block


grid = sns.FacetGrid(df, col='is_canceled')
grid.map(plt.hist, 'lead_time', width=50)
grid.add_legend()
"""
Maximum cancelations occur if the booking is made 60-70 days before the checkin 
 date. Longer the lead_time, lower is the cancelation. This invalidates our 
 2nd assumption. 
"""


print(len(df[(df['stays_in_weekend_nights']==0) & (df['stays_in_week_nights']==0)])) 
#715
"""715 bookings don't have both weekday or weekend nights which could be an 
 error in the data as this is not possible in real life scenario. Therefore 
 these rows can be eliminated from the dataset."""


((len(df.loc[(df['children']!=0) | (df['babies']!=0)]))/(len(df))) * 100
"""The number of customers having children or babies or both are only 8% of the
 total population. Therefore this information can be ignored as it will not play 
 a significatn role in deciding whether to cancel the booking or not. 
 Assumption 4 can be discarded."""


sns.countplot(data=df, x='is_repeated_guest', hue='is_canceled')

new_guest = df[(df['is_repeated_guest']==0) & (df['is_canceled']==1)]
old_guest = df[(df['is_repeated_guest']==1) & (df['is_canceled']==1)]
print('Cancelations among new guests= ', (len(new_guest))/(len(df[df['is_repeated_guest']==0])))
print('Cancelations among old guests= ', (len(old_guest))/(len(df[df['is_repeated_guest']==1])))
"""As seen in the correlation table, the above graph bolsters the evidence that
 maximum customers are new comers and they are less likely to cancel their 
 current booking. Old guests are less likely to cancel the booking (14%).
 Assumption 5 holds true."""


sns.countplot(data=df, x='previous_cancellations', hue='is_canceled')
"""Maximum customers have 0 previous cancellations. They are less likely to
 cancel the current booking. However, customers who have cancelled once 
 earlier are more likely to cancel the current booking. This also matches with
 the positive correlation between previous_cancellations and is_cancelled and
 supports Assumption 6.
"""


temp = df.loc[df['reserved_room_type']!=df['assigned_room_type']]
temp['is_canceled'].value_counts(normalize=True)*100
"""Assumption 7 that there more cancellations when assigned room type is 
 different from reserved room type is not valid. There are only 5% 
 cancellations in such a case.
"""


sns.pointplot(data=df, x='booking_changes', y='is_canceled')
"""Assumption 8 about the bookings does not hold as there is no trend in 
 it's impact on the cancellation of bookings."""
 
 
sns.countplot(x="deposit_type", hue="is_cancelled_str",data=df);
"""Contrary to assumption 9, bookings that are non_refundable are canceled."""


sns.relplot(data=df, x='days_in_waiting_list', y='is_canceled', kind='line', estimator=None)
"""No relation can be established between days_in_waiting_list and is_canceled.
 Therefore, we will take this feature for further analysis. Assumption 10 can 
 be discarded."""


#df['is_canceled_str']= df['is_canceled'].astype(str)
sns.countplot(data=df, x='arrival_date_year', hue='is_canceled_str')

chart = sns.catplot(data=df, x='arrival_date_month', hue='is_canceled_str', kind='count')
chart.set_xticklabels(rotation=65, horizontalalignment='right')
"""Maximum bookings occur in 2016 in the months of July and August."""

year_count = df.groupby(['arrival_date_year', 'is_canceled']).size().to_frame(name='count')
year_perct = year_count.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
print(year_perct)

month_count = df.groupby(['arrival_date_month', 'is_canceled']).size().to_frame(name='count')
month_perct = month_count.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
print(month_perct)
"""Percentage of cancellations was higher in 2015 and 2017 despite higher 
number of bookings in 2016. April and June had the largest 
cancellations overall."""


chart = sns.catplot(data=df, x='market_segment', kind='count', hue='is_canceled_str')
chart.set_xticklabels(rotation=65, horizontalalignment='right')

sns.countplot(data=df, x='distribution_channel', hue='is_canceled_str')

print(df['customer_type'].value_counts(normalize=True)*100)
sns.countplot(data=df, x='customer_type', hue='is_canceled_str')

"""75% bookings occur in Transient category of customers. It also sees the
 highest cancellation among all the categories."""

df['reservation_status'].unique()

grid = sns.FacetGrid(df, col='arrival_date_year')
grid.map(sns.countplot, 'hotel')
"""In all three years city hotels saw more bookings than resort hotels."""


df['meal'].nunique(), df['customer_type'].nunique()
grid = sns.FacetGrid(df, col='customer_type')
grid.map(sns.countplot, 'meal')
"""All kinds of customers prefer BB type meals majorly."""


df.pivot_table(columns='hotel', values='country', aggfunc=lambda x:x.mode())
"""People from country with ISO code 'PRT' made the most number of bookings in 
 both types of hotels."""


g = sns.countplot(data=df, x='hotel', hue='reserved_room_type')
g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
"""Resort hotels room preference : A, D, E
 City hotels room preference : A, D, F"""


print("TABLE 1")
print(df.groupby(['hotel', 'customer_type']).size())
"""For each kind of hotel, Transient type of customers are the highest followed
 by Transient Party. Group bookings are the least."""


print(df.groupby(['customer_type', 'deposit_type']).size())
"""Each category of customers book hotels without deposit. Surprisingly, 
 between refundable and non-refundable type, higher number of people book 
 hotels that are non-refundable."""


print(df.groupby(['customer_type', 'distribution_channel']).size())
print("-"*60)
print(df.groupby(['customer_type', 'market_segment']).size())

print(df.groupby(['hotel', 'distribution_channel']).size())
print("-"*40)
print(df.groupby(['hotel', 'market_segment']).size())

"""Combining table 1 and above table, we see the relation between freqeunt 
 customer types at each hotel and their mode of booking. This information can 
 be used by the hotel to focus on customised publicity stratgies. Similarly the
 market segments can be analysed for a more customer centric approach.  
 Hotel type with distribution channel and market segment can also be analysed. """
 
group = df.groupby(['customer_type', 'reservation_status']).size()
group_pcts = group.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
group_pcts

df.pivot_table(columns='hotel', values=['stays_in_weekend_nights', 
                            'stays_in_week_nights'], aggfunc=lambda x:x.sum())

df.pivot_table(columns='hotel', values='total_of_special_requests', aggfunc=lambda x:x.sum())

sns.catplot(data=df, x='hotel', y='days_in_waiting_list', jitter=False, 
            palette=sns.color_palette(['blue', 'orange']))
"""As it is seen, city hotels have much larger waiting time in days compared 
 to resort hotels which may signify that their demad is higher."""
 

df['country'].value_counts(normalize=True)*100
temp = df.loc[(df['country']=='PRT') | (df['country']=='GBR') 
              | (df['country']=='FRA') | (df['country']=='ESP') 
              | (df['country']=='DEU')]
grid = sns.FacetGrid(temp, col='country')
grid.map(sns.countplot, 'distribution_channel')
"""Using this information hotels can implement models of publicity for getting 
 more bookings in the top 5 countries from where most of their customers hail."""
 
 
sns.barplot(data=df, x='customer_type', y='total_of_special_requests', ci=None)
sns.boxplot(data=df, x='distribution_channel', y='lead_time')



































