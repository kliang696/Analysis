# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%[markdown]
#
# # HW08
# ## Calling on all Weather enthusiasts
#
# Okay, I want to collect weather info for my family living at different zip codes. 
# Ideally, I can put the codes on pythonanywhere, just like the other class video showed, 
# so that it can be set as a cron job on a reliable server 
# instead of on my own laptop.  
# As we did in class, we can use weather.gov to get the weather info 
# for the different zip codes. But the way that website is designed, 
# we cannot encode the zip code into the url. So we used 
# selenium to automate. But we cannot install selenium on 
# pythonanywhere. That is using too much resources on the public server.
# 
# Good thing there are so many options out there. I found 
# [this site](url = 'https://www.wunderground.com/weather/us/dc/20052') works okay.
# 
# Now, get the codes to work, with a list of 10 zip codes with the state abbreviation 
# ( so the list/tuple should looks like this: ('dc/20052' , 'ny/10001' , 'ca/90210' , ... ) ), 
# automatically pull the forecast temperature high and low for the day 
# at 6am at our Washington-DC time? You can pick any 10 zip codes you like in here. 
# The stock portfolio example in class is probably adaptable to 
# handle this task. 
# 
# At least have the codes tested on your computer, and have it ready 
# for deployment to pythonanhywhere.com. 
# 
# You do not need to deploy your codes on pythonanywhere.com. It is 
# an optional exercise if you are interested. Simply run your 
# codes on your computer for a few days, or 2-3 times a day to get 
# the temp info at different times to make sure it works. 
# That's all you need to submit, the working codes and a 
# sample data csv file. 
#  
# # 
# 
import datetime
import pandas as pd
import numpy as np
list = ["dc/20052","ny/10001","ca/90210","md/21035","ny/13126","md/21221","va/23185","pa/16601","in/46375","ct/06405","nc/28173"]
import requests
from bs4 import BeautifulSoup
def getTemp(stocksymbol):
  sourcemain = 'https://www.wunderground.com/weather/us/'
  url = sourcemain + stocksymbol
  thispage = requests.get(url)
  # soup = BeautifulSoup(thispage.content, 'lxml')
  # soup = BeautifulSoup(thispage.content, 'html.parser')
  soup = BeautifulSoup(thispage.content, 'html5lib')
  ans = soup.find('div',class_='current-temp').text.split()[0]

  return ans


#def getTemptatTime(): This would be the framwork for the function needed for schedukle.

x = np.zeros(len(list)) # This is my fucntion to create a list of temperatures, but I'm too lazy to make an acutual funciton
for i in range(len(list)):
    x[i] = getTemp(list[i])
df = pd.DataFrame(x,index=list,columns=[datetime.datetime.now().strftime("%m/%d,%H:%M")])
#%%
x = np.zeros(len(list)) #copied and apsted from above
for i in range(len(list)):
    x[i] = getTemp(list[i])
df.insert(np.shape(df)[1],datetime.datetime.now().strftime("%m/%d,%H:%M"),x) #Must wait 1 minute before running again otherwise, throwback error
# I could include a conditional statement to prevent this... but... lazy
# If the 6am 

#%%
df.to_csv("/Users/alexlange/Desktop/weather_times.csv")
# #%%
# import schedule
# import time
# schedule.every().day.at("06:00").do(getTemp(),'It is 01:00')

