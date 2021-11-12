# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%% [markdown]
# # Web scraping
#
# Our goal here is to learn
# * Basic HTML structure
# * Basic CSS structure and selector rules
# * Basic webpage inspection tools from browsers
# * Use of BeautifulSoup or other web scraping libraries, plus parsers

#%%
import numpy as np 
# import pandas as pd
# import matplotlib.pyplot as plt
# plt.style.use('classic')

#%%
# We can use Python library "requests" to download the html 
# file (via a GET request to the web server). 
# In html terminology, the traditional way of posting forms are via 
# either the 'post' or 'get' methods. Collectively these are the standard 'requests'.
# As web design becomes more and more interactive, 'requests' are getting more sophisticated.
# For now, requests here are still the basic ones with 'get' and 'post'
import requests
# You can get a list of attributes and methods for the python requests object from w3school very handy
# https://www.w3schools.com/python/ref_requests_response.asp


#%%
myportfolio = ( 'MSFT', 'AAPL', 'GOOG' )
url = 'https://money.cnn.com/quote/quote.html?symb=MSFT' # we aim to loop thru our portfolio later when things are working
thispage = requests.get(url)
print(thispage)
# a response object, with status_code [200] means successful. It could be a blank page or error page, however...
print(thispage.status_code)
# a status code starting with a 2 generally indicates success, and a code starting with a 4 or a 5 indicates an error.
print("\nReady to continue.")

#%%
# To get the html body (and head) from the response object, we can use
print(thispage.content)
# The results typically should be like this
# b'<!DOCTYPE html>\n ...
# The starting 'b' character indicates it's in bytes, where as 
print("\nReady to continue.")

#%% 
# A cleaner way to extract the content is this:
print(thispage.text)
# will be in unicodes

# These are what you would see from "downloading" or "inspecting" a webpage in chrome/firefox/safari/etc.
# Try that.

print("\nReady to continue.")

#%%
# Next step is to use some kind of parsers to parse the HTML codes into standard tree-like or object-like structure (HTML DOM  Document-Object-Model)

# Useful to have basic HTML knowledge, the (XML) structure

# CSS (Cascade Style Sheet) is also an integral part of most HTML design these days.

# Most parsers can use CSS-style selectors, as well as Xpaths

#%%
# We will use the library beautifulSoup with the default parser lxml. 
# Another common alternative to beautifulSoup is the scrapy library.
# Need to install for the first time:
# !pip install bs4
# !pip3 install bs4
# One of the parser is lxml. We will use 'html5lib' mainly here. They are very similar and 
# you can't tell the difference most of the time. 
# If you need lxml, and your system does not have it, do: 
# !pip install lxml 
from bs4 import BeautifulSoup

#%%
# soup = BeautifulSoup(thispage.content, 'lxml')
# soup = BeautifulSoup(thispage.content, 'html.parser')
soup = BeautifulSoup(thispage.content, 'html5lib')
# beautifulSoup needs the contents given by the requests.get(url) object, then parse it.
print("\nReady to continue.")

#%%
print(soup.prettify())
print("\nReady to continue.")

#%% 
# #################### EXPLORATORY HERE ##################################
# What is the structure of soup?
soupkids = list(soup.children)
print(len(soupkids)) # length of 3 if html.parser is used, or 2 if lxml or html5.lib is used.
# soupkids = ['HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"', <html> <head> <title> ...
# print(soupkids)
print("\nReady to continue.")

#%%
# Use list comprehension to see what is in the list
[ type(item) for item in soupkids ]
print("\nReady to continue.")

#%%
print(soupkids[0])   # HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"
print("\nReady to continue.")

#%%
print(soupkids[1]) # the actual html document that we really care about
# At this point, soupkids[1] is much like the request.get(url).text 
# which gives the full html text of the page. 
# Below is to use the power of beautifulSoup to extract and find info from the html/xml content.
print("\nReady to continue.")

#%%
# If we want to extract a certain piece inside the body of the html doc, we can do
htmlbody = list(soupkids[1].children)[2]   
# soupkids[1].children is a list_iterator 
# we need to create a list out of it. Turns out the 0-th is the html head, 1-st is blank, 2-nd is the html body 
print(htmlbody)
print("\nReady to continue.")

# ####################  END EXPLORATION ##################################
# We do not need to pull the htmlbody separately. Beautiful soup will handle that for us. 
# Below are the steps.

#%%[markdown]
# First load up the page on chrome (or firefox)  
# <https://money.cnn.com/quote/quote.html?symb=MSFT>  
# Let us try to pull the stock price value from the site  
# right click on what you what, and select "Inspect" to open the developer console. 
# With some understandings of html DOM (document-object-model, model is synonymous with data, ever since, forever)
# we can navigate the DOM object to find what we need.
# We can use Tag elements by its name, its ID, 
# or use CSS-style selector (usually my preference) 
# or use xpath.

#%%
# know what you want to find. Use .find() or .find_all()
foundlast = soup.find('td', class_='wsod_last') # by tag name and class values
print(foundlast, '\n')
print(foundlast.text) # almost got it

#%%
# Try
print(list(foundlast.children), '\n') # Here you are, the quote, in the first element of the list
print(list(foundlast.children)[0].text) # the stock quote 


#%%
# Also try to use CSS selectors
# Use dot . for className, use # for id
# cnnBody > div.cnnBody_Left.wsodContent > div.mod-quoteinfo > div:nth-child(2) > table > tbody > tr > td.wsod_last > span
# selectlast = soup.select('tr td.wsod_last span')  # CSS-style selector  # under <tr> any levels of descendants lower, find <td>
selectlast = soup.select('tr > td.wsod_last > span')  # CSS-style selector  # under <tr>, immediate children, find <td> 
# the select function will return a list of nodes. Could be an empty list.
selectlast[0].text
# Or if we know for sure there is only one, or just want the first one, we can do
# selectlast = soup.select_one('tr > td.wsod_last > span') # return a single node instead of a list
# selectlast.text


#%%
# myportfolio = ( 'MSFT', 'AAPL', 'GOOG' )

# So with all these hard work, we can now streamline all these into a combined function call
import requests
from bs4 import BeautifulSoup
def getStockQuote(stocksymbol):
  sourcemain = 'https://money.cnn.com/quote/quote.html?symb='
  url = sourcemain + stocksymbol.upper()
  thispage = requests.get(url)
  # soup = BeautifulSoup(thispage.content, 'lxml')
  # soup = BeautifulSoup(thispage.content, 'html.parser')
  soup = BeautifulSoup(thispage.content, 'html5lib')
  selectlast = soup.select_one('tr > td.wsod_last > span') # return a single node instead of a list
  return float(selectlast.text)

print("\nReady to continue.")

#%%
# Testing...
# print(getStockQuote('aapl')) # works
print(getStockQuote('silly')) # error
# try to make your codes fool-proof

#%%
import requests
from bs4 import BeautifulSoup

def getStockQuote(stocksymbol):
  sourcemain = 'https://money.cnn.com/quote/quote.html?symb='
  url = sourcemain + stocksymbol.upper()
  thispage = requests.get(url)  # thispage.status_code is still 200 whether the stock symbol exists or not. 
  # soup = BeautifulSoup(thispage.content, 'lxml')
  # soup = BeautifulSoup(thispage.content, 'html.parser')
  soup = BeautifulSoup(thispage.content, 'html5lib')
  selectlast = soup.select('tr > td.wsod_last > span') # return a list
  return float(selectlast[0].text.replace(',','') ) if (len(selectlast)==1) else np.nan # or anything you can flag easily. I try to keep the return value numeric. Other option would be -1

# Now it works for 'silly'

print("\nReady to continue.")

#%%
# Testing...
print(getStockQuote('aapl')) # works
print(getStockQuote('silly')) # fool-proof


# %%
for symbol in myportfolio:
  print(getStockQuote(symbol))

print("\nReady to continue.")

#%%
