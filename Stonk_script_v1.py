#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 03:42:22 2021

@author: alexlange
"""

import numpy as np
import pandas as pd
import webbrowser as wb
from selenium import webdriver
import time 
import seaborn as sns
#from selenium.webdriver.chrome.options import Options


#%%
# chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument('--disable-notifications')
# chrome_options.add_argument("--headless")
# driver = webdriver.Chrome(executable_path="/Users/alexlange/Downloads/chromedriver",options=chrome_options)
# driver.get("https://www.instagram.com/accounts/login/")
# #wb.open("https://finance.yahoo.com/quote/SNAP?p=SNAP")
#%%
def enable_download_in_headless_chrome(self, driver, download_dir):
    # add missing support for chrome "send_command"  to selenium webdriver
    driver.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')

    params = {'cmd': 'Page.setDownloadBehavior', 'params': {'behavior': 'allow', 'downloadPath': download_dir}}
    command_result = driver.execute("send_command", params)
#%% Download 1 year code 
list = ["GME", "AMC", "NOK"]
#%%
chrome_options = webdriver.ChromeOptions()
#chrome_options.add_argument('--disable-notifications')
chrome_options.add_argument("--headless")


for i in range(len(list)):
##########################################################################################################
# THIS LINE IS MACHINE DEPENDENT ON WHERE WEBDRIVER IS INSTALLED, BE SURE TO TAKE CARE
##########################################################################################################
    driver = webdriver.Chrome(executable_path='/usr/local/Caskroom/chromedriver/94.0.4606.61/chromedriver',options=chrome_options)  # mac OS
##########################################################################################################
    driver.get("https://finance.yahoo.com/quote/"+list[i]+"/history?p="+list[i])
    time.sleep(1)
    driver.find_element_by_css_selector('#Col1-1-HistoricalDataTable-Proxy > section > div.Pt\(15px\) > div.C\(\$tertiaryColor\).Mt\(20px\).Mb\(15px\) > span.Fl\(end\).Pos\(r\).T\(-6px\) > a').click()
#
    print(list[i])
# <a class="Fl(end) Mt(3px) Cur(p)" href="https://query1.finance.yahoo.com/v7/finance/download/GME?period1=1479254400&amp;period2=1637020800&amp;interval=1d&amp;events=history&amp;includeAdjustedClose=true" download="GME.csv"><svg class="Va(m)! Mend(5px) Stk($linkColor)! Fill($linkColor)! Cur(p)" width="15" height="15" viewBox="0 0 24 24" data-icon="download" style="fill: rgb(0, 129, 242); stroke: rgb(0, 129, 242); stroke-width: 0; vertical-align: bottom;"><path d="M21 20c.552 0 1 .448 1 1s-.448 1-1 1H3c-.552 0-1-.448-1-1s.448-1 1-1h18zM13 2v12.64l3.358-3.356c.375-.375.982-.375 1.357 0s.375.983 0 1.357L12 18l-5.715-5.36c-.375-.373-.375-.98 0-1.356.375-.375.983-.375 1.358 0L11 14.64V2h2z"></path></svg><span>Download</span></a>

#%% Work in progress combining all the data together. Will update
data1 = pd.read_csv("/Users/alexlange/Downloads/GME.csv")
data2 = pd.read_csv("/Users/alexlange/Downloads/AMC.csv")
#data = pddata1
data1a = pd.DataFrame(data1)
#data2a = pd.DataFrame({list[0]: data2})
data1['Ticker'] = str("GME")
data2['Ticker'] = str("AMC")
frames = [data1,data2]
data = pd.concat(frames)
data =data.reset_index(drop=True)
test = data.pivot(index="Date",columns="Ticker")
#%%
sns.scatterplot(data=data,x="Date",y="Close",hue="Ticker")