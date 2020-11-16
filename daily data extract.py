#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 07:50:49 2020

@author: rajkumar
"""

from selenium import webdriver
from selenium.webdriver.support.ui import Select


options = webdriver.ChromeOptions()#we use the chrome options to change the default downloads path to a new path
options.add_argument("--start-maximized")
prefs = {"profile.default_content_settings.popups": 0,
             "download.default_directory": 
                        r"F:\downloads",#give a new download's path in order to download the files in that location
             "directory_upgrade": True}
options.add_experimental_option("prefs", prefs)#to select the download path of a new folder
driver=webdriver.Chrome('C:\Program Files (x86)\chromedriver.exe', options=options)#opening the chromedriver from its location

driver.get("https://agmarknet.gov.in/")#opening the website
driver.implicitly_wait(600)#we are using implicitly wait function to hault the program execution till the website loads



commodities = ['17','19','154','20','18','24','3','1']
#states =['WB','JR']
    
for commodity in commodities:
    cdty=Select(driver.find_element_by_id("ddlCommodity"))
    cdty.select_by_value(commodity)
   # cdty.select_by_value('17')
    state = driver.find_element_by_id('ddlState')
    state.send_keys("WB")
    start_date = driver.find_element_by_id('txtDate').clear()
    start_date = driver.find_element_by_id('txtDate')
    start_date.send_keys("25-Aug-2020")
    search = driver.find_element_by_id('btnGo')
    search.click()
    driver.implicitly_wait(2000)
    importtoexcel = driver.find_element_by_id('cphBody_ButtonExcel')
    #if not load recursion
    importtoexcel.click()
    home =driver.find_element_by_class_name('home')
    home.click()
    driver.implicitly_wait(2000)
driver.implicitly_wait(5000)
#======================================================================================================================

#merging the files
import pandas as pd
import glob
glb = glob.glob("F:\downloads")

a_dta = pd.DataFrame()

for g in glb:
    s_data = pd.read_html(g)
    a_dta = a_dta.append(s_data, ignore_index=True)

a_dta.to_csv(r'F:\converted\merged_data.csv')

#removing no data found records and filling State Name
ndf = a_dta[(a_dta['State Name'] == 'No Data Found')].index
a_dta.drop(ndf, inplace=True)
a_dta['State Name'].fillna("West Bengal", inplace=True)

#dropping unused columns
a_dta.columns
a_dta = pd.DataFrame(a_dta, columns = ['State Name', 'District Name', 'Market Name', 'Commodity', 'Variety', 'Grade', 'Price Date', 'Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)', 'Modal Price (Rs./Quintal)'])

#=================================================================================================================

#deleting the downloaded files
import os

xls_path = "F:\downloads"

for xlsfiles in glob.iglob(os.path.join(xls_path, '*.xls')):
    os.remove(xlsfiles)

#=================================================================================================================    
# Connecting to Database 
mrg_data = pd.read_csv("F:\converted\merged_data.csv")

from sqlalchemy import create_engine
connection = create_engine("mysql+pymysql://root:$he1by@localhost/agrmarknet")

# uploading data to the db
mrg_data.to_sql('agrmarknet', connection)

#=================================================================================================================    

#deleting merged csv file
csv_path = "F:\converted"

for csvfiles in glob.iglob(os.path.join(csv_path, '*.csv')):
    os.remove(csvfiles)

#=================================================================================================================    

# Creating a dataframe from MySQL_DB
agr = "select * from agrmarnet"
agr_data = pd.read_sql(agr, connection,  index_col = "index")
