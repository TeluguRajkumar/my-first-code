#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 07:43:32 2020

@author: rajkumar
"""
import pymysql
import pandas as pd
from sqlalchemy import create_engine
df=pd.read_excel("C:\\Users\\Rajkumar\\Desktop\\Live project\\Kol dataset\\west bengal.xlsx")
con=create_engine("mysql+pymysql://root:toor@localhost/agridb")
df.to_sql("mydb",con)

cmd = "select * from mydb"

Df = pd.read_sql(cmd, con,  index_col = "index")

