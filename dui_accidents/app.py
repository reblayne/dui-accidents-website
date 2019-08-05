#Load the packages
from flask import Flask, render_template, request, redirect, jsonify

import pdb

import dill
import shapefile

import numpy as np
import pandas as pd
import math
import datetime
from sklearn.linear_model import LogisticRegression

from bokeh.models import HoverTool, ColumnDataSource, LogColorMapper
from bokeh.palettes import Viridis6
from bokeh.plotting import figure, show, output_notebook, output_file, ColumnDataSource
from bokeh.layouts import gridplot, column, widgetbox, row, column
from bokeh.embed import components
from bokeh.io import show, curdoc
from bokeh.models.widgets import TextInput
# from bokeh.palettes import Blues8 as palette   # this is the blue color palette  # https://bokeh.pydata.org/en/latest/docs/reference/palettes.html#bokeh-palettes
# from bokeh.sampledata.us_counties import data as counties

import requests
import simplejson as json

import os
import requests
import itertools

#Connect the app
app = Flask(__name__)

# dictionary that maps month to month number
month_code_dict = {
    "January" : 1,
    "February" : 2,
    "March" : 3,
    "April" : 4,
    "May" : 5,
    "June" : 6,
    "July" : 7,
    "August" : 8,
    "September" : 9,
    "October" : 10,
    "November" : 11,
    "December" : 12,
}

# dictionary that maps weekday number to weekday name
weekday_code_dict = {
    "Monday" : 0,
    "Tuesday" : 1,
    "Wednesday" : 2,
    "Thursday" : 3,
    "Friday" : 4,
    "Saturday" : 5,
    "Sunday" : 6,
}

# dictionary that maps date to holiday
holiday_dict = {
    datetime.datetime(2007, 7, 4, 0, 0) : '4th of July',
    datetime.datetime(2008, 7, 4, 0, 0) : '4th of July',
    datetime.datetime(2009, 7, 4, 0, 0) : '4th of July',
    datetime.datetime(2010, 7, 4, 0, 0) : '4th of July',
    datetime.datetime(2011, 7, 4, 0, 0) : '4th of July',
    datetime.datetime(2012, 7, 4, 0, 0) : '4th of July',
    datetime.datetime(2013, 7, 4, 0, 0) : '4th of July',
    datetime.datetime(2014, 7, 4, 0, 0) : '4th of July',
    datetime.datetime(2015, 7, 4, 0, 0) : '4th of July',
    datetime.datetime(2016, 7, 4, 0, 0) : '4th of July',
    datetime.datetime(2017, 7, 4, 0, 0) : '4th of July',
    datetime.datetime(2018, 7, 4, 0, 0) : '4th of July',
    datetime.datetime(2019, 7, 4, 0, 0) : '4th of July',
    datetime.datetime(2020, 7, 4, 0, 0) : '4th of July',
    datetime.datetime(2007, 12, 25, 0, 0) : 'Christmas',
    datetime.datetime(2008, 12, 25, 0, 0) : 'Christmas',
    datetime.datetime(2009, 12, 25, 0, 0) : 'Christmas',
    datetime.datetime(2010, 12, 25, 0, 0) : 'Christmas',
    datetime.datetime(2011, 12, 25, 0, 0) : 'Christmas',
    datetime.datetime(2012, 12, 25, 0, 0) : 'Christmas',
    datetime.datetime(2013, 12, 25, 0, 0) : 'Christmas',
    datetime.datetime(2014, 12, 25, 0, 0) : 'Christmas',
    datetime.datetime(2015, 12, 25, 0, 0) : 'Christmas',
    datetime.datetime(2016, 12, 25, 0, 0) : 'Christmas',
    datetime.datetime(2017, 12, 25, 0, 0) : 'Christmas',
    datetime.datetime(2018, 12, 25, 0, 0) : 'Christmas',
    datetime.datetime(2019, 12, 25, 0, 0) : 'Christmas',
    datetime.datetime(2020, 12, 25, 0, 0) : 'Christmas',
    datetime.datetime(2012, 12, 25, 0, 0) : 'Christmas',
    datetime.datetime(2007, 10, 8, 0, 0) : 'Columbus Day',
    datetime.datetime(2008, 10, 13, 0, 0) : 'Columbus Day',
    datetime.datetime(2009, 10, 12, 0, 0) : 'Columbus Day',
    datetime.datetime(2010, 10, 11, 0, 0) : 'Columbus Day',
    datetime.datetime(2011, 10, 10, 0, 0) : 'Columbus Day',
    datetime.datetime(2012, 10, 8, 0, 0) : 'Columbus Day',
    datetime.datetime(2013, 10, 14, 0, 0) : 'Columbus Day',
    datetime.datetime(2014, 10, 13, 0, 0) : 'Columbus Day',
    datetime.datetime(2015, 10, 12, 0, 0) : 'Columbus Day',
    datetime.datetime(2016, 10, 10, 0, 0) : 'Columbus Day',
    datetime.datetime(2017, 10, 9, 0, 0) : 'Columbus Day',
    datetime.datetime(2018, 10, 8, 0, 0) : 'Columbus Day',
    datetime.datetime(2019, 10, 14, 0, 0) : 'Columbus Day',
    datetime.datetime(2020, 10, 12, 0, 0) : 'Columbus Day',
    datetime.datetime(2007, 9, 3, 0, 0) : 'Labor Day',
    datetime.datetime(2008, 9, 1, 0, 0) : 'Labor Day',
    datetime.datetime(2009, 9, 7, 0, 0) : 'Labor Day',
    datetime.datetime(2010, 9, 6, 0, 0) : 'Labor Day',
    datetime.datetime(2011, 9, 5, 0, 0) : 'Labor Day',
    datetime.datetime(2012, 9, 3, 0, 0) : 'Labor Day',
    datetime.datetime(2013, 9, 2, 0, 0) : 'Labor Day',
    datetime.datetime(2014, 9, 1, 0, 0) : 'Labor Day',
    datetime.datetime(2015, 9, 7, 0, 0) : 'Labor Day',
    datetime.datetime(2016, 9, 5, 0, 0) : 'Labor Day',
    datetime.datetime(2017, 9, 4, 0, 0) : 'Labor Day',
    datetime.datetime(2018, 9, 3, 0, 0) : 'Columbus Day',
    datetime.datetime(2019, 9, 2, 0, 0) : 'Labor Day',
    datetime.datetime(2020, 9, 7, 0, 0) : 'Labor Day',
    datetime.datetime(2007, 1, 15, 0, 0) : "Martin Luther King's Birthday",
    datetime.datetime(2008, 1, 21, 0, 0) : "Martin Luther King's Birthday",
    datetime.datetime(2009, 1, 19, 0, 0) : "Martin Luther King's Birthday",
    datetime.datetime(2010, 1, 18, 0, 0) : "Martin Luther King's Birthday",
    datetime.datetime(2011, 1, 17, 0, 0) : "Martin Luther King's Birthday",
    datetime.datetime(2012, 1, 16, 0, 0) : "Martin Luther King's Birthday",
    datetime.datetime(2013, 1, 21, 0, 0) : "Martin Luther King's Birthday",
    datetime.datetime(2014, 1, 20, 0, 0) : "Martin Luther King's Birthday",
    datetime.datetime(2015, 1, 19, 0, 0) : "Martin Luther King's Birthday",
    datetime.datetime(2016, 1, 18, 0, 0) : "Martin Luther King's Birthday",
    datetime.datetime(2017, 1, 16, 0, 0) : "Martin Luther King's Birthday",
    datetime.datetime(2018, 1, 15, 0, 0) : "Martin Luther King's Birthday",
    datetime.datetime(2019, 1, 21, 0, 0) : "Martin Luther King's Birthday",
    datetime.datetime(2020, 1, 20, 0, 0) : "Martin Luther King's Birthday",
    datetime.datetime(2007, 5, 28, 0, 0) : 'Memorial Day',
    datetime.datetime(2008, 5, 26, 0, 0) : 'Memorial Day',
    datetime.datetime(2009, 5, 25, 0, 0) : 'Memorial Day',
    datetime.datetime(2010, 5, 31, 0, 0) : 'Memorial Day',
    datetime.datetime(2011, 5, 20, 0, 0) : 'Memorial Day',
    datetime.datetime(2012, 5, 28, 0, 0) : 'Memorial Day',
    datetime.datetime(2013, 5, 27, 0, 0) : 'Memorial Day',
    datetime.datetime(2014, 5, 26, 0, 0) : 'Memorial Day',
    datetime.datetime(2015, 5, 25, 0, 0) : 'Memorial Day',
    datetime.datetime(2016, 5, 30, 0, 0) : 'Memorial Day',
    datetime.datetime(2017, 5, 29, 0, 0) : 'Memorial Day',
    datetime.datetime(2018, 5, 28, 0, 0) : 'Memorial Day',
    datetime.datetime(2019, 5, 27, 0, 0) : 'Memorial Day',
    datetime.datetime(2020, 5, 25, 0, 0) : 'Memorial Day',
    datetime.datetime(2007, 1, 1, 0, 0) : "New Year's Day",
    datetime.datetime(2008, 1, 1, 0, 0) : "New Year's Day",
    datetime.datetime(2009, 1, 1, 0, 0) : "New Year's Day",
    datetime.datetime(2010, 1, 1, 0, 0) : "New Year's Day",
    datetime.datetime(2011, 1, 1, 0, 0) : "New Year's Day",
    datetime.datetime(2012, 1, 1, 0, 0) : "New Year's Day",
    datetime.datetime(2013, 1, 1, 0, 0) : "New Year's Day",
    datetime.datetime(2014, 1, 1, 0, 0) : "New Year's Day",
    datetime.datetime(2015, 1, 1, 0, 0) : "New Year's Day",
    datetime.datetime(2016, 1, 1, 0, 0) : "New Year's Day",
    datetime.datetime(2017, 1, 1, 0, 0) : "New Year's Day",
    datetime.datetime(2018, 1, 1, 0, 0) : "New Year's Day",
    datetime.datetime(2019, 1, 1, 0, 0) : "New Year's Day",
    datetime.datetime(2020, 1, 1, 0, 0) : "New Year's Day",
    datetime.datetime(2007, 12, 31, 0, 0) : "New Year's Eve",
    datetime.datetime(2008, 12, 31, 0, 0) : "New Year's Eve",
    datetime.datetime(2009, 12, 31, 0, 0) : "New Year's Eve",
    datetime.datetime(2010, 12, 31, 0, 0) : "New Year's Eve",
    datetime.datetime(2011, 12, 31, 0, 0) : "New Year's Eve",
    datetime.datetime(2012, 12, 31, 0, 0) : "New Year's Eve",
    datetime.datetime(2013, 12, 31, 0, 0) : "New Year's Eve",
    datetime.datetime(2014, 12, 31, 0, 0) : "New Year's Eve",
    datetime.datetime(2015, 12, 31, 0, 0) : "New Year's Eve",
    datetime.datetime(2016, 12, 31, 0, 0) : "New Year's Eve",
    datetime.datetime(2017, 12, 31, 0, 0) : "New Year's Eve",
    datetime.datetime(2018, 12, 31, 0, 0) : "New Year's Eve",
    datetime.datetime(2019, 12, 31, 0, 0) : "New Year's Eve",
    datetime.datetime(2020, 12, 31, 0, 0) : "New Year's Eve",
    datetime.datetime(2007, 2, 19, 0, 0) : "President's Day",
    datetime.datetime(2008, 2, 18, 0, 0) : "President's Day",
    datetime.datetime(2009, 2, 16, 0, 0) : "President's Day",
    datetime.datetime(2010, 2, 15, 0, 0) : "President's Day",
    datetime.datetime(2011, 2, 21, 0, 0) : "President's Day",
    datetime.datetime(2012, 2, 20, 0, 0) : "President's Day",
    datetime.datetime(2013, 2, 18, 0, 0) : "President's Day",
    datetime.datetime(2014, 2, 17, 0, 0) : "President's Day",
    datetime.datetime(2015, 2, 16, 0, 0) : "President's Day",
    datetime.datetime(2016, 2, 15, 0, 0) : "President's Day",
    datetime.datetime(2017, 2, 20, 0, 0) : "President's Day",
    datetime.datetime(2018, 2, 19, 0, 0) : "President's Day",
    datetime.datetime(2019, 2, 18, 0, 0) : "President's Day",
    datetime.datetime(2020, 2, 17, 0, 0) : "President's Day",
    datetime.datetime(2007, 11, 22, 0, 0) : "Thanksgiving",
    datetime.datetime(2008, 11, 27, 0, 0) : "Thanksgiving",
    datetime.datetime(2009, 11, 26, 0, 0) : "Thanksgiving",
    datetime.datetime(2010, 11, 25, 0, 0) : "Thanksgiving",
    datetime.datetime(2011, 11, 24, 0, 0) : "Thanksgiving",
    datetime.datetime(2012, 11, 22, 0, 0) : "Thanksgiving",
    datetime.datetime(2013, 11, 28, 0, 0) : "Thanksgiving",
    datetime.datetime(2014, 11, 27, 0, 0) : "Thanksgiving",
    datetime.datetime(2015, 11, 26, 0, 0) : "Thanksgiving",
    datetime.datetime(2016, 11, 24, 0, 0) : "Thanksgiving",
    datetime.datetime(2017, 11, 23, 0, 0) : "Thanksgiving",
    datetime.datetime(2018, 11, 22, 0, 0) : "Thanksgiving",
    datetime.datetime(2019, 11, 28, 0, 0) : "Thanksgiving",
    datetime.datetime(2020, 11, 26, 0, 0) : "Thanksgiving"
}


# dictionary that maps weekday number to weekday name
weekday_dict = {
    0 : "Monday",
    1 : "Tuesday",
    2 : "Wednesday",
    3 : "Thursday",
    4 : "Friday",
    5 : "Saturday",
    6 : "Sunday",
}

def get_map_output():
    map_high_res = "cb_2015_us_county_500k"
    path = "/Users/rebeccalayne/Library/Mobile Documents/com~apple~CloudDocs/Documents/TDI Fellowship/capstone_project/dui_accidents/cb_2015_us_county_500k/"
    map_output = get_map_data(map_high_res, path)
    dill.dump(map_output, open('map_output.pkd', 'wb'))

def get_map_data(shape_data_file, local_file_path):
    url = "http://www2.census.gov/geo/tiger/GENZ2015/shp/" + \
      shape_data_file + ".zip"
    zfile = local_file_path + shape_data_file + ".zip"
    sfile = local_file_path + shape_data_file + ".shp"
    dfile = local_file_path + shape_data_file + ".dbf"
    if not os.path.exists(zfile):
        print("Getting file: ", url)
        response = requests.get(url)
        with open(zfile, "wb") as code:
            code.write(response.content)

    if not os.path.exists(sfile):
        uz_cmd = 'unzip ' + zfile + " -d " + local_file_path
        print("Executing command: " + uz_cmd)
        os.system(uz_cmd)

    shp = open(sfile, "rb")
    dbf = open(dfile, "rb")
    sf = shapefile.Reader(shp=shp, dbf=dbf)

    lats = []
    lons = []
    ct_name = []
    st_id = []
    for shprec in sf.shapeRecords():
        st_id.append(int(shprec.record[0]))
        ct_name.append(shprec.record[5])
        lat, lon = map(list, zip(*shprec.shape.points))
        indices = shprec.shape.parts.tolist()
        lat = [lat[i:j] + [float('NaN')] for i, j in zip(indices, indices[1:]+[None])]
        lon = [lon[i:j] + [float('NaN')] for i, j in zip(indices, indices[1:]+[None])]
        lat = list(itertools.chain.from_iterable(lat))
        lon = list(itertools.chain.from_iterable(lon))
        lats.append(lat)
        lons.append(lon)

    map_data = pd.DataFrame({'x': lats, 'y': lons, 'state': st_id, 'county_name': ct_name})
    return map_data


def parse_map_data(map_output):
    map_output = map_output[map_output["state"] == 6]   # let's only keep California --> state number 6
    county_names = map_output['county_name']

    df = dill.load(open('df_for_predictive_model.pkd', 'rb'))
    first_names = sorted(county_names)
    full_names = sorted(list(set(df['county_name'])))

    # map from county first name to county full_name
    county_dict = dict(zip(first_names, full_names))
    county_names = [county_dict[i] for i in county_names]   # convert counties to full names

    lat = []
    for lat_list in map_output['x']:
        lat.append(lat_list[:-1])

    long = []
    for long_list in map_output['y']:
        long.append(long_list[:-1])

    return county_names, lat, long


def get_state_outline():
    # from bokeh.sampledata.us_states import data as states    # state info for plotting border lines
    states = dill.load(open('states.pkd', 'rb'))
    # outline for CA
    state_xs = [states['CA']["lons"]]
    state_ys = [states['CA']["lats"]]
    return state_xs, state_ys

# get subset of df based on user chosen date
def get_subset(positive_df, month, day, hour):
    from datetime import datetime
    month_code = month_code_dict[month]
    year = datetime.today().year
    # year = 2019
    date = datetime(year, month_code, day, 0, 0)    # I should change 2018 to curr year
    if date in holiday_dict:   # check if day is holiday first and return instances on past holiday
        holiday = holiday_dict[date]
        subset = positive_df[positive_df.holiday == holiday]
        return subset
    else:    # show past accidents on that month and day of week (specify not holiday)
        subset = positive_df[positive_df.holiday == 'Not Holiday Related']
        weekday_code = date.weekday()
        weekday = weekday_dict[weekday_code]
        subset = subset[subset.dayofweek == weekday]
        subset = subset[subset.month == month]
        return subset

# @param df
# @return tuple of info about accidents in df including total #, percent male, avg age, percent white
def summarize_data(df):
    num_instances = len(df)
    num_male = len(df[df.sex == 'Male'])
    percent_male = num_male/df['sex'].count()   # divide by number of non-nan values in column
    avg_age = df['age'].mean()
    subset = df[df.race != 'Not a Fatality (not applicable)']
    num_white = len(subset[subset.race == 'White'])
    percent_white = num_white/subset['race'].count()
    info = (num_instances, percent_male, avg_age, percent_white)
    return info


# spit out dict of counties --> descriptive values
def summarize_data_by_county(county_names, df, county_prob_dict):
    county_info_dict = {}
    for county in county_names:
        county_subset = df[df.county_name == county]
        info = summarize_data(county_subset)
        prob = county_prob_dict[county]     # get predicted probability from county
        info = info + (prob,)   # add prob to tuple
        county_info_dict[county] = info
    return county_info_dict


def format_county_info(county_names, county_info_dict, county_prob_dict):
    instances = []
    percent_male = []
    avg_age = []
    percent_white = []
    predicted_prob = []
    for county in county_names:
        info = county_info_dict[county]
        instances.append(info[0])
        percent_male.append(info[1])
        avg_age.append(info[2])
        percent_white.append(info[3])
        predicted_prob.append(county_prob_dict[county])
    instances = pd.Series(instances)
    percent_male = [round(100*i, 1) for i in percent_male]
    percent_male = pd.Series(percent_male)
    avg_age = [round(i,1) for i in avg_age]
    avg_age = pd.Series(avg_age)
    percent_white = [round(100*i, 1) for i in percent_white]
    percent_white = pd.Series(percent_white)
    predicted_prob = pd.Series(predicted_prob)
    return (instances, percent_male, avg_age, percent_white, predicted_prob)

# summarize accidents that actually happened to suplement the map for a particular date
# I want to be able to select a month, hour, day (day of week) and see past accidents in each county
# return a list of the name of each county with value = # instance, % men
def get_county_summaries(month, day, hour, county_names, county_prob_dict):
    # positive_df = pd.read_csv('Fatal Accidents with County Level Info (Condensed).csv')
    # positive_df = dill.load(open('positive_df.pkd', 'rb'))
    # positive_df = positive_df[positive_df.alcohol_involved == 1]
    # positive_df = positive_df[positive_df.State == 'CA']
    positive_df = dill.load(open('positive_df_for_CA.pkd', 'rb'))
    subset = get_subset(positive_df, month, day, hour)     # user chooses day
    county_info_dict = summarize_data_by_county(county_names, subset, county_prob_dict)
    instances, percent_male, avg_age, percent_white, predicted_prob = format_county_info(county_names, county_info_dict, county_prob_dict)
    return instances, percent_male, avg_age, percent_white, predicted_prob


#functions below work to take in user-specified day and return the probability
# of an accident happening in each county on that day
def get_county_level_info(year, county_info, county_names):
    df = county_info[county_info.Year == year]
    df = df[['Area name', 'population_estimate', 'unemployment_rate',
            'Rural_urban_continuum_code_2013', 'Percent_no_college_degree']]
    df = df.iloc[1:]   # drop top row
    return df

# generate year, hour, month, dayofweek, holiday col
def set_user_input(year, month, day, hour, county_df):
    county_df['year'] = year
    county_df['month'] = month
    county_df['day'] = day
    county_df['hour'] = float(hour)

    month_code = month_code_dict[month]
    county_df['month_code'] = month_code

    from datetime import datetime
    date = datetime(year, month_code, day, 0, 0)

    # get weekday
    weekday_code = date.weekday()
    weekday = weekday_dict[weekday_code]
    county_df['dayofweek'] = weekday
    # add day of week to dataframe
    dayofweek_code = weekday_code_dict
    county_df['dayofweek_code'] = weekday_code_dict[weekday]

    # check holiday
    if date in holiday_dict:   # check if day is holiday first and return instances on past holiday
        holiday = holiday_dict[date]
        county_df['holiday'] = holiday
    else:
        county_df['holiday'] = 'Not Holiday Related'

    return(county_df)

def preprocess_user_df(df):
    # do all the stuff I did to og df before running model
    df.rename(columns={'Area name': 'county_name'}, inplace=True)

    # ordinal cyclic variables
    df["xhour"] = np.sin(2*math.pi*df["hour"]/24)
    df["yhour"] = np.cos(2*math.pi*df["hour"]/24)

    df["xmonth"] = np.sin(2*math.pi*df["month_code"]/12)
    df["ymonth"] = np.cos(2*math.pi*df["month_code"]/12)

    df["xdayofweek"] = np.sin(2*math.pi*df["dayofweek_code"]/7)
    df["ydayofweek"] = np.cos(2*math.pi*df["dayofweek_code"]/7)

    # year as ordered factor variable
    df['year'] = df['year'].apply(str)
    df['year'] = pd.Categorical(df['year'], categories = ["2007","2008","2009","2010","2011","2012","2013",
                                      "2014","2015","2016","2017","2018","2019","2020"], ordered = True)

    df['dayofweek'] = pd.Categorical(df['dayofweek'], categories = ["Monday", "Tuesday", "Wednesday",
                                                                   "Thursday", "Friday", "Saturday",
                                                                    "Sunday"], ordered = True)
    df['month'] = pd.Categorical(df['month'], categories = ["January", "February", "March", "April", "May",
                                                            "June", "July", "August", "September", "October",
                                                            "November", "December"], ordered = True)

    # convert holiday, Rural_urban_continuum_code_2013, fatal_DUI_accident to factor variables
    df['holiday'] = df['holiday'].astype('category')
    df['Rural_urban_continuum_code_2013'] = df['Rural_urban_continuum_code_2013'].astype('category')

    return df

# create dummy variables for categorical variables: year, holiday, Rural_urban_continuum_code_2013
# and for month and dayofweek
def create_dummy_vars(df, cat_vars):
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(df[var], prefix=var)
        data1=df.join(cat_list)
        df=data1
    data_vars=df.columns.values.tolist()
    to_keep=[i for i in data_vars if i not in cat_vars]

    data_final = df[to_keep]
    return (df, data_final)

def add_dummy_vars(df, cat_vars):
    df, data_final = create_dummy_vars(df, cat_vars)
    holiday_list = ['4th of July', 'Christmas', 'Columbus Day', 'Labor Day', "Martin Luther King's Birthday",
                   'Memorial Day', "New Year's Day", "New Year's Eve", 'Not Holiday Related', "President's Day",
                   'Thanksgiving', '.']
    var = 'holiday'
    curr_holiday = df['holiday'].iloc[0]
    for holiday in holiday_list:
        if (holiday != curr_holiday):
            var_name = var + '_' + holiday
            df[var_name] = 0
    return df


# wrap up all functions to get quick answer on probabilities:
def get_county_probs(year, month, day, hour, county_info, county_names, county_logreg, time_logreg):
    county_cols=['population_estimate', 'unemployment_rate', 'Percent_no_college_degree',
       'Rural_urban_continuum_code_2013_1.0',
       'Rural_urban_continuum_code_2013_2.0',
       'Rural_urban_continuum_code_2013_3.0',
       'Rural_urban_continuum_code_2013_4.0',
       'Rural_urban_continuum_code_2013_5.0',
       'Rural_urban_continuum_code_2013_6.0',
       'Rural_urban_continuum_code_2013_7.0',
       'Rural_urban_continuum_code_2013_8.0']

    time_cols=['year_2007', 'year_2008', 'year_2009',
           'year_2010', 'year_2011', 'year_2012', 'year_2013', 'year_2014',
           'year_2015', 'year_2016', 'year_2017', 'year_2018', 'year_2019',
           'year_2020', 'holiday_.', 'holiday_4th of July',
           'holiday_Christmas', 'holiday_Columbus Day', 'holiday_Labor Day',
           "holiday_Martin Luther King's Birthday", 'holiday_Memorial Day',
           "holiday_New Year's Day", "holiday_New Year's Eve",
           'holiday_Not Holiday Related', "holiday_President's Day",
           'holiday_Thanksgiving', 'xmonth',
           'ymonth', 'xdayofweek', 'ydayofweek']

    county_names = sorted(list(set(county_names)))  # sort county names
    county_df = get_county_level_info(year, county_info, county_names)
    user_df = set_user_input(year, month, day, hour, county_df)
    user_df = preprocess_user_df(user_df)
    cat_vars=['year','holiday','Rural_urban_continuum_code_2013', 'month', 'dayofweek']
    user_df = add_dummy_vars(user_df, cat_vars)
    X = user_df[county_cols]
    time_X = user_df[time_cols]

    probabilities = county_logreg.predict_proba(X)[:,1]     # predicted probabilities based on model
    time_prob = time_logreg.predict_proba(time_X)[:,1][0]
    county_prob_dict = {}
    for i in range(len(county_names)):
        county_prob_dict[county_names[i]] = probabilities[i]*time_prob*100.0     # combine both models by weighting pred by time prob

    return county_prob_dict






def get_plot(month, day, hour):
    map_output = dill.load(open('map_output.pkd', 'rb'))
    county_names, lat, long = parse_map_data(map_output)
    state_xs, state_ys = get_state_outline()

    county_info = dill.load(open('county_info.pkd', 'rb'))
    county_info = county_info[county_info.State == 'CA']

    county_logreg = dill.load(open('county_logreg.pkd', 'rb'))
    time_logreg = dill.load(open('time_logreg.pkd', 'rb'))

    county_prob_dict = get_county_probs(2018, month, day, hour, county_info, county_names, county_logreg, time_logreg)

    # county_prob_dict = dill.load(open('county_prob_dict.pkd', 'rb'))
    instances, percent_male, avg_age, percent_white, predicted_prob = get_county_summaries(month, day, hour, county_names, county_prob_dict)

    palette = ['#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#ebf3fa']  # blues
    palette.reverse()

    color_mapper = LogColorMapper(palette=palette, low=0.1, high=25)

    data=dict(
        x=lat,
        y=long,
        name=county_names,
        num=instances,
        male_rate=percent_male,
        age=avg_age,
        white_rate=percent_white,
        prob = predicted_prob,
    #     rate=county_rates,
    )

    TOOLS = "pan,wheel_zoom,reset,hover,save"

    # output_notebook()  # I guess this is the line I need ...

    p = figure(
        title="Californa by County", tools=TOOLS,
        x_range = (-125, -113), y_range = (32, 43),  # zoom in on US
        x_axis_location=None, y_axis_location=None,
        plot_width=480, plot_height=560,
        tooltips=[
            ("County Name", "@name"), ("Predicted Probability", "@prob{1.1}%"), ("Past Instances", "@num"),
            ("Percent Male", "@male_rate{1.1}%"),
            ("Average Age", "@age{1.1}"), ("Percent White", "@white_rate{1.1}%"), ("(Long, Lat)", "($x, $y)")
        ])

    p.grid.grid_line_color = None
    p.hover.point_policy = "follow_mouse"

    # plot state outlines
    p.patches(state_xs, state_ys, fill_alpha=0.0,
              line_color="#000000", line_width=2, line_alpha=0.3)

    p.patches('x', 'y', source=data,
              fill_color={'field': 'prob', 'transform': color_mapper},    # modify how we fill color
              fill_alpha=.9, line_color="white", line_width=0.5)

    return p



@app.route('/', methods=['POST'])
def my_form_post():
    curr_date = request.form['text']    # input from user

    if curr_date == None:
        curr_date = "January 1"
    else:
        date_info = curr_date.split(' ')
        month = date_info[0]
        day = int(date_info[1])


    p = get_plot(month, day, 5)
    script, div = components(p)

    #Render the page
    return render_template('index.html', script=script, div=div, curr_date=curr_date)

@app.route('/')
def homepage():
    print("reading file")
    # df = pd.read_csv('df_for_predictive_model.csv')
    # print("read file")
    # print("dumping data")
    # dill.dump(df, open('df_for_predictive_model.pkd', 'wb'))

    # df = dill.load(open('df_for_predictive_model.pkd', 'rb'))
    # get_map_output()
    # map_output = dill.load(open('map_output.pkd', 'rb'))
    # positive_df = dill.load(open('positive_df.pkd', 'rb'))
    # get_plot22()





    print("loaded files")

    # pdb.set_trace()

    # stock = 'GOOG'  # set the default plot to GOOG
    # subset = get_dataset(stock)

    #Setup plot
    p = get_plot('January', 1, 5)

    # inputs, plot = get_plot22('December', 25, 5)
    # curdoc().add_root(row(inputs, plot, width=800))
    # curdoc().title = "Sliders"

    # p = get_plot(subset)
    script, div = components(p)

    #Render the page
    return render_template('index.html', script=script, div=div, curr_date="January 1")


@app.route("/templates/index.html")
def return_home():
    return redirect("http://0.0.0.0:5000/", code=302)

@app.route('/index.html')
def return_home1():
    return redirect("http://0.0.0.0:5000/", code=302)

@app.route("/templates/post.html")
def data_page():
    return render_template('post.html')

@app.route("/post.html")
def data_page1():
    return render_template('post.html')

@app.route("/templates/contact.html")
def model_page():
    return render_template('contact.html')

@app.route("/contact.html")
def model_page1():
    return render_template('contact.html')

@app.route("/templates/about.html")
def about_page():
    return render_template('about.html')

@app.route("/about.html")
def about_page1():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
    # app.run()





# Read csv once: df = pd.read_csv('df_for_predictive_model.csv')
# Then dill.dump it once: dill.dump(df, open('df_for_predictive_model.pkd', 'wb'))
# Now I can always dill.load it: df = dill.load(open('df_for_predictive_model.pkd', 'rb'))





#
# # Simple app.py that works!! :)
#
# from flask import Flask, render_template, request, redirect
#
# app = Flask(__name__)
#
# @app.route('/')
# def index():
#     return "OK!"
#   # return render_template('index.html')
#
# def hello_world():
#     return 'Hello World!'
#
# @app.route('/about')
# def about():
#   return render_template('about.html')
#
# if __name__ == '__main__':
#   app.run(port=33507)


 # To run: python3 app.py
 # Then type into browser: localhost:33507
