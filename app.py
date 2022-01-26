#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings('ignore')

from flask import Flask

import pandas as pd
import glob
import os
from sklearn.neighbors import DistanceMetric #for creating the matrix


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

import plotly.express as px
import plotly.graph_objects as go


app = Flask(__name__)

@app.route("/")
def mainpy():

    path = 'data' # use your path
    all_files = glob.glob(path + "/*.csv")

    col_list = ('START DATE', 'START TIME', 'START TIME ET', 'SUBJECT', 'LOCATION')

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0, usecols= col_list)
        df['team_file'] = os.path.basename(filename)
        li.append(df)

    mlb_df = pd.concat(li, axis=0, ignore_index=True)


    #create empty columns for lat and lon

    mlb_df["lon"] = ""
    mlb_df["lat"] = ""

    #populate based on stadium name



    mlb_df.loc[(mlb_df.LOCATION =='Nationals Park - Washington'), ('lon', 'lat')] = ('-77.007433','38.87301')
    mlb_df.loc[(mlb_df.LOCATION =='Dodger Stadium - Los Angeles'), ('lon', 'lat')] = ('-118.239958','34.073851')
    mlb_df.loc[(mlb_df.LOCATION =='Busch Stadium - St. Louis'), ('lon', 'lat')] = ('-90.192821','38.622619')
    mlb_df.loc[(mlb_df.LOCATION =='Citi Field - Flushing'), ('lon', 'lat')] = ('-73.845821','40.757088')
    mlb_df.loc[(mlb_df.LOCATION =='TD Ballpark - Dunedin'), ('lon', 'lat')] = ('-82.7873','28.00041')
    mlb_df.loc[(mlb_df.LOCATION =='Yankee Stadium - Bronx'), ('lon', 'lat')] = ('-73.926175','40.829643')
    mlb_df.loc[(mlb_df.LOCATION =='Chase Field - Phoenix'), ('lon', 'lat')] = ('-112.066664','33.445526')
    mlb_df.loc[(mlb_df.LOCATION =='Wrigley Field - Chicago'), ('lon', 'lat')] = ('-87.6553333','41.948438')
    mlb_df.loc[(mlb_df.LOCATION =='Truist Park - Atlanta'), ('lon', 'lat')] = ('-84.467684','33.8907')
    mlb_df.loc[(mlb_df.LOCATION =='Citizens Bank Park - Philadelphia'), ('lon', 'lat')] = ('-87.166495','39.906057')
    mlb_df.loc[(mlb_df.LOCATION =='Tropicana Field - St. Petersburg'), ('lon', 'lat')] = ('-82.653392','27.768225')
    mlb_df.loc[(mlb_df.LOCATION =='loanDepot park - Miami'), ('lon', 'lat')] = ('-80.21957','25.778318')
    mlb_df.loc[(mlb_df.LOCATION =='Petco Park - San Diego'), ('lon', 'lat')] = ('-117.156904','47.591391')
    mlb_df.loc[(mlb_df.LOCATION =='Oracle Park - San Francisco'), ('lon', 'lat')] = ('-112.332327','37.778595')
    mlb_df.loc[(mlb_df.LOCATION =='Oriole Park at Camden Yards - Baltimore'), ('lon', 'lat')] = ('-76.621512','39.284052')
    mlb_df.loc[(mlb_df.LOCATION =='American Family Field - Milwaukee'), ('lon', 'lat')] = ('-87.97115','43.027978')
    mlb_df.loc[(mlb_df.LOCATION == 'PNC Park - Pittsburgh'), ('lon', 'lat')] = ('-80.005666','40.446855')
    mlb_df.loc[(mlb_df.LOCATION == 'Great American Ball Park - Cincinnati'), ('lon', 'lat')] = ('-84.508151','39.097931')
    mlb_df.loc[(mlb_df.LOCATION == 'Coors Field - Denver'), ('lon', 'lat')] = ('-104.994178','39.755882')
    mlb_df.loc[(mlb_df.LOCATION == 'Fenway Park - Boston'), ('lon', 'lat')] = ('-71.097218','42.346676')
    mlb_df.loc[(mlb_df.LOCATION == 'Globe Life Field - Arlington'), ('lon', 'lat')] = ('-97.082504','32.75128')
    mlb_df.loc[(mlb_df.LOCATION == 'Oakland Coliseum - Oakland'), ('lon', 'lat')] = ('-122.332327','37.752483')
    mlb_df.loc[(mlb_df.LOCATION == 'T-Mobile Park - Seattle'), ('lon', 'lat')] = ('-122.332327','47.591391')
    mlb_df.loc[(mlb_df.LOCATION == 'Target Field - Minneapolis'), ('lon', 'lat')] = ('-93.27776','44.981712')
    mlb_df.loc[(mlb_df.LOCATION == 'Guaranteed Rate Field - Chicago'), ('lon', 'lat')] = ('-87.633752','41.829902')
    mlb_df.loc[(mlb_df.LOCATION == 'Progressive Field - Cleveland'), ('lon', 'lat')] = ('-81.685229','41.496211')


    mlb_df.loc[(mlb_df.LOCATION == 'Sahlen Field - Buffalo'), ('lon', 'lat')] = ('-78.8738','42.8804')
    mlb_df.loc[(mlb_df.LOCATION == 'Rogers Centre - Toronto'), ('lon', 'lat')] = ('-79.389353','43.641438')

    mlb_df.loc[(mlb_df.LOCATION == 'Minute Maid Park - Houston'), ('lon', 'lat')] = ('-95.354538','29.757697')
    mlb_df.loc[(mlb_df.LOCATION == 'Angel Stadium - Anaheim'), ('lon', 'lat')] = ('-117.882732','33.800308')
    mlb_df.loc[(mlb_df.LOCATION == 'Kauffman Stadium - Kansas City'), ('lon', 'lat')] = ('-94.480314','39.051672')
    mlb_df.loc[(mlb_df.LOCATION == 'Comerica Park - Detroit'), ('lon', 'lat')] = ('-83.04852','42.338998')



    #drop all the Spring Training sites
    nan_value = float("NaN")
    mlb_df.lon.replace("", nan_value, inplace=True)

    mlb_df.dropna(subset = ["lon"], inplace=True)



    #extract team name and year from filename into their own columns

    # new data frame with split value columns
    new = mlb_df["team_file"].str.split(".", n = 2, expand = True)
  
    # making separate first name column from new data frame
    mlb_df["ScheduledTeam"]= new[0]
  
    # making separate last name column from new data frame
    mlb_df["Year"]= new[1]
  
    mlb_df.head()



    #extract the Opponent from the SUBJECT column

    # new data frame with split value columns
    opp = mlb_df["SUBJECT"].str.split(" at", n = 2, expand = True)
  
    #columns for visitor and home
    mlb_df["Visitor"]= opp[0]
    mlb_df["Home"]= opp[1]
  
    mlb_df.head()


    #clean up any stray scheduling comments

    mlb_df.Home.replace("Time", "", inplace=True)
    mlb_df.Home.replace("-", "", inplace=True)
    mlb_df.Home.replace("TBD", "", inplace=True)



    #import the color map and apply to the main df

    color_df=pd.read_excel('data/colors.xlsx')

    #apply map
    #there are other ways of doing this but plotly is not happy with them for some reason



    mapping1 = dict(color_df[['ScheduledTeam', 'Color1Hex']].values)
    mapping2 = dict(color_df[['ScheduledTeam', 'Color2Hex']].values)
    mapping3 = dict(color_df[['ScheduledTeam', 'Color1Name']].values)
    mapping4 = dict(color_df[['ScheduledTeam', 'Color2Name']].values)



    mlb_df['Color1Hex'] = mlb_df.ScheduledTeam.map(mapping1)
    mlb_df['Color2Hex'] = mlb_df.ScheduledTeam.map(mapping2)
    mlb_df['Color1Name'] = mlb_df.ScheduledTeam.map(mapping3)
    mlb_df['Color2Name'] = mlb_df.ScheduledTeam.map(mapping4)


    #prepare matrix of distances between stadiums
    #reference https://kanoki.org/2019/12/27/how-to-calculate-distance-in-python-and-pandas-using-scipy-spatial-and-distance-functions/

    #first we need to deduplicate the location-lat-lon sets
    mlb_deduped =mlb_df.drop_duplicates(subset=['LOCATION', 'lat','lon'], keep='first')

    #make sure our lat and lon are in numeric format
    mlb_deduped['lon'] = pd.to_numeric(mlb_deduped['lon'],errors='coerce')
    mlb_deduped['lat'] = pd.to_numeric(mlb_deduped['lat'],errors='coerce')



    #convert lat and lon to radians

    mlb_deduped['r_lat'] = np.radians(mlb_deduped['lat'])
    mlb_deduped['r_lon'] = np.radians(mlb_deduped['lon'])

    mlb_deduped.head()


    #calculate the distance

    dist = DistanceMetric.get_metric('haversine')
    mlb_deduped[['r_lat','r_lon']].to_numpy()
    dist.pairwise(mlb_deduped [['r_lat','r_lon']].to_numpy())*3958


    dist_matrix=pd.DataFrame(dist.pairwise(mlb_deduped[['r_lat','r_lon']].to_numpy())*3958,  columns=mlb_deduped.LOCATION.unique(), index=mlb_deduped.LOCATION.unique())
    dist_matrix




    #new reference https://cduvallet.github.io/posts/2020/02/road-trip-map

    #start with a copy of the datafrane

    teams_df= mlb_df


    #deduplicate the teams_df

    teams_deduped = teams_df[teams_df['LOCATION'].ne(teams_df['LOCATION'].shift())]

    teams_deduped.head()



    #fill in the travel path


    #create blank columns to hold the depart_lat and depart_lon which we'll use to build out the travel map

    teams_deduped["depart_lon"] = ""
    teams_deduped["depart_lat"] = ""
    teams_deduped["dotsize"] = 0.3




    #then populate those with the lat and lon of the previous stop

    teams_deduped['depart_lat'] = teams_deduped['lat'].shift(1)
    teams_deduped['depart_lon'] = teams_deduped['lon'].shift(1)


    teams_deduped.head()



    #drop any NaN

    teams_deduped.dropna()



    #testing out some alternate syntax

    teams_deduped['lat'] = teams_deduped['lat'].astype(float)
    teams_deduped['lon'] = teams_deduped['lon'].astype(float)



    #fig3 = px.line_mapbox(lat=teams_deduped.lat, lon=teams_deduped.lon, hover_name=teams_deduped.ScheduledTeam,
    #                     mapbox_style = 'carto-positron',color =teams_deduped.Color1Hex)
    #
    
    #fig3.show()




    #create a dictnioary to map the colors to the teams

    teamcolor_dict=dict(zip(teams_deduped.ScheduledTeam, teams_deduped.Color1Name))

    #create basic figure with the stadiums
    fig2 = go.Figure()

    # Add stadiums
    fig2.add_trace(go.Scattergeo(
        lon=teams_deduped['lon'], lat=teams_deduped['lat'], marker=dict(size=14, color='black'),
        hoverinfo='text', text=teams_deduped['LOCATION'] , name='Team'
    ))
    # Add team travel traces
    for teamname in teams_deduped.ScheduledTeam.unique():
        travel_team_df = teams_deduped[teams_deduped.ScheduledTeam == teamname]
        team_col = teamcolor_dict[teamname]
        fig2.add_trace(go.Scattergeo(
            locationmode='USA-states', mode="lines",
            lon=np.append(travel_team_df['depart_lon'].values, travel_team_df['lon'].values[-1]),
            lat=np.append(travel_team_df['depart_lat'].values, travel_team_df['lat'].values[-1]),
            line=dict(width=1, color=team_col), opacity=0.5,
            hoverinfo='none', name=teamname
        ))
    
    
    fig2['data'][0]['marker']['symbol'] = 'star'

    fig2.update_layout(
        title_text='MLB Travel Routes',
        geo=dict(
            scope='north america',
            projection_type='azimuthal equal area',
            showland=True,
            fitbounds="locations",
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',    
       ),
    )

    fig2.show()





    #make another copy of the df

    travel_df=teams_deduped
    #the lat and lon are already float bu the depart_lat and depart_lon are strings, so we convert them 
    #otherwise, our math below will fail

    travel_df['depart_lat']=travel_df['depart_lat'].astype(float)
    travel_df['depart_lon']=travel_df['depart_lon'].astype(float)



    #create the function to calculate the travel between city pairs


    def haversine_vectorize(lon1, lat1, lon2, lat2):

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        newlon = lon2 - lon1
        newlat = lat2 - lat1

        haver_formula = np.sin(newlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(newlon/2.0)**2

        dist = 2 * np.arcsin(np.sqrt(haver_formula ))
        mi = 3958 * dist # for distance in miles;  for KM use 6367
        return mi


    #here we make sure our function works properly
    #there will be some NaN's-- one for each team, since that is the starting location

    #add column to dataframe when it's been calculated
    travel_df['haversine_dist'] = haversine_vectorize(travel_df['depart_lon'],travel_df['depart_lat'],travel_df['lon'],
                       travel_df['lat'])

    #then round it
    travel_df['haversine_dist']= travel_df['haversine_dist'].round(2)


if __name__ == "__main__":
    app.run()
