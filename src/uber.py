import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import seaborn as sns
import datetime
import plotly
import plotly.graph_objects as go
from matplotlib.ticker import FuncFormatter

def dateParser(s):
    """
    Function that takes a string in the format yyyy-mm-dd hh:mm:ss, and
    returns the same as a datetime object.
    :param: s
    :type: str
    """
    # assert isinstance(s, str)
    return datetime.datetime(int(s[0:4]), int(s[5:7]), int(s[8:10]), int(s[11:13]))

def duration_to_minutes(s):
    """
    Function that takes a string with the hh:mm:ss format and
    returns the integer equivalent of the total time in minutes, 
    or zero for missing values in a Pandas dataframe.
    """
    # assert isinstance(s, str)
    if pd.isnull(s):
        val = 0 #note: this fills with 0 the 38 instances with null (missing) values
    else:
        hms = s.split(':')
        val = int(hms[0])*60 + int(hms[1]) + int(hms[2])/60.0
    return val

def millions_format(x, pos):
    """
    Args are the value and tick position. 
    Returns number of millions with one decimal, and M in lieu of 6 zeros.
    """
    return '{:.1f}{}'.format(x * 1e-6, 'M')

def get_visualization_dataframe(path):
    '''
    Takes raw data in uber_nyc_data.csv and turns it into a more useful dataframe for visualization
    :param: path
    :type: str
    :return: df_viz
    :type: pd.DataFrame
    '''

    assert isinstance(path,str) # path = 'data/uber_nyc_data.csv'

    df_uber = pd.read_csv(path)
    df38 = df_uber[df_uber.trip_duration.isnull() & df_uber.trip_distance.isnull()]
    df_uber['pu_date_hour'] = df_uber.pickup_datetime.apply(dateParser)
    df_uber = df_uber.drop('pickup_datetime', axis=1)
    df_uber['pu_date'] = pd.Series(map(lambda x: x.astype('datetime64[D]'), df_uber['pu_date_hour'].values))
    df_uber['year'] = df_uber['pu_date_hour'].dt.year
    df_uber['month'] = df_uber['pu_date_hour'].dt.month
    df_uber['day'] = df_uber['pu_date_hour'].dt.day
    df_uber['hour'] = df_uber['pu_date_hour'].dt.hour
    df_uber['weekday'] = df_uber['pu_date_hour'].dt.dayofweek
    df_uber['duration_min'] = df_uber.trip_duration.apply(duration_to_minutes)
    df_DistDur = df_uber.groupby(['origin_taz', 'destination_taz'])[['trip_distance', 'duration_min']].mean()
    for i in df38.index:
        orig = df_uber.loc[i, 'origin_taz']
        dest = df_uber.loc[i, 'destination_taz']
        df_uber.loc[i, 'trip_distance'] = df_DistDur.loc[orig, dest].trip_distance
        df_uber.loc[i, 'duration_min'] = df_DistDur.loc[orig, dest].duration_min
    df_uber['trip_mph_avg'] = df_uber.trip_distance/(df_uber.duration_min/60.0)
    df_uber = df_uber.drop('trip_duration', axis=1)
    df_uber = df_uber.drop('pu_date_hour', axis=1)
    base_fare = 2.55
    per_minute = 0.35
    per_mile = 1.75
    min_fare = 8
    df_uber['est_revenue'] = df_uber.eval('@base_fare + duration_min * @per_minute + trip_distance * @per_mile')
    df_uber.loc[df_uber.est_revenue < 8, 'est_revenue'] = min_fare
    df_viz = df_uber[(df_uber.pu_date != datetime.date(2015, 9, 1)) & (df_uber.duration_min <= 960)].copy()
    return df_viz

def plot_bivariate_distributions(dataframe):
    '''
    Plots the distribution of trip distance, trip duration, and average speed
    over the total trips in the dataset
    :param: dataframe
    :type: pd.DataFrame
    '''
    assert isinstance(dataframe,pd.DataFrame)

    freq, bins_dist = np.histogram(dataframe.trip_distance, bins=10, range=(0, 25))
    freq, bins_dur = np.histogram(dataframe.duration_min, bins=10, range=(0, 50))
    freq, bins = np.histogram(dataframe.trip_mph_avg, bins=10, range=(0, 50))
    #Here we plot the distribution of trip duration next to the histogram of trip distance for comparison.
    fig = plt.figure(figsize=(16, 7))

    formatter = FuncFormatter(millions_format)

    plt.subplot(1,3,1)
    ax1 = dataframe.trip_distance.dropna().hist(bins=bins_dist, color = 'darkslategrey')
    ax1.yaxis.set_major_formatter(formatter)

    plt.xlabel('Distance (miles)', fontsize=14, weight='bold')
    plt.ylabel('Number of Trips', fontsize=14, weight='bold')
    plt.tick_params(labelsize=14)
    plt.title('Distribution of Trip Distance', color='k', fontsize=16)


    plt.subplot(1,3,2)
    ax2 = dataframe.duration_min.hist(bins=bins_dur, color = 'darkslategrey')
    ax2.yaxis.set_major_formatter(formatter)
    ax2.set_xticks(bins_dur) #bins are in 5 minutes intervals.

    plt.xlabel('Duration (minutes)', fontsize=14, weight='bold')
    plt.ylabel('')
    plt.tick_params(labelsize=14)
    plt.title('Distribution of Trip Duration', color='k', fontsize=16)

    plt.subplot(1,3,3)
    ax3 = dataframe.trip_mph_avg.hist(bins=bins, color = 'darkslategrey')
    ax3.yaxis.set_major_formatter(formatter)
    ax3.set_xticks(bins) #bins are in 5 minutes intervals.

    plt.xlabel('Average Speed (mph)', fontsize=14, weight='bold')
    # plt.ylabel('Number of Trips', fontsize=14, weight='bold')
    plt.tick_params(labelsize=14)
    plt.title('Distribution of Average Speed', color='k', fontsize=16)
    plt.savefig('tripdist_tripdur_avgspeedvs_num_trips', bbox_inches = "tight")

def produce_bar_graph(df_viz):
    '''
    Creates 4 bar graphs that show the number of trips for the
    weekday(>5 miles and <5 miles) and weekaned(>5 miles and <5 miles)
    :param: df_viz
    :type: pd.DataFrame
    '''

    assert(isinstance(df_viz,pd.DataFrame))

    fig = plt.figure(figsize = (16,12))

    plt.subplot(2,2,1)
    ax1 = df_viz[(df_viz.weekday < 5) & (df_viz.trip_distance >= 5)].    groupby('hour')['trip_distance'].count().plot(kind='bar', rot = 0,color='darkslategrey')
    plt.xlabel('Hour', fontsize=14, color='darkslategrey')
    plt.ylabel('Number of Trips', fontsize=14, color='darkslategrey')
    plt.ylim(0, 1400000)
    plt.title('Weekday: 5 miles or greater', fontsize=14, weight='bold', color='darkslategrey')

    plt.subplot(2,2,2)
    ax2 = df_viz[(df_viz.weekday < 5) & (df_viz.trip_distance < 5)].    groupby('hour')['trip_distance'].count().plot(kind='bar', rot = 0,color='darkslategrey')
    plt.xlabel('Hour', fontsize=14, color='darkslategrey')
    plt.title('Weekday: Less than 5 miles', fontsize=14, weight='bold', color='darkslategrey')

    plt.subplot(2,2,3)
    ax3 = df_viz[(df_viz.weekday >= 5) & (df_viz.trip_distance >= 5)].    groupby('hour')['trip_distance'].count().plot(kind='bar', rot = 0,color='darkslategrey')
    plt.xlabel('Hour', fontsize=14, color='darkslategrey')
    plt.ylabel('Number of Trips', fontsize=14, color='darkslategrey')
    plt.ylim(0, 500000)
    plt.title('Weekend: 5 miles or greater', fontsize=14, weight='bold',color='darkslategrey')

    plt.subplot(2,2,4)
    ax4 = df_viz[(df_viz.weekday >= 5) & (df_viz.trip_distance < 5)].    groupby('hour')['trip_distance'].count().plot(kind='bar', rot = 0, color='darkslategrey')
    plt.xlabel('Hour', fontsize=14, color='darkslategrey')
    plt.title('Weekend: Less than 5 miles', fontsize=14, weight='bold', color='darkslategrey')


    fig.subplots_adjust(hspace=0.4)
    plt.show(block=False)
    plt.pause(0.01)


def produce_line_graph(df_viz):
    '''
    Plots the number of trips vs. the days of the year
    :param: df_viz
    :type: pd.DataFrame
    '''
    assert(isinstance(df_viz,pd.DataFrame))

    byDate = df_viz.groupby('pu_date')['id'].count()
    

    fig = plt.figure()

    ax = byDate.plot(figsize = (16, 8), fontsize = 12, ylim = (10000, 180000), color = 'darkslategrey')

    plt.title('Total Trips per Day Considering Weather Conditions and Major Events', fontsize= 24)
    plt.tick_params(labelsize=18)
    plt.xlabel('Month', size=24)
    plt.ylabel('Number of Trips', size=24)

    plt.show(block=False)
    plt.pause(0.01)

def plot_weekday_avg_speed(dataframe):
    '''
    Plots the weekday average speed vs the hour of the day
    :param: dataframe
    :type: pd.DataFrame
    '''

    assert(isinstance(dataframe,pd.DataFrame))

    df_plt7 = dataframe[dataframe.weekday < 5].groupby('hour')['trip_mph_avg'].median()
    plt.figure(figsize =(14, 6))
    kwargs = {'fontsize': 12, 'ha':'center', 'va': 'top', 'color': 'k', 'weight': 'bold'}
    #weekdays only: rush hour traffic 7-9 + 16-18
    ax = df_plt7.plot(marker = 'o', color = 'k')
    for x, y in zip(df_plt7.index, df_plt7.values):
        ax.annotate('{:.0f}'.format(y), xy=(x, y), xytext= (0, 24), textcoords='offset points', **kwargs)
    ax.set_facecolor('#F9F9F9')
    ax.get_yaxis().set_ticks([]) #hide tick labels on y-axis
    plt.fill([7,9,9,7], [0,0,30,30], 'red', alpha=0.2)
    plt.fill([16,18,18,16], [0,0,30,30], 'red', alpha=0.2)
    plt.xticks(range(24))
    plt.xlabel('Hour', fontsize=14)
    plt.ylabel('Trip Average Speed', fontsize=14)
    plt.ylim(5, 30)
    plt.xlim(-0.5, 23.5)
    plt.tick_params(labelsize=14)
    plt.title('Weekday Average Speed per Hour of the Day - Highlight for Peak Traffic', fontsize = 16, color='k')
    plt.show(block=False)
    plt.pause(0.01)

def produce_heatmap(df_viz):
    '''
    Creates heatmap of traffic speed over the days of the week and the hour of the day
    :param: df_viz
    :type: pd.DataFrame
    '''
    assert(isinstance(df_viz,pd.DataFrame))

    plot_tepm = np.zeros((24, 7))
    np.save('to_send.npy', plot_tepm)
    for i in range(7):
        plot_tepm[:, i] = list(df_viz[(df_viz.weekday == i)].groupby('hour')['trip_mph_avg'].median())
        
    speeds = [list(i) for i in plot_tepm]
    
    fig = go.Figure(data=go.Heatmap(
                   z=speeds,
                   x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday','Sunday'],
                   y=['12 a.m.','1 a.m.','2 a.m.', '3 a.m.', '4 a.m.','5 a.m.','6 a.m.','7 a.m.','8 a.m.','9 a.m.',\
                      '10 a.m.', '11 a.m.', '12 p.m.', '1 p.m.', '2 p.m.','3 p.m.','4 p.m.','5 p.m.',\
                      '6 p.m.','7 p.m.', '8 p.m.', '9 p.m.','10 p.m.','11 p.m.']
                   ,colorscale = 'portland'
                    ))
                    
    fig.update_layout(autosize=False,width=1000,height=800,title="Traffic distribution during the week",font=dict(size=18,color="black"),yaxis_title="hour")
    fig.show()

def most_popular_pickups_and_dropoff(df_uber):
    '''
    This function plots the most popular pick-ups and drop-offs
    Input:
    df_uber: The general dataframe we will be working on
    x-axis: Number of trips per origin location
    y-axis: zone names
    '''
    
    assert(isinstance(df_uber,pd.DataFrame))
    df_viz = df_uber[['origin_taz', 'destination_taz']]
    fig = plt.figure(figsize = (12,8))

    plt.subplot(1,2,1)
    ax1 = df_viz.origin_taz.value_counts(ascending = True).plot(kind = 'barh', color = 'black')
    ax1.set_xticklabels(['0', '1M', '2M', '3M', '4M', '5M', '6M'])
    plt.tick_params(labelsize=12)
    plt.xlabel('Number of Trips per Origin Locations', fontsize = 16, color='black')

    plt.subplot(1,2,2)
    ax2 = df_viz.destination_taz.value_counts(ascending = True).plot(kind = 'barh', color = 'black')
    ax2.set_xticklabels(['0', '1M', '2M', '3M', '4M', '5M', '6M'])
    plt.tick_params(labelsize=12)
    plt.xlabel('Number of Trips per Destination Locations', fontsize = 16, color='black')
    plt.show(block=False)
    plt.pause(0.01)