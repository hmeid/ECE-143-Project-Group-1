# Uber and Yellow Taxi Trip Data Analysis for New York City
### ECE 143 Group 1 Winter 2020 <br> Pranav Rao, Nikhil Dutt, Hassan Eid, Mayank Gupta


### File Structure
* data/
    * [get_data.py](https://github.com/hmeid/ECE-143-Project-Group-1/blob/master/data/get_data.py)
        `script to download data`
* src/
    * [uber.py](https://github.com/hmeid/ECE-143-Project-Group-1/blob/master/src/uber.py)<br>
        ```
        -> description: this file contains all functions necessary for uber analysis
        -> included functions:
            
            dateParser(s): Function that takes a string in the format yyyy-mm-dd hh:mm:ss,
            and returns the same as a datetime object.

            duration_to_minutes(s): Function that takes a string with the hh:mm:ss format
            and returns the integer equivalent of the total time in minutes, or zero for
            missing values in a Pandas dataframe.

            millions_format(x, pos): Args are the value and tick position. Returns number
            of millions with one decimal, and M in lieu of 6 zeros.

            get_visualization_dataframe(path):  Takes raw data in uber_nyc_data.csv and
            turns it into a more useful dataframe for visualization.

            plot_bivariate_distributions(dataframe): Plots the distribution of trip distance,
            trip duration, and average speed over the total trips in the dataset.

            produce_bar_graph(df_viz): Creates 4 bar graphs that show the number of trips for
            the weekday(>5 miles and <5 miles) and weekaned(>5 miles and <5 miles).

            produce_line_graph(df_viz): Plots the number of trips vs. the days of the year.

            plot_weekday_avg_speed(dataframe): Plots the weekday average speed vs. the hour of the day.

            produce_heatmap(df_viz): Creates heatmap of traffic speed over the days of the week
            and the hour of the day.

            most_popular_pickups_and_dropoff(df_uber): This function plots the 
            most popular pick-ups and drop-offs.
        ```
    * [taxi.py](https://github.com/hmeid/ECE-143-Project-Group-1/blob/master/src/taxi.py)<br>
        ```
        -> description: this file contains all functions necessary for taxi analysis
        -> included functions:

            create_database(): Creates a database to store taxi data.

            load_to_database(nyc_database): Loads in the downloaded taxi data to the created database.

            plt_clock(ax, radii, title, color): Plots a clock.

            get_lat_lon(sf,shp_dic): Gets lattitude and longitude for plotting.

            get_boundaries(sf): Gets boundaries for the plot.

            draw_region_map(ax, sf, shp_dic, heat={}): Draws the boroughs of New York

            draw_zone_map(ax, sf, shp_dic, heat={}, text=[], arrows=[]): Draws the PU and DO zones
            of New York.

            plot_boroughs_zones(): Plots all the PU and DO zones in the boroughs of New York.

            def plot_mostpickups_zones_boroughs(nyc_database): Plots a heatmap of the most popular
            PU and DO zones.
        ```
* [ECE_143_Final_Project_1.pdf](https://github.com/hmeid/ECE-143-Project-Group-1/blob/master/ECE_143_Final_Project_1.pdf)
* [ECE_143_Final_Visualizations_1.ipynb](https://github.com/hmeid/ECE-143-Project-Group-1/blob/master/ECE_143_Final_Visualizations_1.ipynb)
* [main.py](https://github.com/hmeid/ECE-143-Project-Group-1/blob/master/main.py)

### Dataset
After cloning the repository, the first step in running this project is to get the data. To get the necessary dataset, run the following commands to download both the Uber and Yellow Taxi data. NOTE: These are large files and will take time to download!

```
cd data
python get_data.py
```
This will add multiple .csv files and a shape directory into the data folder. The data folder should now have this structure: <br>
* data/
   * shape/
   * get_data.py
   * nyc.2017-01.csv
   * nyc.2017-02.csv
   * nyc.2017-03.csv
   * nyc.2017-04.csv
   * nyc.2017-05.csv
   * nyc.2017-06.csv
   * taxi_zones.zip
   * uber_nyc_data.csv
   
Now it's time to analyze the data.

### Dependencies
Along with Python 3.7.X, our project requires the following third-party modules:
* pandas
* numpy
* matplotlib
* seaborn
* pyshp
* plotly
* shapely
* descartes
* sqlalchemy

All third-party modules can be installed using the following command:
```
conda install 'package_name'
```

### Analysis
In order to run the analysis, you can either run the command:
```
python main.py
```
or open the Jupyter Notebook inlcuded in this repository:
```
jupyter notebook ECE_143_Final_Visualizations_1
```
Both methods will result in the same plots.
