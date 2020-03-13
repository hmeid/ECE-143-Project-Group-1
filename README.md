# Uber and Yellow Taxi Trip Data Analysis for New York City
#### ECE 143 Group 1 Winter 2020 <br> Pranav Rao, Nikhil Dutt, Hassan Eid, Mayank Gupta

## Overview
### Dataset
After cloning the repository, the first step in running this project is to get the data. To get the necessary dataset, run the following commands to download both the Uber and Yellow Taxi data. NOTE: These are large files and will take time to download!

```
cd data
python get_data.py
```
This will add multiple .csv files and a shape directory into the data folder. Now it's time to analyze the data.
 
### Dependencies
Along with Python 3.7.X, our project requires the following third-party modules:
* pandas
* numpy
* matplotlib
* pyshp
* shapely
* descartes
* sqlalchemy

All third-party modules can be installed using the following command:
```
conda install 'package_name'
```
