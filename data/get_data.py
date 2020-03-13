import pandas as pd
import numpy as np
import urllib.request
import zipfile

def download_dataset():
    '''
    Function is used to download uber and taxi data
    !!! File sizes are large and thus take a long time to download
    '''

    # get uber data
    urllib.request.urlretrieve('https://s3.amazonaws.com/nyc-tlc/misc/uber_nyc_data.csv', 'uber_nyc_data.csv')

    # get taxi data
    for month in range(1,7):
        urllib.request.urlretrieve("https://s3.amazonaws.com/nyc-tlc/trip+data/"+ "yellow_tripdata_2017-{0:0=2d}.csv".format(month), "nyc.2017-{0:0=2d}.csv".format(month))
    # Download the location Data
    urllib.request.urlretrieve("https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip", "taxi_zones.zip")
    with zipfile.ZipFile("taxi_zones.zip","r") as zip_ref:
        zip_ref.extractall("./shape")

def main():
    download_dataset()

if __name__ == "__main__":
    main()