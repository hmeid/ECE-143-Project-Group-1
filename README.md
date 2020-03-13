# Uber and Yellow Taxi Trip Data Analysis for New York City
#### ECE 143 Group 1 Winter 2020 <br> Pranav Rao, Nikhil Dutt, Hassan Eid, Mayank Gupta


### File Structure
* data/
    * [get_data.py](https://github.com/hmeid/ECE-143-Project-Group-1/blob/master/data/get_data.py)
        `script to download data`
* src/
    * \_\_init\_\_.py
    * [uber.py](https://github.com/hmeid/ECE-143-Project-Group-1/blob/master/src/uber.py)
        `contains functions necessary for uber analysis`
    * [taxi.py](https://github.com/hmeid/ECE-143-Project-Group-1/blob/master/src/taxi.py)
        `contains functions necessary for taxi analysis`
* [ECE_143_Final_Project_1.pdf](https://github.com/hmeid/ECE-143-Project-Group-1/ECE_143_Final_Project_1.pdf)
* [ECE_143_Final_Visualizations_1.ipynb](https://github.com/hmeid/ECE-143-Project-Group-1/ECE_143_Final_Visualizations_1.pdf)
* [main.py](https://github.com/hmeid/ECE-143-Project-Group-1/main.py)

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
* seaborn
* pyshp
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
