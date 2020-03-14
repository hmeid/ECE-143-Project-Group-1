# Load in src files
from src.uber import *
from src.taxi import *

# Load in dataframes
df_uber = pd.read_csv('data/uber_nyc_data.csv')
df_viz = get_visualization_dataframe('data/uber_nyc_data.csv')

# Analysis of Number of Trips
plot_bivariate_distributions(df_viz)
produce_bar_graph(df_viz)
produce_line_graph(df_viz)

# Traffic Distribution
plot_weekday_avg_speed(df_viz)

# see Jupyter Notebook for this plot
# produce_heatmap(df_viz)

# Popular Pick-ups and Drop-offs
most_popular_pickups_and_dropoff(df_uber)

# Boroughs with Most Pick-ups and Drop-offs
# Load taxi data into database
nyc_database = create_engine('sqlite:///nyc_database1.db')
load_to_database(nyc_database)
plot_boroughs_zones()
plot_mostpickups_zones_boroughs(nyc_database)
plt.show()