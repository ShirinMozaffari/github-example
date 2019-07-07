#!/usr/bin/env python
# coding: utf-8

# # Week 3 Assignment

# In this part, I am reading the data from the Wikipedia page and storing it in 'df'

# 

# In[ ]:





# In[20]:


import pandas as pd


# In[41]:


dfs = pd.read_html('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M',header=0)
df = dfs[0]


# In[42]:


df.columns=['PostalCode', 'Borough', 'Neighbourhood']
df.head()


# In[43]:


df.shape


# As can be seen, there are 'Not Assigned' Values in columns 'Borough' and 'Neighborhood'.

# ### Removing 'Not Assigned' values from the data frame

# I am removing 'Not Assigned' values from the column 'Borough' using the following command:

# In[44]:


df= df[df['Borough'] != 'Not assigned']


# In[45]:


df.shape


# As can be seen, the number of rows is decreased. It was 288 and after removing 'Not Assigned' values, there is only 211 rows in the data frame.

# In[46]:


df.head(10)


# There is only one row that the neighborhood is not assigned

# In[29]:


#df.iloc[8, 2] = 'Queen\'s Park'


# In[38]:


#df.head(40)


# As can be seen, there are no 'Not Assigned' values in neither 'Borough' nor 'Neighbourhood'

# ### Adding Postal Code to the data frame

# Grouping the neighborhood based on their postral code

# In[47]:



df_post = df.groupby(['PostalCode']).agg({'Borough':pd.Series.unique,'Neighbourhood':lambda x:','.join(x)})


# In[48]:


df_post.reset_index(inplace = True)


# In[49]:


df.head()


# In[51]:


df_post.shape


# If a cell has a borough but a Not assigned neighborhood, then the neighborhood will be the same as the borough.

# In[53]:



df_post.loc[df_post['Neighbourhood'] =='Not assigned','Neighbourhood'] = df_post.loc[df_post['Neighbourhood'] =='Not assigned', 'Borough']


# In[54]:


df_post.shape


# In[ ]:





# # Second Part 

# In[56]:


geo = pd.read_csv('http://cocl.us/Geospatial_data', header = 0 )


# In[57]:


geo.head()


# Changing 'Postal Code' to 'PostalCode' in order to be compatible with the previous dataset (df)

# In[58]:



geo.rename(columns={"Postal Code":"PostalCode"}, inplace = True)


# In[59]:


geo.head()


# As can be seen, the name of the column has changed. The following command will merge the two datasets based on the Postal Code.

# In[60]:


df = pd.merge(df_post, geo, on='PostalCode')
df.shape


# Now, we'll a few rown of the new dataset :

# In[61]:


df.head(10)


# In[35]:


df.shape


# # Third Part

# Generating map of toronto and clustering the different neighbourhoods. First, we need to import the necessary libraries.

# In[62]:


from geopy.geocoders import Nominatim

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

import folium # map rendering library


# Based on the labs, we need to define the address and location

# In[63]:


address = 'Toronto'

geolocator = Nominatim(user_agent="Toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto City are {}, {}.'.format(latitude, longitude))


# In[64]:


# create map of Toronto using latitude and longitude values
map_Toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df['Latitude'], df['Longitude'], df['Borough'], df['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_Toronto)  
    
map_Toronto


# In[66]:


df_Toronto = df[df['Borough'].str.contains('Toronto')].reset_index(drop=True)
df_Toronto


# In[ ]:





# In[67]:


# create map of Toronto Borough using latitude and longitude values
map_Toronto_Borough = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df_Toronto['Latitude'], df_Toronto['Longitude'], df_Toronto['Borough'], df_Toronto['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_Toronto_Borough)  
    
map_Toronto_Borough


# We need our ID and secret from foursquare developer account.

# In[68]:


CLIENT_ID = 'NCJ0VG54N3HI1VTDLYFVWAR2SOHO5FFM0MKB50V4PNPKBN4G'
CLIENT_SECRET = '5EIKV3ACBYW1QEWSQSAWCWKNU2WVCXI13HFTFCP4XNRMSMVC'
VERSION = '20180604' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[69]:


df_Toronto.shape


# In[70]:


df_Toronto.loc[0, 'Neighbourhood']


# In[71]:


#Get the neighborhood's latitude and longitude values.
neighbourhood_latitude = df_Toronto.loc[0, 'Latitude'] # neighborhood latitude value
neighbourhood_longitude = df_Toronto.loc[0, 'Longitude'] # neighborhood longitude value

neighbourhood_name = df_Toronto.loc[0, 'Neighbourhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighbourhood_name, 
                                                               neighbourhood_latitude, 
                                                               neighbourhood_longitude))


# In[72]:


LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # define radius
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighbourhood_latitude, 
    neighbourhood_longitude, 
    radius, 
    LIMIT)
url # display URL


# In[73]:


#Send the GET request and examine the resutls
results = requests.get(url).json()
results


# In[74]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[75]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues


# In[76]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# In[77]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood', 
                  'Neighbourhood Latitude', 
                  'Neighbourhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[81]:


Toronto_venues = getNearbyVenues(names=df_Toronto['Neighbourhood'],
                                   latitudes=df_Toronto['Latitude'],
                                   longitudes=df_Toronto['Longitude']
                                  )


# In[83]:



#Let's check the size of the resulting dataframe
print(Toronto_venues.shape)
Toronto_venues.head()


# In[84]:


#Let's check how many venues were returned for each neighborhood
Toronto_venues.groupby('Neighbourhood').count()


# In[85]:



# Let's find out how many unique categories can be curated from all the returned venues
print('There are {} uniques categories.'.format(len(Toronto_venues['Venue Category'].unique())))


# In[86]:


# one hot encoding
Toronto_onehot = pd.get_dummies(Toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
Toronto_onehot['Neighbourhood'] = Toronto_venues['Neighbourhood'] 

# move neighborhood column to the first column
fixed_columns = [Toronto_onehot.columns[-1]] + list(Toronto_onehot.columns[:-1])
Toronto_onehot = Toronto_onehot[fixed_columns]

Toronto_onehot.head(100)


# In[87]:


Toronto_onehot.shape


# In[89]:


Toronto_grouped = Toronto_onehot.groupby('Neighbourhood').mean().reset_index()
Toronto_grouped.head()


# In[90]:



Toronto_grouped.shape


# In[91]:


num_top_venues = 5

for hood in Toronto_grouped['Neighbourhood']:
    print("----"+hood+"----")
    temp = Toronto_grouped[Toronto_grouped['Neighbourhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[92]:


# First, let's write a function to sort the venues in descending order.
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[93]:


# Now let's create the new dataframe and display the top 10 venues for each neighborhood.
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighbourhoods_venues_sorted = pd.DataFrame(columns=columns)
neighbourhoods_venues_sorted['Neighbourhood'] = Toronto_grouped['Neighbourhood']

for ind in np.arange(Toronto_grouped.shape[0]):
    neighbourhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(Toronto_grouped.iloc[ind, :], num_top_venues)

neighbourhoods_venues_sorted.head()


# In[ ]:





# In[94]:


# set number of clusters
kclusters = 5

Toronto_grouped_clustering = Toronto_grouped.drop('Neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(Toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[95]:


# Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.
# add clustering labels
neighbourhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

Toronto_merged = df_Toronto

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
Toronto_merged = Toronto_merged.join(neighbourhoods_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')

Toronto_merged.head() # check the last columns!


# In[96]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(Toronto_merged['Latitude'], Toronto_merged['Longitude'], Toronto_merged['Neighbourhood'], Toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[97]:



Toronto_merged.loc[Toronto_merged['Cluster Labels'] == 0, Toronto_merged.columns[[1] + list(range(5, Toronto_merged.shape[1]))]]


# In[98]:



Toronto_merged.loc[Toronto_merged['Cluster Labels'] == 1, Toronto_merged.columns[[1] + list(range(5, Toronto_merged.shape[1]))]]


# In[99]:


Toronto_merged.loc[Toronto_merged['Cluster Labels'] == 2, Toronto_merged.columns[[1] + list(range(5, Toronto_merged.shape[1]))]]


# In[100]:



Toronto_merged.loc[Toronto_merged['Cluster Labels'] == 3, Toronto_merged.columns[[1] + list(range(5, Toronto_merged.shape[1]))]]


# In[101]:


Toronto_merged.loc[Toronto_merged['Cluster Labels'] == 4, Toronto_merged.columns[[1] + list(range(5, Toronto_merged.shape[1]))]]


# In[ ]:





# In[ ]:




