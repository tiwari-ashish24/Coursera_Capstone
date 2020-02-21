
###  Import libraries


```python
import numpy as np
import pandas as pd 
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

import json 
from geopy.geocoders import Nominatim 
import requests 
from bs4 import BeautifulSoup 
from pandas.io.json import json_normalize 

# Matplotlib 
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#map library
#!conda install -c conda-forge folium=0.5.0 --yes
import folium 

print("Libraries imported.")
```

    Libraries imported.


###  Scrap data from Wikipedia page into a DataFrame


```python
# send the GET request
data = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M').text
```


```python
# parse data into a beautifulsoup object
soup = BeautifulSoup(data, 'html.parser')
```


```python
# create three lists to store table data
postalCodeList = []
boroughList = []
neighborhoodList = []
```


```python
# put data into the lists
for row in soup.find('table').find_all('tr'):
    cells = row.find_all('td')
    if(len(cells) > 0):
        postalCodeList.append(cells[0].text)
        boroughList.append(cells[1].text)
        neighborhoodList.append(cells[2].text.rstrip('\n')) 
```


```python
# create a new DataFrame from the three lists
df = pd.DataFrame({"PostalCode": postalCodeList,
                           "Borough": boroughList,
                           "Neighborhood": neighborhoodList})

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M2A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
    </tr>
  </tbody>
</table>
</div>



###  Drop cells that is "Not assigned"


```python
df_dropna = df[df.Borough != "Not assigned"].reset_index(drop=True)
df_dropna.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Heights</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor</td>
    </tr>
  </tbody>
</table>
</div>



###  Group neighborhoods



```python
# group neighborhoods in the same borough
df_grouped = df_dropna.groupby(["PostalCode", "Borough"], as_index=False).agg(lambda x: ", ".join(x))
df_grouped.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge, Malvern</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
    </tr>
  </tbody>
</table>
</div>



###  Assign Neighborhood same as Borough where it is 'Not Assigned'


```python
for index, row in df_grouped.iterrows():
    if row["Neighborhood"] == "Not assigned":
        row["Neighborhood"] = row["Borough"]
        
df_grouped.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge, Malvern</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
    </tr>
  </tbody>
</table>
</div>



###  Print shape of dataframe


```python
df_grouped.shape
```




    (103, 3)




```python
df_grouped
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge, Malvern</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M1J</td>
      <td>Scarborough</td>
      <td>Scarborough Village</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M1K</td>
      <td>Scarborough</td>
      <td>East Birchmount Park, Ionview, Kennedy Park</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M1L</td>
      <td>Scarborough</td>
      <td>Clairlea, Golden Mile, Oakridge</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M1M</td>
      <td>Scarborough</td>
      <td>Cliffcrest, Cliffside, Scarborough Village West</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M1N</td>
      <td>Scarborough</td>
      <td>Birch Cliff, Cliffside West</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M1P</td>
      <td>Scarborough</td>
      <td>Dorset Park, Scarborough Town Centre, Wexford ...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M1R</td>
      <td>Scarborough</td>
      <td>Maryvale, Wexford</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M1S</td>
      <td>Scarborough</td>
      <td>Agincourt</td>
    </tr>
    <tr>
      <th>13</th>
      <td>M1T</td>
      <td>Scarborough</td>
      <td>Clarks Corners, Sullivan, Tam O'Shanter</td>
    </tr>
    <tr>
      <th>14</th>
      <td>M1V</td>
      <td>Scarborough</td>
      <td>Agincourt North, L'Amoreaux East, Milliken, St...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>M1W</td>
      <td>Scarborough</td>
      <td>L'Amoreaux West</td>
    </tr>
    <tr>
      <th>16</th>
      <td>M1X</td>
      <td>Scarborough</td>
      <td>Upper Rouge</td>
    </tr>
    <tr>
      <th>17</th>
      <td>M2H</td>
      <td>North York</td>
      <td>Hillcrest Village</td>
    </tr>
    <tr>
      <th>18</th>
      <td>M2J</td>
      <td>North York</td>
      <td>Fairview, Henry Farm, Oriole</td>
    </tr>
    <tr>
      <th>19</th>
      <td>M2K</td>
      <td>North York</td>
      <td>Bayview Village</td>
    </tr>
    <tr>
      <th>20</th>
      <td>M2L</td>
      <td>North York</td>
      <td>Silver Hills, York Mills</td>
    </tr>
    <tr>
      <th>21</th>
      <td>M2M</td>
      <td>North York</td>
      <td>Newtonbrook, Willowdale</td>
    </tr>
    <tr>
      <th>22</th>
      <td>M2N</td>
      <td>North York</td>
      <td>Willowdale South</td>
    </tr>
    <tr>
      <th>23</th>
      <td>M2P</td>
      <td>North York</td>
      <td>York Mills West</td>
    </tr>
    <tr>
      <th>24</th>
      <td>M2R</td>
      <td>North York</td>
      <td>Willowdale West</td>
    </tr>
    <tr>
      <th>25</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>26</th>
      <td>M3B</td>
      <td>North York</td>
      <td>Don Mills North</td>
    </tr>
    <tr>
      <th>27</th>
      <td>M3C</td>
      <td>North York</td>
      <td>Flemingdon Park, Don Mills South</td>
    </tr>
    <tr>
      <th>28</th>
      <td>M3H</td>
      <td>North York</td>
      <td>Bathurst Manor, Downsview North, Wilson Heights</td>
    </tr>
    <tr>
      <th>29</th>
      <td>M3J</td>
      <td>North York</td>
      <td>Northwood Park, York University</td>
    </tr>
    <tr>
      <th>30</th>
      <td>M3K</td>
      <td>North York</td>
      <td>CFB Toronto, Downsview East</td>
    </tr>
    <tr>
      <th>31</th>
      <td>M3L</td>
      <td>North York</td>
      <td>Downsview West</td>
    </tr>
    <tr>
      <th>32</th>
      <td>M3M</td>
      <td>North York</td>
      <td>Downsview Central</td>
    </tr>
    <tr>
      <th>33</th>
      <td>M3N</td>
      <td>North York</td>
      <td>Downsview Northwest</td>
    </tr>
    <tr>
      <th>34</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>35</th>
      <td>M4B</td>
      <td>East York</td>
      <td>Woodbine Gardens, Parkview Hill</td>
    </tr>
    <tr>
      <th>36</th>
      <td>M4C</td>
      <td>East York</td>
      <td>Woodbine Heights</td>
    </tr>
    <tr>
      <th>37</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
    </tr>
    <tr>
      <th>38</th>
      <td>M4G</td>
      <td>East York</td>
      <td>Leaside</td>
    </tr>
    <tr>
      <th>39</th>
      <td>M4H</td>
      <td>East York</td>
      <td>Thorncliffe Park</td>
    </tr>
    <tr>
      <th>40</th>
      <td>M4J</td>
      <td>East York</td>
      <td>East Toronto</td>
    </tr>
    <tr>
      <th>41</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
    </tr>
    <tr>
      <th>42</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>The Beaches West, India Bazaar</td>
    </tr>
    <tr>
      <th>43</th>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
    </tr>
    <tr>
      <th>44</th>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
    </tr>
    <tr>
      <th>45</th>
      <td>M4P</td>
      <td>Central Toronto</td>
      <td>Davisville North</td>
    </tr>
    <tr>
      <th>46</th>
      <td>M4R</td>
      <td>Central Toronto</td>
      <td>North Toronto West</td>
    </tr>
    <tr>
      <th>47</th>
      <td>M4S</td>
      <td>Central Toronto</td>
      <td>Davisville</td>
    </tr>
    <tr>
      <th>48</th>
      <td>M4T</td>
      <td>Central Toronto</td>
      <td>Moore Park, Summerhill East</td>
    </tr>
    <tr>
      <th>49</th>
      <td>M4V</td>
      <td>Central Toronto</td>
      <td>Deer Park, Forest Hill SE, Rathnelly, South Hi...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>M4W</td>
      <td>Downtown Toronto</td>
      <td>Rosedale</td>
    </tr>
    <tr>
      <th>51</th>
      <td>M4X</td>
      <td>Downtown Toronto</td>
      <td>Cabbagetown, St. James Town</td>
    </tr>
    <tr>
      <th>52</th>
      <td>M4Y</td>
      <td>Downtown Toronto</td>
      <td>Church and Wellesley</td>
    </tr>
    <tr>
      <th>53</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
    </tr>
    <tr>
      <th>54</th>
      <td>M5B</td>
      <td>Downtown Toronto</td>
      <td>Ryerson, Garden District</td>
    </tr>
    <tr>
      <th>55</th>
      <td>M5C</td>
      <td>Downtown Toronto</td>
      <td>St. James Town</td>
    </tr>
    <tr>
      <th>56</th>
      <td>M5E</td>
      <td>Downtown Toronto</td>
      <td>Berczy Park</td>
    </tr>
    <tr>
      <th>57</th>
      <td>M5G</td>
      <td>Downtown Toronto</td>
      <td>Central Bay Street</td>
    </tr>
    <tr>
      <th>58</th>
      <td>M5H</td>
      <td>Downtown Toronto</td>
      <td>Adelaide, King, Richmond</td>
    </tr>
    <tr>
      <th>59</th>
      <td>M5J</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront East, Toronto Islands, Union Station</td>
    </tr>
    <tr>
      <th>60</th>
      <td>M5K</td>
      <td>Downtown Toronto</td>
      <td>Design Exchange, Toronto Dominion Centre</td>
    </tr>
    <tr>
      <th>61</th>
      <td>M5L</td>
      <td>Downtown Toronto</td>
      <td>Commerce Court, Victoria Hotel</td>
    </tr>
    <tr>
      <th>62</th>
      <td>M5M</td>
      <td>North York</td>
      <td>Bedford Park, Lawrence Manor East</td>
    </tr>
    <tr>
      <th>63</th>
      <td>M5N</td>
      <td>Central Toronto</td>
      <td>Roselawn</td>
    </tr>
    <tr>
      <th>64</th>
      <td>M5P</td>
      <td>Central Toronto</td>
      <td>Forest Hill North, Forest Hill West</td>
    </tr>
    <tr>
      <th>65</th>
      <td>M5R</td>
      <td>Central Toronto</td>
      <td>The Annex, North Midtown, Yorkville</td>
    </tr>
    <tr>
      <th>66</th>
      <td>M5S</td>
      <td>Downtown Toronto</td>
      <td>Harbord, University of Toronto</td>
    </tr>
    <tr>
      <th>67</th>
      <td>M5T</td>
      <td>Downtown Toronto</td>
      <td>Chinatown, Grange Park, Kensington Market</td>
    </tr>
    <tr>
      <th>68</th>
      <td>M5V</td>
      <td>Downtown Toronto</td>
      <td>CN Tower, Bathurst Quay, Island airport, Harbo...</td>
    </tr>
    <tr>
      <th>69</th>
      <td>M5W</td>
      <td>Downtown Toronto</td>
      <td>Stn A PO Boxes 25 The Esplanade</td>
    </tr>
    <tr>
      <th>70</th>
      <td>M5X</td>
      <td>Downtown Toronto</td>
      <td>First Canadian Place, Underground city</td>
    </tr>
    <tr>
      <th>71</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Heights, Lawrence Manor</td>
    </tr>
    <tr>
      <th>72</th>
      <td>M6B</td>
      <td>North York</td>
      <td>Glencairn</td>
    </tr>
    <tr>
      <th>73</th>
      <td>M6C</td>
      <td>York</td>
      <td>Humewood-Cedarvale</td>
    </tr>
    <tr>
      <th>74</th>
      <td>M6E</td>
      <td>York</td>
      <td>Caledonia-Fairbanks</td>
    </tr>
    <tr>
      <th>75</th>
      <td>M6G</td>
      <td>Downtown Toronto</td>
      <td>Christie</td>
    </tr>
    <tr>
      <th>76</th>
      <td>M6H</td>
      <td>West Toronto</td>
      <td>Dovercourt Village, Dufferin</td>
    </tr>
    <tr>
      <th>77</th>
      <td>M6J</td>
      <td>West Toronto</td>
      <td>Little Portugal, Trinity</td>
    </tr>
    <tr>
      <th>78</th>
      <td>M6K</td>
      <td>West Toronto</td>
      <td>Brockton, Exhibition Place, Parkdale Village</td>
    </tr>
    <tr>
      <th>79</th>
      <td>M6L</td>
      <td>North York</td>
      <td>Downsview, North Park, Upwood Park</td>
    </tr>
    <tr>
      <th>80</th>
      <td>M6M</td>
      <td>York</td>
      <td>Del Ray, Keelesdale, Mount Dennis, Silverthorn</td>
    </tr>
    <tr>
      <th>81</th>
      <td>M6N</td>
      <td>York</td>
      <td>The Junction North, Runnymede</td>
    </tr>
    <tr>
      <th>82</th>
      <td>M6P</td>
      <td>West Toronto</td>
      <td>High Park, The Junction South</td>
    </tr>
    <tr>
      <th>83</th>
      <td>M6R</td>
      <td>West Toronto</td>
      <td>Parkdale, Roncesvalles</td>
    </tr>
    <tr>
      <th>84</th>
      <td>M6S</td>
      <td>West Toronto</td>
      <td>Runnymede, Swansea</td>
    </tr>
    <tr>
      <th>85</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park</td>
    </tr>
    <tr>
      <th>86</th>
      <td>M7R</td>
      <td>Mississauga</td>
      <td>Canada Post Gateway Processing Centre</td>
    </tr>
    <tr>
      <th>87</th>
      <td>M7Y</td>
      <td>East Toronto</td>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
    </tr>
    <tr>
      <th>88</th>
      <td>M8V</td>
      <td>Etobicoke</td>
      <td>Humber Bay Shores, Mimico South, New Toronto</td>
    </tr>
    <tr>
      <th>89</th>
      <td>M8W</td>
      <td>Etobicoke</td>
      <td>Alderwood, Long Branch</td>
    </tr>
    <tr>
      <th>90</th>
      <td>M8X</td>
      <td>Etobicoke</td>
      <td>The Kingsway, Montgomery Road, Old Mill North</td>
    </tr>
    <tr>
      <th>91</th>
      <td>M8Y</td>
      <td>Etobicoke</td>
      <td>Humber Bay, King's Mill Park, Kingsway Park So...</td>
    </tr>
    <tr>
      <th>92</th>
      <td>M8Z</td>
      <td>Etobicoke</td>
      <td>Kingsway Park South West, Mimico NW, The Queen...</td>
    </tr>
    <tr>
      <th>93</th>
      <td>M9A</td>
      <td>Queen's Park</td>
      <td>Queen's Park</td>
    </tr>
    <tr>
      <th>94</th>
      <td>M9B</td>
      <td>Etobicoke</td>
      <td>Cloverdale, Islington, Martin Grove, Princess ...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>M9C</td>
      <td>Etobicoke</td>
      <td>Bloordale Gardens, Eringate, Markland Wood, Ol...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>M9L</td>
      <td>North York</td>
      <td>Humber Summit</td>
    </tr>
    <tr>
      <th>97</th>
      <td>M9M</td>
      <td>North York</td>
      <td>Emery, Humberlea</td>
    </tr>
    <tr>
      <th>98</th>
      <td>M9N</td>
      <td>York</td>
      <td>Weston</td>
    </tr>
    <tr>
      <th>99</th>
      <td>M9P</td>
      <td>Etobicoke</td>
      <td>Westmount</td>
    </tr>
    <tr>
      <th>100</th>
      <td>M9R</td>
      <td>Etobicoke</td>
      <td>Kingsview Village, Martin Grove Gardens, Richv...</td>
    </tr>
    <tr>
      <th>101</th>
      <td>M9V</td>
      <td>Etobicoke</td>
      <td>Albion Gardens, Beaumond Heights, Humbergate, ...</td>
    </tr>
    <tr>
      <th>102</th>
      <td>M9W</td>
      <td>Etobicoke</td>
      <td>Northwest</td>
    </tr>
  </tbody>
</table>
</div>



### Load geographical coordinates from the csv file


```python
coordinates = pd.read_csv("https://cocl.us/Geospatial_data")
coordinates.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postal Code</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>




```python
# rename column "PostalCode"
coordinates.rename(columns={"Postal Code": "PostalCode"}, inplace=True)
coordinates.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>



### Merge Both Dataframes to get the coordinates


```python
df_merged = df_grouped.merge(coordinates, on="PostalCode", how="left")
df_merged.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge, Malvern</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_merged
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge, Malvern</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Highland Creek, Rouge Hill, Port Union</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M1J</td>
      <td>Scarborough</td>
      <td>Scarborough Village</td>
      <td>43.744734</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M1K</td>
      <td>Scarborough</td>
      <td>East Birchmount Park, Ionview, Kennedy Park</td>
      <td>43.727929</td>
      <td>-79.262029</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M1L</td>
      <td>Scarborough</td>
      <td>Clairlea, Golden Mile, Oakridge</td>
      <td>43.711112</td>
      <td>-79.284577</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M1M</td>
      <td>Scarborough</td>
      <td>Cliffcrest, Cliffside, Scarborough Village West</td>
      <td>43.716316</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M1N</td>
      <td>Scarborough</td>
      <td>Birch Cliff, Cliffside West</td>
      <td>43.692657</td>
      <td>-79.264848</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M1P</td>
      <td>Scarborough</td>
      <td>Dorset Park, Scarborough Town Centre, Wexford ...</td>
      <td>43.757410</td>
      <td>-79.273304</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M1R</td>
      <td>Scarborough</td>
      <td>Maryvale, Wexford</td>
      <td>43.750072</td>
      <td>-79.295849</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M1S</td>
      <td>Scarborough</td>
      <td>Agincourt</td>
      <td>43.794200</td>
      <td>-79.262029</td>
    </tr>
    <tr>
      <th>13</th>
      <td>M1T</td>
      <td>Scarborough</td>
      <td>Clarks Corners, Sullivan, Tam O'Shanter</td>
      <td>43.781638</td>
      <td>-79.304302</td>
    </tr>
    <tr>
      <th>14</th>
      <td>M1V</td>
      <td>Scarborough</td>
      <td>Agincourt North, L'Amoreaux East, Milliken, St...</td>
      <td>43.815252</td>
      <td>-79.284577</td>
    </tr>
    <tr>
      <th>15</th>
      <td>M1W</td>
      <td>Scarborough</td>
      <td>L'Amoreaux West</td>
      <td>43.799525</td>
      <td>-79.318389</td>
    </tr>
    <tr>
      <th>16</th>
      <td>M1X</td>
      <td>Scarborough</td>
      <td>Upper Rouge</td>
      <td>43.836125</td>
      <td>-79.205636</td>
    </tr>
    <tr>
      <th>17</th>
      <td>M2H</td>
      <td>North York</td>
      <td>Hillcrest Village</td>
      <td>43.803762</td>
      <td>-79.363452</td>
    </tr>
    <tr>
      <th>18</th>
      <td>M2J</td>
      <td>North York</td>
      <td>Fairview, Henry Farm, Oriole</td>
      <td>43.778517</td>
      <td>-79.346556</td>
    </tr>
    <tr>
      <th>19</th>
      <td>M2K</td>
      <td>North York</td>
      <td>Bayview Village</td>
      <td>43.786947</td>
      <td>-79.385975</td>
    </tr>
    <tr>
      <th>20</th>
      <td>M2L</td>
      <td>North York</td>
      <td>Silver Hills, York Mills</td>
      <td>43.757490</td>
      <td>-79.374714</td>
    </tr>
    <tr>
      <th>21</th>
      <td>M2M</td>
      <td>North York</td>
      <td>Newtonbrook, Willowdale</td>
      <td>43.789053</td>
      <td>-79.408493</td>
    </tr>
    <tr>
      <th>22</th>
      <td>M2N</td>
      <td>North York</td>
      <td>Willowdale South</td>
      <td>43.770120</td>
      <td>-79.408493</td>
    </tr>
    <tr>
      <th>23</th>
      <td>M2P</td>
      <td>North York</td>
      <td>York Mills West</td>
      <td>43.752758</td>
      <td>-79.400049</td>
    </tr>
    <tr>
      <th>24</th>
      <td>M2R</td>
      <td>North York</td>
      <td>Willowdale West</td>
      <td>43.782736</td>
      <td>-79.442259</td>
    </tr>
    <tr>
      <th>25</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
      <td>43.753259</td>
      <td>-79.329656</td>
    </tr>
    <tr>
      <th>26</th>
      <td>M3B</td>
      <td>North York</td>
      <td>Don Mills North</td>
      <td>43.745906</td>
      <td>-79.352188</td>
    </tr>
    <tr>
      <th>27</th>
      <td>M3C</td>
      <td>North York</td>
      <td>Flemingdon Park, Don Mills South</td>
      <td>43.725900</td>
      <td>-79.340923</td>
    </tr>
    <tr>
      <th>28</th>
      <td>M3H</td>
      <td>North York</td>
      <td>Bathurst Manor, Downsview North, Wilson Heights</td>
      <td>43.754328</td>
      <td>-79.442259</td>
    </tr>
    <tr>
      <th>29</th>
      <td>M3J</td>
      <td>North York</td>
      <td>Northwood Park, York University</td>
      <td>43.767980</td>
      <td>-79.487262</td>
    </tr>
    <tr>
      <th>30</th>
      <td>M3K</td>
      <td>North York</td>
      <td>CFB Toronto, Downsview East</td>
      <td>43.737473</td>
      <td>-79.464763</td>
    </tr>
    <tr>
      <th>31</th>
      <td>M3L</td>
      <td>North York</td>
      <td>Downsview West</td>
      <td>43.739015</td>
      <td>-79.506944</td>
    </tr>
    <tr>
      <th>32</th>
      <td>M3M</td>
      <td>North York</td>
      <td>Downsview Central</td>
      <td>43.728496</td>
      <td>-79.495697</td>
    </tr>
    <tr>
      <th>33</th>
      <td>M3N</td>
      <td>North York</td>
      <td>Downsview Northwest</td>
      <td>43.761631</td>
      <td>-79.520999</td>
    </tr>
    <tr>
      <th>34</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
      <td>43.725882</td>
      <td>-79.315572</td>
    </tr>
    <tr>
      <th>35</th>
      <td>M4B</td>
      <td>East York</td>
      <td>Woodbine Gardens, Parkview Hill</td>
      <td>43.706397</td>
      <td>-79.309937</td>
    </tr>
    <tr>
      <th>36</th>
      <td>M4C</td>
      <td>East York</td>
      <td>Woodbine Heights</td>
      <td>43.695344</td>
      <td>-79.318389</td>
    </tr>
    <tr>
      <th>37</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
    </tr>
    <tr>
      <th>38</th>
      <td>M4G</td>
      <td>East York</td>
      <td>Leaside</td>
      <td>43.709060</td>
      <td>-79.363452</td>
    </tr>
    <tr>
      <th>39</th>
      <td>M4H</td>
      <td>East York</td>
      <td>Thorncliffe Park</td>
      <td>43.705369</td>
      <td>-79.349372</td>
    </tr>
    <tr>
      <th>40</th>
      <td>M4J</td>
      <td>East York</td>
      <td>East Toronto</td>
      <td>43.685347</td>
      <td>-79.338106</td>
    </tr>
    <tr>
      <th>41</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
    </tr>
    <tr>
      <th>42</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>The Beaches West, India Bazaar</td>
      <td>43.668999</td>
      <td>-79.315572</td>
    </tr>
    <tr>
      <th>43</th>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
      <td>43.659526</td>
      <td>-79.340923</td>
    </tr>
    <tr>
      <th>44</th>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
      <td>43.728020</td>
      <td>-79.388790</td>
    </tr>
    <tr>
      <th>45</th>
      <td>M4P</td>
      <td>Central Toronto</td>
      <td>Davisville North</td>
      <td>43.712751</td>
      <td>-79.390197</td>
    </tr>
    <tr>
      <th>46</th>
      <td>M4R</td>
      <td>Central Toronto</td>
      <td>North Toronto West</td>
      <td>43.715383</td>
      <td>-79.405678</td>
    </tr>
    <tr>
      <th>47</th>
      <td>M4S</td>
      <td>Central Toronto</td>
      <td>Davisville</td>
      <td>43.704324</td>
      <td>-79.388790</td>
    </tr>
    <tr>
      <th>48</th>
      <td>M4T</td>
      <td>Central Toronto</td>
      <td>Moore Park, Summerhill East</td>
      <td>43.689574</td>
      <td>-79.383160</td>
    </tr>
    <tr>
      <th>49</th>
      <td>M4V</td>
      <td>Central Toronto</td>
      <td>Deer Park, Forest Hill SE, Rathnelly, South Hi...</td>
      <td>43.686412</td>
      <td>-79.400049</td>
    </tr>
    <tr>
      <th>50</th>
      <td>M4W</td>
      <td>Downtown Toronto</td>
      <td>Rosedale</td>
      <td>43.679563</td>
      <td>-79.377529</td>
    </tr>
    <tr>
      <th>51</th>
      <td>M4X</td>
      <td>Downtown Toronto</td>
      <td>Cabbagetown, St. James Town</td>
      <td>43.667967</td>
      <td>-79.367675</td>
    </tr>
    <tr>
      <th>52</th>
      <td>M4Y</td>
      <td>Downtown Toronto</td>
      <td>Church and Wellesley</td>
      <td>43.665860</td>
      <td>-79.383160</td>
    </tr>
    <tr>
      <th>53</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
      <td>43.654260</td>
      <td>-79.360636</td>
    </tr>
    <tr>
      <th>54</th>
      <td>M5B</td>
      <td>Downtown Toronto</td>
      <td>Ryerson, Garden District</td>
      <td>43.657162</td>
      <td>-79.378937</td>
    </tr>
    <tr>
      <th>55</th>
      <td>M5C</td>
      <td>Downtown Toronto</td>
      <td>St. James Town</td>
      <td>43.651494</td>
      <td>-79.375418</td>
    </tr>
    <tr>
      <th>56</th>
      <td>M5E</td>
      <td>Downtown Toronto</td>
      <td>Berczy Park</td>
      <td>43.644771</td>
      <td>-79.373306</td>
    </tr>
    <tr>
      <th>57</th>
      <td>M5G</td>
      <td>Downtown Toronto</td>
      <td>Central Bay Street</td>
      <td>43.657952</td>
      <td>-79.387383</td>
    </tr>
    <tr>
      <th>58</th>
      <td>M5H</td>
      <td>Downtown Toronto</td>
      <td>Adelaide, King, Richmond</td>
      <td>43.650571</td>
      <td>-79.384568</td>
    </tr>
    <tr>
      <th>59</th>
      <td>M5J</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront East, Toronto Islands, Union Station</td>
      <td>43.640816</td>
      <td>-79.381752</td>
    </tr>
    <tr>
      <th>60</th>
      <td>M5K</td>
      <td>Downtown Toronto</td>
      <td>Design Exchange, Toronto Dominion Centre</td>
      <td>43.647177</td>
      <td>-79.381576</td>
    </tr>
    <tr>
      <th>61</th>
      <td>M5L</td>
      <td>Downtown Toronto</td>
      <td>Commerce Court, Victoria Hotel</td>
      <td>43.648198</td>
      <td>-79.379817</td>
    </tr>
    <tr>
      <th>62</th>
      <td>M5M</td>
      <td>North York</td>
      <td>Bedford Park, Lawrence Manor East</td>
      <td>43.733283</td>
      <td>-79.419750</td>
    </tr>
    <tr>
      <th>63</th>
      <td>M5N</td>
      <td>Central Toronto</td>
      <td>Roselawn</td>
      <td>43.711695</td>
      <td>-79.416936</td>
    </tr>
    <tr>
      <th>64</th>
      <td>M5P</td>
      <td>Central Toronto</td>
      <td>Forest Hill North, Forest Hill West</td>
      <td>43.696948</td>
      <td>-79.411307</td>
    </tr>
    <tr>
      <th>65</th>
      <td>M5R</td>
      <td>Central Toronto</td>
      <td>The Annex, North Midtown, Yorkville</td>
      <td>43.672710</td>
      <td>-79.405678</td>
    </tr>
    <tr>
      <th>66</th>
      <td>M5S</td>
      <td>Downtown Toronto</td>
      <td>Harbord, University of Toronto</td>
      <td>43.662696</td>
      <td>-79.400049</td>
    </tr>
    <tr>
      <th>67</th>
      <td>M5T</td>
      <td>Downtown Toronto</td>
      <td>Chinatown, Grange Park, Kensington Market</td>
      <td>43.653206</td>
      <td>-79.400049</td>
    </tr>
    <tr>
      <th>68</th>
      <td>M5V</td>
      <td>Downtown Toronto</td>
      <td>CN Tower, Bathurst Quay, Island airport, Harbo...</td>
      <td>43.628947</td>
      <td>-79.394420</td>
    </tr>
    <tr>
      <th>69</th>
      <td>M5W</td>
      <td>Downtown Toronto</td>
      <td>Stn A PO Boxes 25 The Esplanade</td>
      <td>43.646435</td>
      <td>-79.374846</td>
    </tr>
    <tr>
      <th>70</th>
      <td>M5X</td>
      <td>Downtown Toronto</td>
      <td>First Canadian Place, Underground city</td>
      <td>43.648429</td>
      <td>-79.382280</td>
    </tr>
    <tr>
      <th>71</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Heights, Lawrence Manor</td>
      <td>43.718518</td>
      <td>-79.464763</td>
    </tr>
    <tr>
      <th>72</th>
      <td>M6B</td>
      <td>North York</td>
      <td>Glencairn</td>
      <td>43.709577</td>
      <td>-79.445073</td>
    </tr>
    <tr>
      <th>73</th>
      <td>M6C</td>
      <td>York</td>
      <td>Humewood-Cedarvale</td>
      <td>43.693781</td>
      <td>-79.428191</td>
    </tr>
    <tr>
      <th>74</th>
      <td>M6E</td>
      <td>York</td>
      <td>Caledonia-Fairbanks</td>
      <td>43.689026</td>
      <td>-79.453512</td>
    </tr>
    <tr>
      <th>75</th>
      <td>M6G</td>
      <td>Downtown Toronto</td>
      <td>Christie</td>
      <td>43.669542</td>
      <td>-79.422564</td>
    </tr>
    <tr>
      <th>76</th>
      <td>M6H</td>
      <td>West Toronto</td>
      <td>Dovercourt Village, Dufferin</td>
      <td>43.669005</td>
      <td>-79.442259</td>
    </tr>
    <tr>
      <th>77</th>
      <td>M6J</td>
      <td>West Toronto</td>
      <td>Little Portugal, Trinity</td>
      <td>43.647927</td>
      <td>-79.419750</td>
    </tr>
    <tr>
      <th>78</th>
      <td>M6K</td>
      <td>West Toronto</td>
      <td>Brockton, Exhibition Place, Parkdale Village</td>
      <td>43.636847</td>
      <td>-79.428191</td>
    </tr>
    <tr>
      <th>79</th>
      <td>M6L</td>
      <td>North York</td>
      <td>Downsview, North Park, Upwood Park</td>
      <td>43.713756</td>
      <td>-79.490074</td>
    </tr>
    <tr>
      <th>80</th>
      <td>M6M</td>
      <td>York</td>
      <td>Del Ray, Keelesdale, Mount Dennis, Silverthorn</td>
      <td>43.691116</td>
      <td>-79.476013</td>
    </tr>
    <tr>
      <th>81</th>
      <td>M6N</td>
      <td>York</td>
      <td>The Junction North, Runnymede</td>
      <td>43.673185</td>
      <td>-79.487262</td>
    </tr>
    <tr>
      <th>82</th>
      <td>M6P</td>
      <td>West Toronto</td>
      <td>High Park, The Junction South</td>
      <td>43.661608</td>
      <td>-79.464763</td>
    </tr>
    <tr>
      <th>83</th>
      <td>M6R</td>
      <td>West Toronto</td>
      <td>Parkdale, Roncesvalles</td>
      <td>43.648960</td>
      <td>-79.456325</td>
    </tr>
    <tr>
      <th>84</th>
      <td>M6S</td>
      <td>West Toronto</td>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
    </tr>
    <tr>
      <th>85</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park</td>
      <td>43.662301</td>
      <td>-79.389494</td>
    </tr>
    <tr>
      <th>86</th>
      <td>M7R</td>
      <td>Mississauga</td>
      <td>Canada Post Gateway Processing Centre</td>
      <td>43.636966</td>
      <td>-79.615819</td>
    </tr>
    <tr>
      <th>87</th>
      <td>M7Y</td>
      <td>East Toronto</td>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
    </tr>
    <tr>
      <th>88</th>
      <td>M8V</td>
      <td>Etobicoke</td>
      <td>Humber Bay Shores, Mimico South, New Toronto</td>
      <td>43.605647</td>
      <td>-79.501321</td>
    </tr>
    <tr>
      <th>89</th>
      <td>M8W</td>
      <td>Etobicoke</td>
      <td>Alderwood, Long Branch</td>
      <td>43.602414</td>
      <td>-79.543484</td>
    </tr>
    <tr>
      <th>90</th>
      <td>M8X</td>
      <td>Etobicoke</td>
      <td>The Kingsway, Montgomery Road, Old Mill North</td>
      <td>43.653654</td>
      <td>-79.506944</td>
    </tr>
    <tr>
      <th>91</th>
      <td>M8Y</td>
      <td>Etobicoke</td>
      <td>Humber Bay, King's Mill Park, Kingsway Park So...</td>
      <td>43.636258</td>
      <td>-79.498509</td>
    </tr>
    <tr>
      <th>92</th>
      <td>M8Z</td>
      <td>Etobicoke</td>
      <td>Kingsway Park South West, Mimico NW, The Queen...</td>
      <td>43.628841</td>
      <td>-79.520999</td>
    </tr>
    <tr>
      <th>93</th>
      <td>M9A</td>
      <td>Queen's Park</td>
      <td>Queen's Park</td>
      <td>43.667856</td>
      <td>-79.532242</td>
    </tr>
    <tr>
      <th>94</th>
      <td>M9B</td>
      <td>Etobicoke</td>
      <td>Cloverdale, Islington, Martin Grove, Princess ...</td>
      <td>43.650943</td>
      <td>-79.554724</td>
    </tr>
    <tr>
      <th>95</th>
      <td>M9C</td>
      <td>Etobicoke</td>
      <td>Bloordale Gardens, Eringate, Markland Wood, Ol...</td>
      <td>43.643515</td>
      <td>-79.577201</td>
    </tr>
    <tr>
      <th>96</th>
      <td>M9L</td>
      <td>North York</td>
      <td>Humber Summit</td>
      <td>43.756303</td>
      <td>-79.565963</td>
    </tr>
    <tr>
      <th>97</th>
      <td>M9M</td>
      <td>North York</td>
      <td>Emery, Humberlea</td>
      <td>43.724766</td>
      <td>-79.532242</td>
    </tr>
    <tr>
      <th>98</th>
      <td>M9N</td>
      <td>York</td>
      <td>Weston</td>
      <td>43.706876</td>
      <td>-79.518188</td>
    </tr>
    <tr>
      <th>99</th>
      <td>M9P</td>
      <td>Etobicoke</td>
      <td>Westmount</td>
      <td>43.696319</td>
      <td>-79.532242</td>
    </tr>
    <tr>
      <th>100</th>
      <td>M9R</td>
      <td>Etobicoke</td>
      <td>Kingsview Village, Martin Grove Gardens, Richv...</td>
      <td>43.688905</td>
      <td>-79.554724</td>
    </tr>
    <tr>
      <th>101</th>
      <td>M9V</td>
      <td>Etobicoke</td>
      <td>Albion Gardens, Beaumond Heights, Humbergate, ...</td>
      <td>43.739416</td>
      <td>-79.588437</td>
    </tr>
    <tr>
      <th>102</th>
      <td>M9W</td>
      <td>Etobicoke</td>
      <td>Northwest</td>
      <td>43.706748</td>
      <td>-79.594054</td>
    </tr>
  </tbody>
</table>
</div>



### Use geopy library to get the latitude and longitude values of Toronto


```python
address = 'Toronto'

geolocator = Nominatim(user_agent="my-application")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))
```

    The geograpical coordinate of Toronto are 43.653963, -79.387207.


### Create a map of Toronto with neighborhoods superimposed on top


```python
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df_merged['Latitude'], df_merged['Longitude'], df_merged['Borough'], df_merged['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7).add_to(map_toronto)  
    
map_toronto 
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1IiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNScsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjUzOTYzLC03OS4zODcyMDddLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgem9vbTogMTAsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYXhCb3VuZHM6IGJvdW5kcywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGxheWVyczogW10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3b3JsZENvcHlKdW1wOiBmYWxzZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSk7CiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyX2QzYmI0MWZjMmUxMzQ3ZGViZjk2OTY3YmI4NzE5ZjZmID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgICAgICAgICAgewogICJhdHRyaWJ1dGlvbiI6IG51bGwsCiAgImRldGVjdFJldGluYSI6IGZhbHNlLAogICJtYXhab29tIjogMTgsCiAgIm1pblpvb20iOiAxLAogICJub1dyYXAiOiBmYWxzZSwKICAic3ViZG9tYWlucyI6ICJhYmMiCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iMTdiNDg1OTNkNDI0NzQ2OGQ1MGI4OWVjNjVlOGRlYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjgwNjY4NjI5OTk5OTk5NiwtNzkuMTk0MzUzNDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTM1MTI0Zjg4OGY0NDgxNmI3NjU2ZDM1ZmY1YzFjOWEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzczYmYxNTc4MjVmNDA0Y2IxYWM5NjQwZDlhMDE5YzIgPSAkKCc8ZGl2IGlkPSJodG1sX2M3M2JmMTU3ODI1ZjQwNGNiMWFjOTY0MGQ5YTAxOWMyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb3VnZSwgTWFsdmVybiwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2UzNTEyNGY4ODhmNDQ4MTZiNzY1NmQzNWZmNWMxYzlhLnNldENvbnRlbnQoaHRtbF9jNzNiZjE1NzgyNWY0MDRjYjFhYzk2NDBkOWEwMTljMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iMTdiNDg1OTNkNDI0NzQ2OGQ1MGI4OWVjNjVlOGRlYy5iaW5kUG9wdXAocG9wdXBfZTM1MTI0Zjg4OGY0NDgxNmI3NjU2ZDM1ZmY1YzFjOWEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjcyN2JlMmYyYmUwNGYzYmJiMzYxZGQ1ZTI3ODAxNjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODQ1MzUxLC03OS4xNjA0OTcwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kZTcwOWVjZTU5NDA0MmI1YTc2ZDEzYTA1YThlNThlNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81M2Y0MzZiZGRiNDY0MzQ2ODQ2NjY3OWQ5MmY2OWJmNyA9ICQoJzxkaXYgaWQ9Imh0bWxfNTNmNDM2YmRkYjQ2NDM0Njg0NjY2NzlkOTJmNjliZjciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhpZ2hsYW5kIENyZWVrLCBSb3VnZSBIaWxsLCBQb3J0IFVuaW9uLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGU3MDllY2U1OTQwNDJiNWE3NmQxM2EwNWE4ZTU4ZTUuc2V0Q29udGVudChodG1sXzUzZjQzNmJkZGI0NjQzNDY4NDY2Njc5ZDkyZjY5YmY3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzY3MjdiZTJmMmJlMDRmM2JiYjM2MWRkNWUyNzgwMTYyLmJpbmRQb3B1cChwb3B1cF9kZTcwOWVjZTU5NDA0MmI1YTc2ZDEzYTA1YThlNThlNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82OGRlZjJiYmE0MjU0NTA4YmE4NWE3MmVjZmRlMjZhNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc2MzU3MjYsLTc5LjE4ODcxMTVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGM2YTQ5M2I0NDE3NGY4YmFiNTNhNDczN2ViMTc3YWEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzNmM2ZlZmYwZTY1NGQ4NmI2NzgwY2QyZThkNWY5ZTYgPSAkKCc8ZGl2IGlkPSJodG1sXzMzZjNmZWZmMGU2NTRkODZiNjc4MGNkMmU4ZDVmOWU2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HdWlsZHdvb2QsIE1vcm5pbmdzaWRlLCBXZXN0IEhpbGwsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80YzZhNDkzYjQ0MTc0ZjhiYWI1M2E0NzM3ZWIxNzdhYS5zZXRDb250ZW50KGh0bWxfMzNmM2ZlZmYwZTY1NGQ4NmI2NzgwY2QyZThkNWY5ZTYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjhkZWYyYmJhNDI1NDUwOGJhODVhNzJlY2ZkZTI2YTUuYmluZFBvcHVwKHBvcHVwXzRjNmE0OTNiNDQxNzRmOGJhYjUzYTQ3MzdlYjE3N2FhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Q4N2E0OWQxNzZmNDQwNzk4NDRiOTI1Yjc5OTFlYzA0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzcwOTkyMSwtNzkuMjE2OTE3NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTFlN2M0NTBlYjcyNDIyMDk4MGZiYTQ4M2U2NTMwZjEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODUxYzdhZmY1YTM1NGMzMTkyZTRmZTg1YzY5ODBjYTMgPSAkKCc8ZGl2IGlkPSJodG1sXzg1MWM3YWZmNWEzNTRjMzE5MmU0ZmU4NWM2OTgwY2EzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Xb2J1cm4sIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xMWU3YzQ1MGViNzI0MjIwOTgwZmJhNDgzZTY1MzBmMS5zZXRDb250ZW50KGh0bWxfODUxYzdhZmY1YTM1NGMzMTkyZTRmZTg1YzY5ODBjYTMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDg3YTQ5ZDE3NmY0NDA3OTg0NGI5MjViNzk5MWVjMDQuYmluZFBvcHVwKHBvcHVwXzExZTdjNDUwZWI3MjQyMjA5ODBmYmE0ODNlNjUzMGYxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ExNTc2NTFkNzU3ZTRiMTM5ZTVlYzlmOTU3YWJjNTZlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzczMTM2LC03OS4yMzk0NzYwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kNWU1NGJkYzNkYjU0OThiOWY3NTY3ZDIzYmMzOGI1NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kN2ZhNDg5NTljMDI0YzJkYWEyYjkwNjFkNmY2NWE0YiA9ICQoJzxkaXYgaWQ9Imh0bWxfZDdmYTQ4OTU5YzAyNGMyZGFhMmI5MDYxZDZmNjVhNGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlZGFyYnJhZSwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q1ZTU0YmRjM2RiNTQ5OGI5Zjc1NjdkMjNiYzM4YjU1LnNldENvbnRlbnQoaHRtbF9kN2ZhNDg5NTljMDI0YzJkYWEyYjkwNjFkNmY2NWE0Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMTU3NjUxZDc1N2U0YjEzOWU1ZWM5Zjk1N2FiYzU2ZS5iaW5kUG9wdXAocG9wdXBfZDVlNTRiZGMzZGI1NDk4YjlmNzU2N2QyM2JjMzhiNTUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWQ3YjBhYWM5NGY4NDg0YzllMDA4OTMyODdiYzc2NmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NDQ3MzQyLC03OS4yMzk0NzYwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85ZDU5YmQ4NGU2ZTg0OGY4OWE0MDdkNTE2ODM2YjRlMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xOTBiMmFkOWQ5N2Y0ODJkOWYxYWRlYTI1MWI2MTEwMSA9ICQoJzxkaXYgaWQ9Imh0bWxfMTkwYjJhZDlkOTdmNDgyZDlmMWFkZWEyNTFiNjExMDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNjYXJib3JvdWdoIFZpbGxhZ2UsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85ZDU5YmQ4NGU2ZTg0OGY4OWE0MDdkNTE2ODM2YjRlMS5zZXRDb250ZW50KGh0bWxfMTkwYjJhZDlkOTdmNDgyZDlmMWFkZWEyNTFiNjExMDEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMWQ3YjBhYWM5NGY4NDg0YzllMDA4OTMyODdiYzc2NmMuYmluZFBvcHVwKHBvcHVwXzlkNTliZDg0ZTZlODQ4Zjg5YTQwN2Q1MTY4MzZiNGUxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzdiZjZhMGI2ZWNjZjQ4MTliZmQ5N2FmOGExMzU2ZWE5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI3OTI5MiwtNzkuMjYyMDI5NDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTk3ZGZhYzY2MTIyNDdjMjhhYWI2NmVkZDUyYTAwZjggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMThmN2JlNDc0ZDA0NGU2ZTkzNjUxNDBhYzdiNjIyZDcgPSAkKCc8ZGl2IGlkPSJodG1sXzE4ZjdiZTQ3NGQwNDRlNmU5MzY1MTQwYWM3YjYyMmQ3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FYXN0IEJpcmNobW91bnQgUGFyaywgSW9udmlldywgS2VubmVkeSBQYXJrLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTk3ZGZhYzY2MTIyNDdjMjhhYWI2NmVkZDUyYTAwZjguc2V0Q29udGVudChodG1sXzE4ZjdiZTQ3NGQwNDRlNmU5MzY1MTQwYWM3YjYyMmQ3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdiZjZhMGI2ZWNjZjQ4MTliZmQ5N2FmOGExMzU2ZWE5LmJpbmRQb3B1cChwb3B1cF85OTdkZmFjNjYxMjI0N2MyOGFhYjY2ZWRkNTJhMDBmOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jNGM3ZTZjMzk1YmE0MTM5YjhjOWY2YzcyYTgzZGUxMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMTExMTcwMDAwMDAwNCwtNzkuMjg0NTc3Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82NjQzM2M4YjYzMDM0MTljYWYwOGIwZTAyZWRiNTE1NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jYjk1ZjY1YzRlOTg0YjQ3ODhhYWM1NTA4MmVhN2ZkMiA9ICQoJzxkaXYgaWQ9Imh0bWxfY2I5NWY2NWM0ZTk4NGI0Nzg4YWFjNTUwODJlYTdmZDIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNsYWlybGVhLCBHb2xkZW4gTWlsZSwgT2FrcmlkZ2UsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82NjQzM2M4YjYzMDM0MTljYWYwOGIwZTAyZWRiNTE1NS5zZXRDb250ZW50KGh0bWxfY2I5NWY2NWM0ZTk4NGI0Nzg4YWFjNTUwODJlYTdmZDIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzRjN2U2YzM5NWJhNDEzOWI4YzlmNmM3MmE4M2RlMTAuYmluZFBvcHVwKHBvcHVwXzY2NDMzYzhiNjMwMzQxOWNhZjA4YjBlMDJlZGI1MTU1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M1Mzc1MjNmNzRiZDRhNjI5NTU4M2IyZjFhZDYyZjYzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzE2MzE2LC03OS4yMzk0NzYwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMjdjZjBiMzU0NWY0OWJhYjEzZDlmMjk2Y2M0OTgyNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zOWZmMDg2Y2I3MDk0MjlmODFkNjgzNTc3MjE5MDkxMiA9ICQoJzxkaXYgaWQ9Imh0bWxfMzlmZjA4NmNiNzA5NDI5ZjgxZDY4MzU3NzIxOTA5MTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNsaWZmY3Jlc3QsIENsaWZmc2lkZSwgU2NhcmJvcm91Z2ggVmlsbGFnZSBXZXN0LCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTI3Y2YwYjM1NDVmNDliYWIxM2Q5ZjI5NmNjNDk4MjYuc2V0Q29udGVudChodG1sXzM5ZmYwODZjYjcwOTQyOWY4MWQ2ODM1NzcyMTkwOTEyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M1Mzc1MjNmNzRiZDRhNjI5NTU4M2IyZjFhZDYyZjYzLmJpbmRQb3B1cChwb3B1cF9lMjdjZjBiMzU0NWY0OWJhYjEzZDlmMjk2Y2M0OTgyNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82N2MzOTRlZTk3M2E0YmQ3YmJhM2FiYmNhOGMyYjA3YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5MjY1NzAwMDAwMDAwNCwtNzkuMjY0ODQ4MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85ZGY1MWZjYzgwMTQ0OThhOTAyM2M5NmZmOTUzZTFjOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jZTMyZjA0YzJmZGU0ZDhmODAzOTE5YTljNzg5OTc5MCA9ICQoJzxkaXYgaWQ9Imh0bWxfY2UzMmYwNGMyZmRlNGQ4ZjgwMzkxOWE5Yzc4OTk3OTAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJpcmNoIENsaWZmLCBDbGlmZnNpZGUgV2VzdCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzlkZjUxZmNjODAxNDQ5OGE5MDIzYzk2ZmY5NTNlMWM5LnNldENvbnRlbnQoaHRtbF9jZTMyZjA0YzJmZGU0ZDhmODAzOTE5YTljNzg5OTc5MCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82N2MzOTRlZTk3M2E0YmQ3YmJhM2FiYmNhOGMyYjA3Yy5iaW5kUG9wdXAocG9wdXBfOWRmNTFmY2M4MDE0NDk4YTkwMjNjOTZmZjk1M2UxYzkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTRmN2YzOGY0YjU0NDBmOGFjN2EwYzNiMDQ5YmQxNTIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTc0MDk2LC03OS4yNzMzMDQwMDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yYzBiN2ZlYjcwMjA0NDU1YmVhMTUwNTQ5NjAyNDYzMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84MGYwMWUxN2JhMTQ0ZWVlOGU3Yzg4NTBhZDBlOWU4ZSA9ICQoJzxkaXYgaWQ9Imh0bWxfODBmMDFlMTdiYTE0NGVlZThlN2M4ODUwYWQwZTllOGUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvcnNldCBQYXJrLCBTY2FyYm9yb3VnaCBUb3duIENlbnRyZSwgV2V4Zm9yZCBIZWlnaHRzLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMmMwYjdmZWI3MDIwNDQ1NWJlYTE1MDU0OTYwMjQ2MzIuc2V0Q29udGVudChodG1sXzgwZjAxZTE3YmExNDRlZWU4ZTdjODg1MGFkMGU5ZThlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE0ZjdmMzhmNGI1NDQwZjhhYzdhMGMzYjA0OWJkMTUyLmJpbmRQb3B1cChwb3B1cF8yYzBiN2ZlYjcwMjA0NDU1YmVhMTUwNTQ5NjAyNDYzMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hYjE4YWMyNWM1OTI0NDVhOWU5NjYyNzNlYTFhMTA3OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc1MDA3MTUwMDAwMDAwNCwtNzkuMjk1ODQ5MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yMGVmMGE2ZjUwODk0ODBmYWMwOTFiZWY3NmJlMjUwYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84NzViYzA0ODIxNzc0ODM4YmFlZjcxNTMyZmRkZWJmNyA9ICQoJzxkaXYgaWQ9Imh0bWxfODc1YmMwNDgyMTc3NDgzOGJhZWY3MTUzMmZkZGViZjciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1hcnl2YWxlLCBXZXhmb3JkLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjBlZjBhNmY1MDg5NDgwZmFjMDkxYmVmNzZiZTI1MGMuc2V0Q29udGVudChodG1sXzg3NWJjMDQ4MjE3NzQ4MzhiYWVmNzE1MzJmZGRlYmY3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FiMThhYzI1YzU5MjQ0NWE5ZTk2NjI3M2VhMWExMDc4LmJpbmRQb3B1cChwb3B1cF8yMGVmMGE2ZjUwODk0ODBmYWMwOTFiZWY3NmJlMjUwYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81OGZjYjA2YjRhZGQ0ODI5ODQ0ZTE5ZDFkYzNjNzIyNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc5NDIwMDMsLTc5LjI2MjAyOTQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzY4ZGFjNTE0NDliZDRjZDRiMmVlZWY5Zjc4ZDA5N2VlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y4NzU2YTg2ZDM5OTRjMTk5YjFmZmViZGUyYWIxZmEwID0gJCgnPGRpdiBpZD0iaHRtbF9mODc1NmE4NmQzOTk0YzE5OWIxZmZlYmRlMmFiMWZhMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QWdpbmNvdXJ0LCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjhkYWM1MTQ0OWJkNGNkNGIyZWVlZjlmNzhkMDk3ZWUuc2V0Q29udGVudChodG1sX2Y4NzU2YTg2ZDM5OTRjMTk5YjFmZmViZGUyYWIxZmEwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU4ZmNiMDZiNGFkZDQ4Mjk4NDRlMTlkMWRjM2M3MjI0LmJpbmRQb3B1cChwb3B1cF82OGRhYzUxNDQ5YmQ0Y2Q0YjJlZWVmOWY3OGQwOTdlZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84YzYzN2UxODhhOGM0ZGFjOTY1ZTNhMzQ5MjFlMzIxYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4MTYzNzUsLTc5LjMwNDMwMjFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNWIzYzg4NTA3OGRjNDNjMGE2YzExYThhMmI0MTgyOGIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYWUyMDZiMzA0NzcyNGNkYWFhYzdhMWQ0OGYxNGM5NjYgPSAkKCc8ZGl2IGlkPSJodG1sX2FlMjA2YjMwNDc3MjRjZGFhYWM3YTFkNDhmMTRjOTY2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DbGFya3MgQ29ybmVycywgU3VsbGl2YW4sIFRhbSBPJiMzOTtTaGFudGVyLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNWIzYzg4NTA3OGRjNDNjMGE2YzExYThhMmI0MTgyOGIuc2V0Q29udGVudChodG1sX2FlMjA2YjMwNDc3MjRjZGFhYWM3YTFkNDhmMTRjOTY2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzhjNjM3ZTE4OGE4YzRkYWM5NjVlM2EzNDkyMWUzMjFhLmJpbmRQb3B1cChwb3B1cF81YjNjODg1MDc4ZGM0M2MwYTZjMTFhOGEyYjQxODI4Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82ZjE2NzExZGY3NjE0ODg4YTlhZDg3MDI3ODBhOTdmMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjgxNTI1MjIsLTc5LjI4NDU3NzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTAwZTI0MDNiMTZjNGE1YjkxN2ViMmE5NDFkMjJkOGMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTA1ZTI1ZGZiZjJlNGE5N2EwYzJlNDAzY2UyNTk5OGYgPSAkKCc8ZGl2IGlkPSJodG1sX2EwNWUyNWRmYmYyZTRhOTdhMGMyZTQwM2NlMjU5OThmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BZ2luY291cnQgTm9ydGgsIEwmIzM5O0Ftb3JlYXV4IEVhc3QsIE1pbGxpa2VuLCBTdGVlbGVzIEVhc3QsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81MDBlMjQwM2IxNmM0YTViOTE3ZWIyYTk0MWQyMmQ4Yy5zZXRDb250ZW50KGh0bWxfYTA1ZTI1ZGZiZjJlNGE5N2EwYzJlNDAzY2UyNTk5OGYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNmYxNjcxMWRmNzYxNDg4OGE5YWQ4NzAyNzgwYTk3ZjIuYmluZFBvcHVwKHBvcHVwXzUwMGUyNDAzYjE2YzRhNWI5MTdlYjJhOTQxZDIyZDhjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzAyMTBiMmU5NzZlNTQ5OWU5ZDkxZTQ0ZGUxMDBkZTc3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzk5NTI1MjAwMDAwMDA1LC03OS4zMTgzODg3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzAxNTBlOTQwMGEzOTQ5M2I4YjQ4NGQzYmM5ZTI1OWYyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk3YjRlMmIyN2UwNTRmZGY4YTNhMTBhNzg3N2RkZjIwID0gJCgnPGRpdiBpZD0iaHRtbF85N2I0ZTJiMjdlMDU0ZmRmOGEzYTEwYTc4NzdkZGYyMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TCYjMzk7QW1vcmVhdXggV2VzdCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzAxNTBlOTQwMGEzOTQ5M2I4YjQ4NGQzYmM5ZTI1OWYyLnNldENvbnRlbnQoaHRtbF85N2I0ZTJiMjdlMDU0ZmRmOGEzYTEwYTc4NzdkZGYyMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wMjEwYjJlOTc2ZTU0OTllOWQ5MWU0NGRlMTAwZGU3Ny5iaW5kUG9wdXAocG9wdXBfMDE1MGU5NDAwYTM5NDkzYjhiNDg0ZDNiYzllMjU5ZjIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzM0NDVlMjUxMjRjNGM4ODk1MmE5ODk2Yjc0YzhmOTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My44MzYxMjQ3MDAwMDAwMDYsLTc5LjIwNTYzNjA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2VhYzk5YmQ0ZTg3ODQwYTNhYTJkMzk5NmJkMThmNTYxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RlYWY1Yjk0ZWVlMjRmNDJhYjIyZjlmMWM1YTE2OWNmID0gJCgnPGRpdiBpZD0iaHRtbF9kZWFmNWI5NGVlZTI0ZjQyYWIyMmY5ZjFjNWExNjljZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VXBwZXIgUm91Z2UsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lYWM5OWJkNGU4Nzg0MGEzYWEyZDM5OTZiZDE4ZjU2MS5zZXRDb250ZW50KGh0bWxfZGVhZjViOTRlZWUyNGY0MmFiMjJmOWYxYzVhMTY5Y2YpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzM0NDVlMjUxMjRjNGM4ODk1MmE5ODk2Yjc0YzhmOTAuYmluZFBvcHVwKHBvcHVwX2VhYzk5YmQ0ZTg3ODQwYTNhYTJkMzk5NmJkMThmNTYxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJkM2ZjNDY3ODMxYjRiZTc5Zjg2ODg1MjlmZjRlYmQ1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuODAzNzYyMiwtNzkuMzYzNDUxN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81NWY1ZWVkMzIzY2I0MGVlYTliODQzNjIyNTUzM2E1NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85NGY2MTc5Zjk2MDg0NWJhYjAyMGJjYzFiZTc3NGU4ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfOTRmNjE3OWY5NjA4NDViYWIwMjBiY2MxYmU3NzRlOGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhpbGxjcmVzdCBWaWxsYWdlLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81NWY1ZWVkMzIzY2I0MGVlYTliODQzNjIyNTUzM2E1Ny5zZXRDb250ZW50KGh0bWxfOTRmNjE3OWY5NjA4NDViYWIwMjBiY2MxYmU3NzRlOGQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmQzZmM0Njc4MzFiNGJlNzlmODY4ODUyOWZmNGViZDUuYmluZFBvcHVwKHBvcHVwXzU1ZjVlZWQzMjNjYjQwZWVhOWI4NDM2MjI1NTMzYTU3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE3MGI2MzgxYzhkODRjYzNhZDczNDM3NWY5M2MyNDAxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzc4NTE3NSwtNzkuMzQ2NTU1N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kMTc2NzQ4Njc0ZTA0YTcwODg3ZDE3OGQ3YjNjYjllMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hMzkzZjMxNDA4ZTU0NzUyYWM4YTU0YWVkYmEwZmViMiA9ICQoJzxkaXYgaWQ9Imh0bWxfYTM5M2YzMTQwOGU1NDc1MmFjOGE1NGFlZGJhMGZlYjIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZhaXJ2aWV3LCBIZW5yeSBGYXJtLCBPcmlvbGUsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2QxNzY3NDg2NzRlMDRhNzA4ODdkMTc4ZDdiM2NiOWUyLnNldENvbnRlbnQoaHRtbF9hMzkzZjMxNDA4ZTU0NzUyYWM4YTU0YWVkYmEwZmViMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xNzBiNjM4MWM4ZDg0Y2MzYWQ3MzQzNzVmOTNjMjQwMS5iaW5kUG9wdXAocG9wdXBfZDE3Njc0ODY3NGUwNGE3MDg4N2QxNzhkN2IzY2I5ZTIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjA5OGMwYWYzNmFkNDFmNWFhMGU1NzA4NGE4ODhmMjYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODY5NDczLC03OS4zODU5NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTM0ZWFjZTJkYWE3NDM0OTlmMzUwZTUxZTU0NjE3Y2MgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDgzZTE4YmViN2VmNDUxMWE2NjBiYzQzY2RkMmUxMWUgPSAkKCc8ZGl2IGlkPSJodG1sXzQ4M2UxOGJlYjdlZjQ1MTFhNjYwYmM0M2NkZDJlMTFlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CYXl2aWV3IFZpbGxhZ2UsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzUzNGVhY2UyZGFhNzQzNDk5ZjM1MGU1MWU1NDYxN2NjLnNldENvbnRlbnQoaHRtbF80ODNlMThiZWI3ZWY0NTExYTY2MGJjNDNjZGQyZTExZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mMDk4YzBhZjM2YWQ0MWY1YWEwZTU3MDg0YTg4OGYyNi5iaW5kUG9wdXAocG9wdXBfNTM0ZWFjZTJkYWE3NDM0OTlmMzUwZTUxZTU0NjE3Y2MpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWNkMWFlMDVkMjJkNDlmMTgyYmM2NDM3YjFiY2I4NjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTc0OTAyLC03OS4zNzQ3MTQwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lZjVhMGE1NmFkYzY0ZTQ4OGFjNzZiMTNiNTA2OTQ0NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85OGJlZjUwMzgxODQ0Y2QzOTFmNTIxYzg0ZjM1Yzg0MyA9ICQoJzxkaXYgaWQ9Imh0bWxfOThiZWY1MDM4MTg0NGNkMzkxZjUyMWM4NGYzNWM4NDMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNpbHZlciBIaWxscywgWW9yayBNaWxscywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZWY1YTBhNTZhZGM2NGU0ODhhYzc2YjEzYjUwNjk0NDUuc2V0Q29udGVudChodG1sXzk4YmVmNTAzODE4NDRjZDM5MWY1MjFjODRmMzVjODQzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FjZDFhZTA1ZDIyZDQ5ZjE4MmJjNjQzN2IxYmNiODY4LmJpbmRQb3B1cChwb3B1cF9lZjVhMGE1NmFkYzY0ZTQ4OGFjNzZiMTNiNTA2OTQ0NSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81MzA0MDhmNzM0ZWM0ZmYyYmY3ZjNkOWQ5NzRlYTMzMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4OTA1MywtNzkuNDA4NDkyNzk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjA5ZWFjNjQ0ODBiNDZhMjg1NmIzYWU4MWVkMDc2ZjAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMThlZTU0OGZjNjk3NDU5NGJiYjVjZjI4MjUxMmI5ZjQgPSAkKCc8ZGl2IGlkPSJodG1sXzE4ZWU1NDhmYzY5NzQ1OTRiYmI1Y2YyODI1MTJiOWY0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5OZXd0b25icm9vaywgV2lsbG93ZGFsZSwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjA5ZWFjNjQ0ODBiNDZhMjg1NmIzYWU4MWVkMDc2ZjAuc2V0Q29udGVudChodG1sXzE4ZWU1NDhmYzY5NzQ1OTRiYmI1Y2YyODI1MTJiOWY0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzUzMDQwOGY3MzRlYzRmZjJiZjdmM2Q5ZDk3NGVhMzMyLmJpbmRQb3B1cChwb3B1cF9mMDllYWM2NDQ4MGI0NmEyODU2YjNhZTgxZWQwNzZmMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xYjc3Nzk5YTQ3NDY0M2ZlYmRlYzE3NmZjNGFlYjEyMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc3MDExOTksLTc5LjQwODQ5Mjc5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2M2MjY2MTdkODJkNDQwZTI5ZWQ4M2RmNzIzMTk1OTZmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQ4MjhkYzQ0MGMxMzQyMTFiOWZmZjYzMmMwNTMzNTcwID0gJCgnPGRpdiBpZD0iaHRtbF80ODI4ZGM0NDBjMTM0MjExYjlmZmY2MzJjMDUzMzU3MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2lsbG93ZGFsZSBTb3V0aCwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzYyNjYxN2Q4MmQ0NDBlMjllZDgzZGY3MjMxOTU5NmYuc2V0Q29udGVudChodG1sXzQ4MjhkYzQ0MGMxMzQyMTFiOWZmZjYzMmMwNTMzNTcwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFiNzc3OTlhNDc0NjQzZmViZGVjMTc2ZmM0YWViMTIwLmJpbmRQb3B1cChwb3B1cF9jNjI2NjE3ZDgyZDQ0MGUyOWVkODNkZjcyMzE5NTk2Zik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iMzE1MmY4YmUxYWM0MTM1YmYxOTNhZDYxMjZkY2Y2ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc1Mjc1ODI5OTk5OTk5NiwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lZDRhM2Y5YTkxNGM0NDRhOTU1Y2Y0MjY4MTRmZTI5OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jY2M1ODIzY2MyNWM0MGU3OTc1Mjk1YjdhMTFmYTMzNyA9ICQoJzxkaXYgaWQ9Imh0bWxfY2NjNTgyM2NjMjVjNDBlNzk3NTI5NWI3YTExZmEzMzciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPllvcmsgTWlsbHMgV2VzdCwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZWQ0YTNmOWE5MTRjNDQ0YTk1NWNmNDI2ODE0ZmUyOTkuc2V0Q29udGVudChodG1sX2NjYzU4MjNjYzI1YzQwZTc5NzUyOTViN2ExMWZhMzM3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2IzMTUyZjhiZTFhYzQxMzViZjE5M2FkNjEyNmRjZjZkLmJpbmRQb3B1cChwb3B1cF9lZDRhM2Y5YTkxNGM0NDRhOTU1Y2Y0MjY4MTRmZTI5OSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kMjQ4ZDNjYWJjMjI0MDA4YTNkODExZjNiYWE2ZWM3OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4MjczNjQsLTc5LjQ0MjI1OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWM0YzA5NThiNTI5NDFjMmFmZmM5NDFjYzYxM2UyN2MgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzRjYWI3NjQ3ZWViNDRkNjhlM2EzODlhMGRiMTUzMDEgPSAkKCc8ZGl2IGlkPSJodG1sXzM0Y2FiNzY0N2VlYjQ0ZDY4ZTNhMzg5YTBkYjE1MzAxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XaWxsb3dkYWxlIFdlc3QsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFjNGMwOTU4YjUyOTQxYzJhZmZjOTQxY2M2MTNlMjdjLnNldENvbnRlbnQoaHRtbF8zNGNhYjc2NDdlZWI0NGQ2OGUzYTM4OWEwZGIxNTMwMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kMjQ4ZDNjYWJjMjI0MDA4YTNkODExZjNiYWE2ZWM3OS5iaW5kUG9wdXAocG9wdXBfMWM0YzA5NThiNTI5NDFjMmFmZmM5NDFjYzYxM2UyN2MpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfM2VkMmM3M2ExM2IyNGFmNWFiYWIyZGRlNjVkYjM5ZDAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTMyNTg2LC03OS4zMjk2NTY1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2M5YmMxOTRiYTI2ZTRmYzBhZTk0ZjRiNjM1MjUxMjkyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZlOWM4NjFlYTlhODRmMzY4NmUwZGMyNWJjZWU2ODI4ID0gJCgnPGRpdiBpZD0iaHRtbF9mZTljODYxZWE5YTg0ZjM2ODZlMGRjMjViY2VlNjgyOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFya3dvb2RzLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jOWJjMTk0YmEyNmU0ZmMwYWU5NGY0YjYzNTI1MTI5Mi5zZXRDb250ZW50KGh0bWxfZmU5Yzg2MWVhOWE4NGYzNjg2ZTBkYzI1YmNlZTY4MjgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2VkMmM3M2ExM2IyNGFmNWFiYWIyZGRlNjVkYjM5ZDAuYmluZFBvcHVwKHBvcHVwX2M5YmMxOTRiYTI2ZTRmYzBhZTk0ZjRiNjM1MjUxMjkyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQzNTVkYWM1YWM3MTQwZjRhY2Q5ZjFjOGY4NWFjNmZmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzQ1OTA1Nzk5OTk5OTk2LC03OS4zNTIxODhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWQwYmYzZGM1MDQ5NGE3MmE5YjI1OTE0OWQ5OTZiZGMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNGMyMDA1NTE2ZDk1NDNmNjk5ZGMwNDYzYjFjZjZkN2UgPSAkKCc8ZGl2IGlkPSJodG1sXzRjMjAwNTUxNmQ5NTQzZjY5OWRjMDQ2M2IxY2Y2ZDdlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb24gTWlsbHMgTm9ydGgsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFkMGJmM2RjNTA0OTRhNzJhOWIyNTkxNDlkOTk2YmRjLnNldENvbnRlbnQoaHRtbF80YzIwMDU1MTZkOTU0M2Y2OTlkYzA0NjNiMWNmNmQ3ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80MzU1ZGFjNWFjNzE0MGY0YWNkOWYxYzhmODVhYzZmZi5iaW5kUG9wdXAocG9wdXBfMWQwYmYzZGM1MDQ5NGE3MmE5YjI1OTE0OWQ5OTZiZGMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYThiMDFhOWI2ZTRmNGRlMGI3MTAzYjEzNjJiMmQxODggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MjU4OTk3MDAwMDAwMSwtNzkuMzQwOTIzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2VhYzJkNWNkZDZmNjRhMzY5YTk4NWFiZjc4ZmZhNmQyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzFiODJkOWQ0NTU3YTRmNzE4MjMwZjIyOGE5ODAxMDgxID0gJCgnPGRpdiBpZD0iaHRtbF8xYjgyZDlkNDU1N2E0ZjcxODIzMGYyMjhhOTgwMTA4MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RmxlbWluZ2RvbiBQYXJrLCBEb24gTWlsbHMgU291dGgsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VhYzJkNWNkZDZmNjRhMzY5YTk4NWFiZjc4ZmZhNmQyLnNldENvbnRlbnQoaHRtbF8xYjgyZDlkNDU1N2E0ZjcxODIzMGYyMjhhOTgwMTA4MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hOGIwMWE5YjZlNGY0ZGUwYjcxMDNiMTM2MmIyZDE4OC5iaW5kUG9wdXAocG9wdXBfZWFjMmQ1Y2RkNmY2NGEzNjlhOTg1YWJmNzhmZmE2ZDIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGI5YTc4ZjFjMDMzNGRhOWEyNmUzYTQ3YTEwZDk5ODYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTQzMjgzLC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzVhMDA1NTc5NTA1ZDQ3Y2E5Y2M1MjkzNmMyMjdmMzBlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I2Njc0NzhlOTM2MjQzZmRiYmRjOGNhZWUzYzk1MDM5ID0gJCgnPGRpdiBpZD0iaHRtbF9iNjY3NDc4ZTkzNjI0M2ZkYmJkYzhjYWVlM2M5NTAzOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmF0aHVyc3QgTWFub3IsIERvd25zdmlldyBOb3J0aCwgV2lsc29uIEhlaWdodHMsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVhMDA1NTc5NTA1ZDQ3Y2E5Y2M1MjkzNmMyMjdmMzBlLnNldENvbnRlbnQoaHRtbF9iNjY3NDc4ZTkzNjI0M2ZkYmJkYzhjYWVlM2M5NTAzOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wYjlhNzhmMWMwMzM0ZGE5YTI2ZTNhNDdhMTBkOTk4Ni5iaW5kUG9wdXAocG9wdXBfNWEwMDU1Nzk1MDVkNDdjYTljYzUyOTM2YzIyN2YzMGUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDU2NTMwNTY1MTJiNGZlMTgyOTcxNjkxMjViM2M3NTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Njc5ODAzLC03OS40ODcyNjE5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hZWQ3ZjI5OWQ5OGY0Yjg3YmY5NGQxMmUyNmM3MjJjZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83ZTdlNzE2OGNkMGQ0ODJmYWE1ZTNlMzcyMDZkZTBhNyA9ICQoJzxkaXYgaWQ9Imh0bWxfN2U3ZTcxNjhjZDBkNDgyZmFhNWUzZTM3MjA2ZGUwYTciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5vcnRod29vZCBQYXJrLCBZb3JrIFVuaXZlcnNpdHksIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FlZDdmMjk5ZDk4ZjRiODdiZjk0ZDEyZTI2YzcyMmNlLnNldENvbnRlbnQoaHRtbF83ZTdlNzE2OGNkMGQ0ODJmYWE1ZTNlMzcyMDZkZTBhNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kNTY1MzA1NjUxMmI0ZmUxODI5NzE2OTEyNWIzYzc1MS5iaW5kUG9wdXAocG9wdXBfYWVkN2YyOTlkOThmNGI4N2JmOTRkMTJlMjZjNzIyY2UpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWVkMDhjZTU1ODQ5NDk5NWFiMjljMGIwYzMwZDY1ZWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Mzc0NzMyMDAwMDAwMDQsLTc5LjQ2NDc2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U5ZTVjODczN2FkMTQ1Y2JhYWY1N2FlNGMwMmVlNjEzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q0Y2U4ZWY5YmQ0ODRmNjU4N2FkZTI1MWIxNjkyMTllID0gJCgnPGRpdiBpZD0iaHRtbF9kNGNlOGVmOWJkNDg0ZjY1ODdhZGUyNTFiMTY5MjE5ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q0ZCIFRvcm9udG8sIERvd25zdmlldyBFYXN0LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lOWU1Yzg3MzdhZDE0NWNiYWFmNTdhZTRjMDJlZTYxMy5zZXRDb250ZW50KGh0bWxfZDRjZThlZjliZDQ4NGY2NTg3YWRlMjUxYjE2OTIxOWUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOWVkMDhjZTU1ODQ5NDk5NWFiMjljMGIwYzMwZDY1ZWIuYmluZFBvcHVwKHBvcHVwX2U5ZTVjODczN2FkMTQ1Y2JhYWY1N2FlNGMwMmVlNjEzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzMzZTg1ZGU0ZTMxYjQwMTZiZTllNzQyMjFlMzY4MWY0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzM5MDE0NiwtNzkuNTA2OTQzNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mY2UyZmQwZDBkODc0M2I1YjczODllNzYyYTU4OWM1YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80ODIwZjA2OGE1N2I0MWZkYTE4M2ZhMjE0MmYzZWYyZCA9ICQoJzxkaXYgaWQ9Imh0bWxfNDgyMGYwNjhhNTdiNDFmZGExODNmYTIxNDJmM2VmMmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd25zdmlldyBXZXN0LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mY2UyZmQwZDBkODc0M2I1YjczODllNzYyYTU4OWM1Yi5zZXRDb250ZW50KGh0bWxfNDgyMGYwNjhhNTdiNDFmZGExODNmYTIxNDJmM2VmMmQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzNlODVkZTRlMzFiNDAxNmJlOWU3NDIyMWUzNjgxZjQuYmluZFBvcHVwKHBvcHVwX2ZjZTJmZDBkMGQ4NzQzYjViNzM4OWU3NjJhNTg5YzViKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZmOGIyZjBiM2M4MjQ0YzY4NjQ3Y2RmMTdlNjI1NTg2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI4NDk2NCwtNzkuNDk1Njk3NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTI5OWFhMDJmYTRmNGJjZjgyYTQ1MzYxN2RhYTYwOTMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDVkMjI1ODZlMTAwNDA1MmJlMWY5NDk2OWZmNTFlMTUgPSAkKCc8ZGl2IGlkPSJodG1sXzQ1ZDIyNTg2ZTEwMDQwNTJiZTFmOTQ5NjlmZjUxZTE1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3duc3ZpZXcgQ2VudHJhbCwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTI5OWFhMDJmYTRmNGJjZjgyYTQ1MzYxN2RhYTYwOTMuc2V0Q29udGVudChodG1sXzQ1ZDIyNTg2ZTEwMDQwNTJiZTFmOTQ5NjlmZjUxZTE1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZmOGIyZjBiM2M4MjQ0YzY4NjQ3Y2RmMTdlNjI1NTg2LmJpbmRQb3B1cChwb3B1cF85Mjk5YWEwMmZhNGY0YmNmODJhNDUzNjE3ZGFhNjA5Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yYjFiOTQ3OGZiNjc0NGM3YWNkYjE3NTYzODA1ODU3NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc2MTYzMTMsLTc5LjUyMDk5OTQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzE3Yjc2ZGQzMzYxNDRhMmM5OTA3MTAwNjI5NTU3MmE2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FhYzFkZjY5NzE2NjRmMjU4YjU0YmQzOWUwOTg0NTlmID0gJCgnPGRpdiBpZD0iaHRtbF9hYWMxZGY2OTcxNjY0ZjI1OGI1NGJkMzllMDk4NDU5ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnN2aWV3IE5vcnRod2VzdCwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTdiNzZkZDMzNjE0NGEyYzk5MDcxMDA2Mjk1NTcyYTYuc2V0Q29udGVudChodG1sX2FhYzFkZjY5NzE2NjRmMjU4YjU0YmQzOWUwOTg0NTlmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJiMWI5NDc4ZmI2NzQ0YzdhY2RiMTc1NjM4MDU4NTc2LmJpbmRQb3B1cChwb3B1cF8xN2I3NmRkMzM2MTQ0YTJjOTkwNzEwMDYyOTU1NzJhNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hMTY2ZDhmMDhhMjc0MDdjYjdjODkxODhjMzBhY2M1NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyNTg4MjI5OTk5OTk5NSwtNzkuMzE1NTcxNTk5OTk5OThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTAzYmU2MmRiZGI3NDA4OTg4NzRhZGNkNTY4MjFjM2IgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGZmOTc4ODRjMDE4NGRiMWIwY2VjMjAyMTUxODFjODcgPSAkKCc8ZGl2IGlkPSJodG1sXzBmZjk3ODg0YzAxODRkYjFiMGNlYzIwMjE1MTgxYzg3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5WaWN0b3JpYSBWaWxsYWdlLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lMDNiZTYyZGJkYjc0MDg5ODg3NGFkY2Q1NjgyMWMzYi5zZXRDb250ZW50KGh0bWxfMGZmOTc4ODRjMDE4NGRiMWIwY2VjMjAyMTUxODFjODcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTE2NmQ4ZjA4YTI3NDA3Y2I3Yzg5MTg4YzMwYWNjNTYuYmluZFBvcHVwKHBvcHVwX2UwM2JlNjJkYmRiNzQwODk4ODc0YWRjZDU2ODIxYzNiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzdmYmNkZWRhZTE4NTQ4MWZiM2RlZjYzMmJjMDZkZGIyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA2Mzk3MiwtNzkuMzA5OTM3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzkyYmIxZGRlOTkyZjRiYjE5M2M2YTM3OTYyNDM2OTc1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzAzYjRkYjRkMGVhMTRlYjlhZTFjMDVjMzQ3MTBlMGFkID0gJCgnPGRpdiBpZD0iaHRtbF8wM2I0ZGI0ZDBlYTE0ZWI5YWUxYzA1YzM0NzEwZTBhZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V29vZGJpbmUgR2FyZGVucywgUGFya3ZpZXcgSGlsbCwgRWFzdCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MmJiMWRkZTk5MmY0YmIxOTNjNmEzNzk2MjQzNjk3NS5zZXRDb250ZW50KGh0bWxfMDNiNGRiNGQwZWExNGViOWFlMWMwNWMzNDcxMGUwYWQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfN2ZiY2RlZGFlMTg1NDgxZmIzZGVmNjMyYmMwNmRkYjIuYmluZFBvcHVwKHBvcHVwXzkyYmIxZGRlOTkyZjRiYjE5M2M2YTM3OTYyNDM2OTc1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I1NDhiNTc3YzRjNDQ5MWJiMjg2ZGFjMmIyMDFiNDg2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjk1MzQzOTAwMDAwMDA1LC03OS4zMTgzODg3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JmMDQ2YmE0MzYwMDRhMjliOTQ5MDdiYzVmODJiNjE0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FkOWZhMWI1MDQxMjQ3MWE4ZGZhODYyZWNjY2E5Yjk2ID0gJCgnPGRpdiBpZD0iaHRtbF9hZDlmYTFiNTA0MTI0NzFhOGRmYTg2MmVjY2NhOWI5NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V29vZGJpbmUgSGVpZ2h0cywgRWFzdCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iZjA0NmJhNDM2MDA0YTI5Yjk0OTA3YmM1ZjgyYjYxNC5zZXRDb250ZW50KGh0bWxfYWQ5ZmExYjUwNDEyNDcxYThkZmE4NjJlY2NjYTliOTYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjU0OGI1NzdjNGM0NDkxYmIyODZkYWMyYjIwMWI0ODYuYmluZFBvcHVwKHBvcHVwX2JmMDQ2YmE0MzYwMDRhMjliOTQ5MDdiYzVmODJiNjE0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2QwYWJmMzE2Y2Q0MTQ2NmJiZWRhZTNhYmRlMDI1YmEyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc2MzU3Mzk5OTk5OTksLTc5LjI5MzAzMTJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTExMDk2YjVhZGI1NDJkYjhhN2E4ZGE2NTVhNTMwOWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODE1Zjg5MmE0MzMxNGI4YTg4ZTgzZTRmNWYyMTFiMzMgPSAkKCc8ZGl2IGlkPSJodG1sXzgxNWY4OTJhNDMzMTRiOGE4OGU4M2U0ZjVmMjExYjMzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgQmVhY2hlcywgRWFzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MTEwOTZiNWFkYjU0MmRiOGE3YThkYTY1NWE1MzA5ZS5zZXRDb250ZW50KGh0bWxfODE1Zjg5MmE0MzMxNGI4YTg4ZTgzZTRmNWYyMTFiMzMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDBhYmYzMTZjZDQxNDY2YmJlZGFlM2FiZGUwMjViYTIuYmluZFBvcHVwKHBvcHVwXzkxMTA5NmI1YWRiNTQyZGI4YTdhOGRhNjU1YTUzMDllKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQwMGU1MmIyY2Y5ODRhNTY5OWIyNDMyODVjZjdkODBmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA5MDYwNCwtNzkuMzYzNDUxN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xM2QxNDk2ZjZhOGY0MzZkODMzMjZiNjM1N2U4ZjY2YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80OTVlNjJiMGI1NTk0NjI0YTE1OTU3NDVjNDBlMzg0NyA9ICQoJzxkaXYgaWQ9Imh0bWxfNDk1ZTYyYjBiNTU5NDYyNGExNTk1NzQ1YzQwZTM4NDciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxlYXNpZGUsIEVhc3QgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTNkMTQ5NmY2YThmNDM2ZDgzMzI2YjYzNTdlOGY2NmIuc2V0Q29udGVudChodG1sXzQ5NWU2MmIwYjU1OTQ2MjRhMTU5NTc0NWM0MGUzODQ3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQwMGU1MmIyY2Y5ODRhNTY5OWIyNDMyODVjZjdkODBmLmJpbmRQb3B1cChwb3B1cF8xM2QxNDk2ZjZhOGY0MzZkODMzMjZiNjM1N2U4ZjY2Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kMGM3ZGNkZWEwYTE0ZjAwYTM1ODIzNDc3MzgxMmRkZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNTM2ODksLTc5LjM0OTM3MTkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzJlZjcyNWQ3MGJkZjQ3YjI5NDZhMTViYmVhMzk3MmYyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RhY2VkZmFhMGI0MzQyM2I5YTA5NmFlNTdlMjFiMGY4ID0gJCgnPGRpdiBpZD0iaHRtbF9kYWNlZGZhYTBiNDM0MjNiOWEwOTZhZTU3ZTIxYjBmOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhvcm5jbGlmZmUgUGFyaywgRWFzdCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yZWY3MjVkNzBiZGY0N2IyOTQ2YTE1YmJlYTM5NzJmMi5zZXRDb250ZW50KGh0bWxfZGFjZWRmYWEwYjQzNDIzYjlhMDk2YWU1N2UyMWIwZjgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDBjN2RjZGVhMGExNGYwMGEzNTgyMzQ3NzM4MTJkZGUuYmluZFBvcHVwKHBvcHVwXzJlZjcyNWQ3MGJkZjQ3YjI5NDZhMTViYmVhMzk3MmYyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRhOWU0OWQwNTA3ZDQ0ZGY5ZWYwM2VkMDllM2RhNjBhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg1MzQ3LC03OS4zMzgxMDY1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYxOTFmNTBiMDE5MDQ4NTBhMzdiMzlhYTVhNWJlNTQyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzNlNTI4ODMzNTdhNTQ3Y2FhZjE5ZDQyZTZhZDY2YTQyID0gJCgnPGRpdiBpZD0iaHRtbF8zZTUyODgzMzU3YTU0N2NhYWYxOWQ0MmU2YWQ2NmE0MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RWFzdCBUb3JvbnRvLCBFYXN0IFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYxOTFmNTBiMDE5MDQ4NTBhMzdiMzlhYTVhNWJlNTQyLnNldENvbnRlbnQoaHRtbF8zZTUyODgzMzU3YTU0N2NhYWYxOWQ0MmU2YWQ2NmE0Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80YTllNDlkMDUwN2Q0NGRmOWVmMDNlZDA5ZTNkYTYwYS5iaW5kUG9wdXAocG9wdXBfNjE5MWY1MGIwMTkwNDg1MGEzN2IzOWFhNWE1YmU1NDIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTJkNmZjMTlhMDIzNGJkYmFlNWUwMzY4OWE0OGZlZDggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NTcxLC03OS4zNTIxODhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDA5YTNkZGEyMGJkNDRiOGE5YThmNDFjMzE1ZjA3MmQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfY2ViNjk0MmIwZmFhNGI5ZWI1ZDljN2M4NWRhYjZjNGEgPSAkKCc8ZGl2IGlkPSJodG1sX2NlYjY5NDJiMGZhYTRiOWViNWQ5YzdjODVkYWI2YzRhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgRGFuZm9ydGggV2VzdCwgUml2ZXJkYWxlLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2QwOWEzZGRhMjBiZDQ0YjhhOWE4ZjQxYzMxNWYwNzJkLnNldENvbnRlbnQoaHRtbF9jZWI2OTQyYjBmYWE0YjllYjVkOWM3Yzg1ZGFiNmM0YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lMmQ2ZmMxOWEwMjM0YmRiYWU1ZTAzNjg5YTQ4ZmVkOC5iaW5kUG9wdXAocG9wdXBfZDA5YTNkZGEyMGJkNDRiOGE5YThmNDFjMzE1ZjA3MmQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTMwZTQ4YTU5MGFhNDVmZTllMjIxNDNmYjk0YjBiZDcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njg5OTg1LC03OS4zMTU1NzE1OTk5OTk5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kYTcwN2EwNzA2Zjk0MjYwOTAxODFhZWEwN2FjYzM5MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wOTVkODY3NjhlN2I0NDg0ODIxZGVlYTUxZmRhM2NjYSA9ICQoJzxkaXYgaWQ9Imh0bWxfMDk1ZDg2NzY4ZTdiNDQ4NDgyMWRlZWE1MWZkYTNjY2EiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBCZWFjaGVzIFdlc3QsIEluZGlhIEJhemFhciwgRWFzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kYTcwN2EwNzA2Zjk0MjYwOTAxODFhZWEwN2FjYzM5MS5zZXRDb250ZW50KGh0bWxfMDk1ZDg2NzY4ZTdiNDQ4NDgyMWRlZWE1MWZkYTNjY2EpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTMwZTQ4YTU5MGFhNDVmZTllMjIxNDNmYjk0YjBiZDcuYmluZFBvcHVwKHBvcHVwX2RhNzA3YTA3MDZmOTQyNjA5MDE4MWFlYTA3YWNjMzkxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNkNTU3ZWZkYTM3MDRiZTA5MWJlOTAxMDQxNDlkZDhmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU5NTI1NSwtNzkuMzQwOTIzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA1ZmVjZjFmNzg5MTQ0MjliY2FlMjhkMzczODYzNDQyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2NhYTBlNGM2YTkwNDRmNmVhZmIwODQ2NWQxZmQyY2UwID0gJCgnPGRpdiBpZD0iaHRtbF9jYWEwZTRjNmE5MDQ0ZjZlYWZiMDg0NjVkMWZkMmNlMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3R1ZGlvIERpc3RyaWN0LCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzA1ZmVjZjFmNzg5MTQ0MjliY2FlMjhkMzczODYzNDQyLnNldENvbnRlbnQoaHRtbF9jYWEwZTRjNmE5MDQ0ZjZlYWZiMDg0NjVkMWZkMmNlMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zZDU1N2VmZGEzNzA0YmUwOTFiZTkwMTA0MTQ5ZGQ4Zi5iaW5kUG9wdXAocG9wdXBfMDVmZWNmMWY3ODkxNDQyOWJjYWUyOGQzNzM4NjM0NDIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTI0MWJiMjdlYmM4NDU4YmJjN2RlMjY1OTdkNjdhZjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MjgwMjA1LC03OS4zODg3OTAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2QyNzVhZWJiNTljMzQ4YjhhOWU5MGQzYjUzMzQ5YWEyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdlZjdjMzQyNTY3YTQyOTJhZDA1NjM3MjI2MjkxZTAwID0gJCgnPGRpdiBpZD0iaHRtbF83ZWY3YzM0MjU2N2E0MjkyYWQwNTYzNzIyNjI5MWUwMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGF3cmVuY2UgUGFyaywgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kMjc1YWViYjU5YzM0OGI4YTllOTBkM2I1MzM0OWFhMi5zZXRDb250ZW50KGh0bWxfN2VmN2MzNDI1NjdhNDI5MmFkMDU2MzcyMjYyOTFlMDApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTI0MWJiMjdlYmM4NDU4YmJjN2RlMjY1OTdkNjdhZjEuYmluZFBvcHVwKHBvcHVwX2QyNzVhZWJiNTljMzQ4YjhhOWU5MGQzYjUzMzQ5YWEyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NlMmQwZDRlNmY2YzQxNjdhM2U4ODVjY2QwMzczNTVkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzEyNzUxMSwtNzkuMzkwMTk3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82MDJhOGQ2MWRiYzI0YTQyODE5Nzg2NTM1MDIyM2JkOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82ZjljMzZmMGU1ZDI0M2NhOGY0MDgzYmExZDMzNTZmMCA9ICQoJzxkaXYgaWQ9Imh0bWxfNmY5YzM2ZjBlNWQyNDNjYThmNDA4M2JhMWQzMzU2ZjAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRhdmlzdmlsbGUgTm9ydGgsIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjAyYThkNjFkYmMyNGE0MjgxOTc4NjUzNTAyMjNiZDguc2V0Q29udGVudChodG1sXzZmOWMzNmYwZTVkMjQzY2E4ZjQwODNiYTFkMzM1NmYwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2NlMmQwZDRlNmY2YzQxNjdhM2U4ODVjY2QwMzczNTVkLmJpbmRQb3B1cChwb3B1cF82MDJhOGQ2MWRiYzI0YTQyODE5Nzg2NTM1MDIyM2JkOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xNWU1N2FjYTkwMGU0NDFkYmE1NjUxNjMwMjhhMWNkNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxNTM4MzQsLTc5LjQwNTY3ODQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA1MDA2ZTExOTNmZTRiNDlhMzIxNWU4MmVmZmVkYjcwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzcxNTQxYmJjYzdlZTQ0M2E4YjY4ZTU3ODllOTExOWRlID0gJCgnPGRpdiBpZD0iaHRtbF83MTU0MWJiY2M3ZWU0NDNhOGI2OGU1Nzg5ZTkxMTlkZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Tm9ydGggVG9yb250byBXZXN0LCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzA1MDA2ZTExOTNmZTRiNDlhMzIxNWU4MmVmZmVkYjcwLnNldENvbnRlbnQoaHRtbF83MTU0MWJiY2M3ZWU0NDNhOGI2OGU1Nzg5ZTkxMTlkZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xNWU1N2FjYTkwMGU0NDFkYmE1NjUxNjMwMjhhMWNkNS5iaW5kUG9wdXAocG9wdXBfMDUwMDZlMTE5M2ZlNGI0OWEzMjE1ZTgyZWZmZWRiNzApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDY4ZWM4MzdjNmE4NGZkODg3MTAwYTA3MDQ0YTk1ZWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDQzMjQ0LC03OS4zODg3OTAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2IyM2ZkYTdlOTkwOTRjZTk5MjQ2ODc2MGM5MGY4YjA5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzUxZDU5MzJmNjE3OTRmN2NhMDk3ZTkxZjRjNWY2ZThjID0gJCgnPGRpdiBpZD0iaHRtbF81MWQ1OTMyZjYxNzk0ZjdjYTA5N2U5MWY0YzVmNmU4YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGF2aXN2aWxsZSwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iMjNmZGE3ZTk5MDk0Y2U5OTI0Njg3NjBjOTBmOGIwOS5zZXRDb250ZW50KGh0bWxfNTFkNTkzMmY2MTc5NGY3Y2EwOTdlOTFmNGM1ZjZlOGMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDY4ZWM4MzdjNmE4NGZkODg3MTAwYTA3MDQ0YTk1ZWYuYmluZFBvcHVwKHBvcHVwX2IyM2ZkYTdlOTkwOTRjZTk5MjQ2ODc2MGM5MGY4YjA5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E5NGNkMDM4MTY0OTRhMDk4NWY5ZmJiNmI1NWIxNmNkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg5NTc0MywtNzkuMzgzMTU5OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfM2Q5MTI3OTRlYWFmNDk1YzkxOGQ4ODUyNjZlMzYyZmIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfM2M2NTYwYzFjZTdjNDliNzhiNTYwN2M0OGJkNjJjNjIgPSAkKCc8ZGl2IGlkPSJodG1sXzNjNjU2MGMxY2U3YzQ5Yjc4YjU2MDdjNDhiZDYyYzYyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Nb29yZSBQYXJrLCBTdW1tZXJoaWxsIEVhc3QsIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2Q5MTI3OTRlYWFmNDk1YzkxOGQ4ODUyNjZlMzYyZmIuc2V0Q29udGVudChodG1sXzNjNjU2MGMxY2U3YzQ5Yjc4YjU2MDdjNDhiZDYyYzYyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2E5NGNkMDM4MTY0OTRhMDk4NWY5ZmJiNmI1NWIxNmNkLmJpbmRQb3B1cChwb3B1cF8zZDkxMjc5NGVhYWY0OTVjOTE4ZDg4NTI2NmUzNjJmYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wMjk1Njg5ZGY1N2U0Y2RkYWI3ZmMxODViOWFlNGZhMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4NjQxMjI5OTk5OTk5LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU2NTNiZDE3OGQyMzQ3YjBiZDk1ODM5MGI1ZWRmNjNlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZhYWYwNjU5MDczMzQ2N2NhYzA2ZmFlM2VlMTI5NTc4ID0gJCgnPGRpdiBpZD0iaHRtbF9mYWFmMDY1OTA3MzM0NjdjYWMwNmZhZTNlZTEyOTU3OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGVlciBQYXJrLCBGb3Jlc3QgSGlsbCBTRSwgUmF0aG5lbGx5LCBTb3V0aCBIaWxsLCBTdW1tZXJoaWxsIFdlc3QsIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTY1M2JkMTc4ZDIzNDdiMGJkOTU4MzkwYjVlZGY2M2Uuc2V0Q29udGVudChodG1sX2ZhYWYwNjU5MDczMzQ2N2NhYzA2ZmFlM2VlMTI5NTc4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzAyOTU2ODlkZjU3ZTRjZGRhYjdmYzE4NWI5YWU0ZmEwLmJpbmRQb3B1cChwb3B1cF81NjUzYmQxNzhkMjM0N2IwYmQ5NTgzOTBiNWVkZjYzZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80ZWJkMDMzMDA4MDM0ZDNhODBkZWJlZWNlZDgxYzAxMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3OTU2MjYsLTc5LjM3NzUyOTQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU1NTI3YjVjMDY2OTQ4ZDY5NTliNGJlNGU0YmVhNWQ1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ViMmZmNzA3OWQ1NjQxZTY4ZjZiMWZhNGY1N2I3NzEyID0gJCgnPGRpdiBpZD0iaHRtbF9lYjJmZjcwNzlkNTY0MWU2OGY2YjFmYTRmNTdiNzcxMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um9zZWRhbGUsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzU1NTI3YjVjMDY2OTQ4ZDY5NTliNGJlNGU0YmVhNWQ1LnNldENvbnRlbnQoaHRtbF9lYjJmZjcwNzlkNTY0MWU2OGY2YjFmYTRmNTdiNzcxMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80ZWJkMDMzMDA4MDM0ZDNhODBkZWJlZWNlZDgxYzAxMS5iaW5kUG9wdXAocG9wdXBfNTU1MjdiNWMwNjY5NDhkNjk1OWI0YmU0ZTRiZWE1ZDUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfM2JlOWM3NjViNGI0NDBmN2FmYmNhMmRjMWU1MzRlOWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njc5NjcsLTc5LjM2NzY3NTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOWQ3NzRiODBkYWRjNDBkZGJmYjhkMDhiOWZhMjhjNDkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZGM2YTk4NDhmMjkwNDhmMDhmNDQyZTBhNDY3NzIzZTIgPSAkKCc8ZGl2IGlkPSJodG1sX2RjNmE5ODQ4ZjI5MDQ4ZjA4ZjQ0MmUwYTQ2NzcyM2UyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DYWJiYWdldG93biwgU3QuIEphbWVzIFRvd24sIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzlkNzc0YjgwZGFkYzQwZGRiZmI4ZDA4YjlmYTI4YzQ5LnNldENvbnRlbnQoaHRtbF9kYzZhOTg0OGYyOTA0OGYwOGY0NDJlMGE0Njc3MjNlMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zYmU5Yzc2NWI0YjQ0MGY3YWZiY2EyZGMxZTUzNGU5YS5iaW5kUG9wdXAocG9wdXBfOWQ3NzRiODBkYWRjNDBkZGJmYjhkMDhiOWZhMjhjNDkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzc5OTVjZTZhYTY1NGVhN2FhNjZkM2ZhMTBmODUwNWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjU4NTk5LC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mZmRkNWVmZWJjZjg0NzdlYjY1OTc4YWVhYTc4M2ViNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85YzY1ZGRiMjgyNjE0MTM2OTZhOTNjOTZhMWQzZDA5MiA9ICQoJzxkaXYgaWQ9Imh0bWxfOWM2NWRkYjI4MjYxNDEzNjk2YTkzYzk2YTFkM2QwOTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNodXJjaCBhbmQgV2VsbGVzbGV5LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mZmRkNWVmZWJjZjg0NzdlYjY1OTc4YWVhYTc4M2ViNy5zZXRDb250ZW50KGh0bWxfOWM2NWRkYjI4MjYxNDEzNjk2YTkzYzk2YTFkM2QwOTIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzc5OTVjZTZhYTY1NGVhN2FhNjZkM2ZhMTBmODUwNWMuYmluZFBvcHVwKHBvcHVwX2ZmZGQ1ZWZlYmNmODQ3N2ViNjU5NzhhZWFhNzgzZWI3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzI3ZTdkZDUxMzQxMDQ2YzVhYWJjNGY0NGZlMWI5MGUyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU0MjU5OSwtNzkuMzYwNjM1OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85ODFjMDg4MGY4NWE0ZGE2OWFiYTUyZTJhYzEzMDcyNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iNDc5ZjJkODI3M2M0MDNkYWZjOGNiNTZiMGJhODIxYiA9ICQoJzxkaXYgaWQ9Imh0bWxfYjQ3OWYyZDgyNzNjNDAzZGFmYzhjYjU2YjBiYTgyMWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhhcmJvdXJmcm9udCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTgxYzA4ODBmODVhNGRhNjlhYmE1MmUyYWMxMzA3MjYuc2V0Q29udGVudChodG1sX2I0NzlmMmQ4MjczYzQwM2RhZmM4Y2I1NmIwYmE4MjFiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzI3ZTdkZDUxMzQxMDQ2YzVhYWJjNGY0NGZlMWI5MGUyLmJpbmRQb3B1cChwb3B1cF85ODFjMDg4MGY4NWE0ZGE2OWFiYTUyZTJhYzEzMDcyNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84YjFkMzE0ZTU2MjQ0NjRlODZhZTllZmJlN2NjNTIwMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1NzE2MTgsLTc5LjM3ODkzNzA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzJlY2MyNzlmNDYzOTQ3ZGE4YjhiZTBlNTQxMzAzNzI5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZmYTc0MmExYzdkOTRiNWE4MTg0YjU4NDU2MDY0MjU3ID0gJCgnPGRpdiBpZD0iaHRtbF82ZmE3NDJhMWM3ZDk0YjVhODE4NGI1ODQ1NjA2NDI1NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnllcnNvbiwgR2FyZGVuIERpc3RyaWN0LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yZWNjMjc5ZjQ2Mzk0N2RhOGI4YmUwZTU0MTMwMzcyOS5zZXRDb250ZW50KGh0bWxfNmZhNzQyYTFjN2Q5NGI1YTgxODRiNTg0NTYwNjQyNTcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOGIxZDMxNGU1NjI0NDY0ZTg2YWU5ZWZiZTdjYzUyMDAuYmluZFBvcHVwKHBvcHVwXzJlY2MyNzlmNDYzOTQ3ZGE4YjhiZTBlNTQxMzAzNzI5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzYyYzhmMTk1OGM3MjQwMjU5ZjMwMDZjZmUwODVhYTcxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNDkzOSwtNzkuMzc1NDE3OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hZTRjZDM2ZWIzNTA0ODA0YWIyYmZiMGNhMTZlZmJiYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hYzIzMzRhYmYzZGM0YjlhYWYxYTQxNWU3ZTZkZjVmYiA9ICQoJzxkaXYgaWQ9Imh0bWxfYWMyMzM0YWJmM2RjNGI5YWFmMWE0MTVlN2U2ZGY1ZmIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hZTRjZDM2ZWIzNTA0ODA0YWIyYmZiMGNhMTZlZmJiYS5zZXRDb250ZW50KGh0bWxfYWMyMzM0YWJmM2RjNGI5YWFmMWE0MTVlN2U2ZGY1ZmIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjJjOGYxOTU4YzcyNDAyNTlmMzAwNmNmZTA4NWFhNzEuYmluZFBvcHVwKHBvcHVwX2FlNGNkMzZlYjM1MDQ4MDRhYjJiZmIwY2ExNmVmYmJhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ0YmU2MzFiZTQwODRhZjNiN2RjNmQ2MmIxZDMzMjcwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ0NzcwNzk5OTk5OTk2LC03OS4zNzMzMDY0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzE0OTYzYTI2MDAwYTRiY2Q5ZjgwYWQ5NTM1MjMyZWJhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzI4ZTUyOGMwNzAxZTRkMDNhYTk3MjZhMGE0ZWU4MmVlID0gJCgnPGRpdiBpZD0iaHRtbF8yOGU1MjhjMDcwMWU0ZDAzYWE5NzI2YTBhNGVlODJlZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmVyY3p5IFBhcmssIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE0OTYzYTI2MDAwYTRiY2Q5ZjgwYWQ5NTM1MjMyZWJhLnNldENvbnRlbnQoaHRtbF8yOGU1MjhjMDcwMWU0ZDAzYWE5NzI2YTBhNGVlODJlZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80NGJlNjMxYmU0MDg0YWYzYjdkYzZkNjJiMWQzMzI3MC5iaW5kUG9wdXAocG9wdXBfMTQ5NjNhMjYwMDBhNGJjZDlmODBhZDk1MzUyMzJlYmEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2M2NTJhM2Y3YTM3NDdiMGFkY2U5N2Y2ZTJjY2FkOWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTc5NTI0LC03OS4zODczODI2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RjOTFiMThlYWExNTQ0ODFhM2E5OTA4ODRmNjI1ZDA4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZiNTM4YzQ1MDdiODQ5ZGNiNTliMDZmZTA0YTQ4OWVmID0gJCgnPGRpdiBpZD0iaHRtbF9mYjUzOGM0NTA3Yjg0OWRjYjU5YjA2ZmUwNGE0ODllZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBCYXkgU3RyZWV0LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kYzkxYjE4ZWFhMTU0NDgxYTNhOTkwODg0ZjYyNWQwOC5zZXRDb250ZW50KGh0bWxfZmI1MzhjNDUwN2I4NDlkY2I1OWIwNmZlMDRhNDg5ZWYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2M2NTJhM2Y3YTM3NDdiMGFkY2U5N2Y2ZTJjY2FkOWUuYmluZFBvcHVwKHBvcHVwX2RjOTFiMThlYWExNTQ0ODFhM2E5OTA4ODRmNjI1ZDA4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I3NGE2ZjNlYTJhNzQzZmY4MmQ3NGZkNmU3MGQ3ZmRkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUwNTcxMjAwMDAwMDEsLTc5LjM4NDU2NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjI3NTg0MzJlMDZhNGU5YWJhYTg1YWJmZmNiZTU5MTQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzM5MGM2OTExZTMwNDgwZmI2MWExY2U2OGExMzc3Y2YgPSAkKCc8ZGl2IGlkPSJodG1sX2MzOTBjNjkxMWUzMDQ4MGZiNjFhMWNlNjhhMTM3N2NmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BZGVsYWlkZSwgS2luZywgUmljaG1vbmQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzIyNzU4NDMyZTA2YTRlOWFiYWE4NWFiZmZjYmU1OTE0LnNldENvbnRlbnQoaHRtbF9jMzkwYzY5MTFlMzA0ODBmYjYxYTFjZTY4YTEzNzdjZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iNzRhNmYzZWEyYTc0M2ZmODJkNzRmZDZlNzBkN2ZkZC5iaW5kUG9wdXAocG9wdXBfMjI3NTg0MzJlMDZhNGU5YWJhYTg1YWJmZmNiZTU5MTQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmZhMTQ3YjJhM2JlNGU2NGExNTI5MzA2MzFmMzNlOWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDA4MTU3LC03OS4zODE3NTIyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85NWIwYjI0MjZkYzI0ODA4OTgxZDI0NjE0MDE1OWE5YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80NDUyN2ZiNjZmYjE0YTVkYmUzZDk3MDc1YTM0OTZiOSA9ICQoJzxkaXYgaWQ9Imh0bWxfNDQ1MjdmYjY2ZmIxNGE1ZGJlM2Q5NzA3NWEzNDk2YjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhhcmJvdXJmcm9udCBFYXN0LCBUb3JvbnRvIElzbGFuZHMsIFVuaW9uIFN0YXRpb24sIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk1YjBiMjQyNmRjMjQ4MDg5ODFkMjQ2MTQwMTU5YTlhLnNldENvbnRlbnQoaHRtbF80NDUyN2ZiNjZmYjE0YTVkYmUzZDk3MDc1YTM0OTZiOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82ZmExNDdiMmEzYmU0ZTY0YTE1MjkzMDYzMWYzM2U5Yy5iaW5kUG9wdXAocG9wdXBfOTViMGIyNDI2ZGMyNDgwODk4MWQyNDYxNDAxNTlhOWEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmY2MDIzMTg0NDQyNDU1OGE5ODJjOWIzNmU2MTI1MjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDcxNzY4LC03OS4zODE1NzY0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82ZWZmZGNiZTFkZjk0ZDc2YmMwMTRjYWZmNzk4MzBjOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80ZTdlOGIzMTFkYzk0NDE4OTFkMzE5ZTFhZjkxMTY4NiA9ICQoJzxkaXYgaWQ9Imh0bWxfNGU3ZThiMzExZGM5NDQxODkxZDMxOWUxYWY5MTE2ODYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRlc2lnbiBFeGNoYW5nZSwgVG9yb250byBEb21pbmlvbiBDZW50cmUsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZlZmZkY2JlMWRmOTRkNzZiYzAxNGNhZmY3OTgzMGM5LnNldENvbnRlbnQoaHRtbF80ZTdlOGIzMTFkYzk0NDE4OTFkMzE5ZTFhZjkxMTY4Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mZjYwMjMxODQ0NDI0NTU4YTk4MmM5YjM2ZTYxMjUyMS5iaW5kUG9wdXAocG9wdXBfNmVmZmRjYmUxZGY5NGQ3NmJjMDE0Y2FmZjc5ODMwYzkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTk2MzkzNjlkNzcyNGMxMmExNDViNGM0Y2VmNjU2OWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDgxOTg1LC03OS4zNzk4MTY5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84YzZlNDc1OGI2Yjg0MTY5YjlkOGFmYWUwNWIxZDg1NCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wMjQ1ZjhjOTQxZGY0ZGIxODMxM2U5NmVjZTc0YjRjNSA9ICQoJzxkaXYgaWQ9Imh0bWxfMDI0NWY4Yzk0MWRmNGRiMTgzMTNlOTZlY2U3NGI0YzUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNvbW1lcmNlIENvdXJ0LCBWaWN0b3JpYSBIb3RlbCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOGM2ZTQ3NThiNmI4NDE2OWI5ZDhhZmFlMDViMWQ4NTQuc2V0Q29udGVudChodG1sXzAyNDVmOGM5NDFkZjRkYjE4MzEzZTk2ZWNlNzRiNGM1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk5NjM5MzY5ZDc3MjRjMTJhMTQ1YjRjNGNlZjY1NjlkLmJpbmRQb3B1cChwb3B1cF84YzZlNDc1OGI2Yjg0MTY5YjlkOGFmYWUwNWIxZDg1NCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lM2MwMzhmNjNiNDU0MDcwYWM3MTY5MGU1Yzc2ZmI4NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczMzI4MjUsLTc5LjQxOTc0OTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWE2YjRlMjhkNTg0NGEwZDlmOGI4ZmNkMzYyMDM2NjUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODBhOThlYjRiZjk1NDA1OTlhMDhkMmE1NzViZDY2MzIgPSAkKCc8ZGl2IGlkPSJodG1sXzgwYTk4ZWI0YmY5NTQwNTk5YTA4ZDJhNTc1YmQ2NjMyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CZWRmb3JkIFBhcmssIExhd3JlbmNlIE1hbm9yIEVhc3QsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VhNmI0ZTI4ZDU4NDRhMGQ5ZjhiOGZjZDM2MjAzNjY1LnNldENvbnRlbnQoaHRtbF84MGE5OGViNGJmOTU0MDU5OWEwOGQyYTU3NWJkNjYzMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lM2MwMzhmNjNiNDU0MDcwYWM3MTY5MGU1Yzc2ZmI4Ni5iaW5kUG9wdXAocG9wdXBfZWE2YjRlMjhkNTg0NGEwZDlmOGI4ZmNkMzYyMDM2NjUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmI2Yjc3NGJlZDQ5NDQ3NjkyMzZmMDA1OTY2MTA2MWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTE2OTQ4LC03OS40MTY5MzU1OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zMTlkNGQyMTFlYzc0ZDJmOGFkYTg2NDI2MzJiYjBlYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wYTRjMDU2YWQ5NmM0ZmI2YjUwMTFlYzgwYjNkYjdlNSA9ICQoJzxkaXYgaWQ9Imh0bWxfMGE0YzA1NmFkOTZjNGZiNmI1MDExZWM4MGIzZGI3ZTUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvc2VsYXduLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMxOWQ0ZDIxMWVjNzRkMmY4YWRhODY0MjYzMmJiMGVjLnNldENvbnRlbnQoaHRtbF8wYTRjMDU2YWQ5NmM0ZmI2YjUwMTFlYzgwYjNkYjdlNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82YjZiNzc0YmVkNDk0NDc2OTIzNmYwMDU5NjYxMDYxZi5iaW5kUG9wdXAocG9wdXBfMzE5ZDRkMjExZWM3NGQyZjhhZGE4NjQyNjMyYmIwZWMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNGY0ZGJmMzUzZGZlNDkxZGJlMzllODBhYTNhMTllMmIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTY5NDc2LC03OS40MTEzMDcyMDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iMzg5ODA1ZGVhMmU0MDBkOTg3Y2NmNjVmNzdiZWU1MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wYzU3NWY4N2NjMDU0YzVlOTEyZDM4NzA0ZDJkODZkMCA9ICQoJzxkaXYgaWQ9Imh0bWxfMGM1NzVmODdjYzA1NGM1ZTkxMmQzODcwNGQyZDg2ZDAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZvcmVzdCBIaWxsIE5vcnRoLCBGb3Jlc3QgSGlsbCBXZXN0LCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2IzODk4MDVkZWEyZTQwMGQ5ODdjY2Y2NWY3N2JlZTUzLnNldENvbnRlbnQoaHRtbF8wYzU3NWY4N2NjMDU0YzVlOTEyZDM4NzA0ZDJkODZkMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80ZjRkYmYzNTNkZmU0OTFkYmUzOWU4MGFhM2ExOWUyYi5iaW5kUG9wdXAocG9wdXBfYjM4OTgwNWRlYTJlNDAwZDk4N2NjZjY1Zjc3YmVlNTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWI4YTdjZDUxNmY1NGQwMGI3NDBkNDVmYzg5NTlhMDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzI3MDk3LC03OS40MDU2Nzg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83ODY2NDVlMGYzMjE0Mjk3YjM3NDkzNDcwMDBiNmNjZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84ZWE0NDNlMTY5OTQ0Y2RmODBhMTFmOGE1ZmEwY2M0ZSA9ICQoJzxkaXYgaWQ9Imh0bWxfOGVhNDQzZTE2OTk0NGNkZjgwYTExZjhhNWZhMGNjNGUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBBbm5leCwgTm9ydGggTWlkdG93biwgWW9ya3ZpbGxlLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzc4NjY0NWUwZjMyMTQyOTdiMzc0OTM0NzAwMGI2Y2NmLnNldENvbnRlbnQoaHRtbF84ZWE0NDNlMTY5OTQ0Y2RmODBhMTFmOGE1ZmEwY2M0ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lYjhhN2NkNTE2ZjU0ZDAwYjc0MGQ0NWZjODk1OWEwMS5iaW5kUG9wdXAocG9wdXBfNzg2NjQ1ZTBmMzIxNDI5N2IzNzQ5MzQ3MDAwYjZjY2YpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzNjOGYzZmNlYTBmNDU0NGIxNTI2N2YzMDliOTc0ZWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI2OTU2LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I1MTY4NGIxNGIxYTQ2NmFhMjExYWMxNGFiODNjMTE4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzI5NTVlNWZhYjJiNTRlN2I5OTFlY2ZmNmJjMTBjNjllID0gJCgnPGRpdiBpZD0iaHRtbF8yOTU1ZTVmYWIyYjU0ZTdiOTkxZWNmZjZiYzEwYzY5ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFyYm9yZCwgVW5pdmVyc2l0eSBvZiBUb3JvbnRvLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iNTE2ODRiMTRiMWE0NjZhYTIxMWFjMTRhYjgzYzExOC5zZXRDb250ZW50KGh0bWxfMjk1NWU1ZmFiMmI1NGU3Yjk5MWVjZmY2YmMxMGM2OWUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzNjOGYzZmNlYTBmNDU0NGIxNTI2N2YzMDliOTc0ZWEuYmluZFBvcHVwKHBvcHVwX2I1MTY4NGIxNGIxYTQ2NmFhMjExYWMxNGFiODNjMTE4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBmNTk0YjRkNzM1NzRlY2ZiMGI5ZDYyY2Y5ODQwZDJlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUzMjA1NywtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hYzU4ZmE5OWFhYTM0NDczYjUyMDE5ZmNhZTlhOGY5NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kMWYyNDg1ODlhYmQ0MDQzYmU3M2JhYzUyODBlMTMzZCA9ICQoJzxkaXYgaWQ9Imh0bWxfZDFmMjQ4NTg5YWJkNDA0M2JlNzNiYWM1MjgwZTEzM2QiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNoaW5hdG93biwgR3JhbmdlIFBhcmssIEtlbnNpbmd0b24gTWFya2V0LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hYzU4ZmE5OWFhYTM0NDczYjUyMDE5ZmNhZTlhOGY5Ny5zZXRDb250ZW50KGh0bWxfZDFmMjQ4NTg5YWJkNDA0M2JlNzNiYWM1MjgwZTEzM2QpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGY1OTRiNGQ3MzU3NGVjZmIwYjlkNjJjZjk4NDBkMmUuYmluZFBvcHVwKHBvcHVwX2FjNThmYTk5YWFhMzQ0NzNiNTIwMTlmY2FlOWE4Zjk3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZkMmZjY2M2MTJjNTRjYWJhMTkwYTBmZmJiZmM5OThlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjI4OTQ2NywtNzkuMzk0NDE5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81YTczMzRlZGYxMmY0Nzc5OGU4Nzk1M2FkYjFjOGY4NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yZDllZjNkZDA5Njk0NTNkYjgzMjFiM2Y3NjdmNTFkZCA9ICQoJzxkaXYgaWQ9Imh0bWxfMmQ5ZWYzZGQwOTY5NDUzZGI4MzIxYjNmNzY3ZjUxZGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNOIFRvd2VyLCBCYXRodXJzdCBRdWF5LCBJc2xhbmQgYWlycG9ydCwgSGFyYm91cmZyb250IFdlc3QsIEtpbmcgYW5kIFNwYWRpbmEsIFJhaWx3YXkgTGFuZHMsIFNvdXRoIE5pYWdhcmEsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVhNzMzNGVkZjEyZjQ3Nzk4ZTg3OTUzYWRiMWM4Zjg3LnNldENvbnRlbnQoaHRtbF8yZDllZjNkZDA5Njk0NTNkYjgzMjFiM2Y3NjdmNTFkZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82ZDJmY2NjNjEyYzU0Y2FiYTE5MGEwZmZiYmZjOTk4ZS5iaW5kUG9wdXAocG9wdXBfNWE3MzM0ZWRmMTJmNDc3OThlODc5NTNhZGIxYzhmODcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWViNGVhODMzNDQxNGUxYzliNjc3N2QyYjU1MTU0NzQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDY0MzUyLC03OS4zNzQ4NDU5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xMDYxNDdjNTZiYTg0NDM2OTQxODg4Y2M4ZTMxNmI5ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lZTgwZGQ2NTY2Mzc0NjM0OGJhNzY4YzhiN2NlYWZlMyA9ICQoJzxkaXYgaWQ9Imh0bWxfZWU4MGRkNjU2NjM3NDYzNDhiYTc2OGM4YjdjZWFmZTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0biBBIFBPIEJveGVzIDI1IFRoZSBFc3BsYW5hZGUsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzEwNjE0N2M1NmJhODQ0MzY5NDE4ODhjYzhlMzE2YjllLnNldENvbnRlbnQoaHRtbF9lZTgwZGQ2NTY2Mzc0NjM0OGJhNzY4YzhiN2NlYWZlMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hZWI0ZWE4MzM0NDE0ZTFjOWI2Nzc3ZDJiNTUxNTQ3NC5iaW5kUG9wdXAocG9wdXBfMTA2MTQ3YzU2YmE4NDQzNjk0MTg4OGNjOGUzMTZiOWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmE2Y2NkZmJjYWFjNGZmMTk4NzFmY2YxYmY3ZGJiMWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDg0MjkyLC03OS4zODIyODAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I2M2VlN2FkMjM5NDRhOGJiYjA0OGQyYjY3MzkzMmQ0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I0OTJhM2E4MTYzMTRkZDJiNzZlYWVlOWMxMDVlZTUxID0gJCgnPGRpdiBpZD0iaHRtbF9iNDkyYTNhODE2MzE0ZGQyYjc2ZWFlZTljMTA1ZWU1MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Rmlyc3QgQ2FuYWRpYW4gUGxhY2UsIFVuZGVyZ3JvdW5kIGNpdHksIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I2M2VlN2FkMjM5NDRhOGJiYjA0OGQyYjY3MzkzMmQ0LnNldENvbnRlbnQoaHRtbF9iNDkyYTNhODE2MzE0ZGQyYjc2ZWFlZTljMTA1ZWU1MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82YTZjY2RmYmNhYWM0ZmYxOTg3MWZjZjFiZjdkYmIxZi5iaW5kUG9wdXAocG9wdXBfYjYzZWU3YWQyMzk0NGE4YmJiMDQ4ZDJiNjczOTMyZDQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOThlOGY0NzBiZWUxNDA3ZTkzMzg3MzlhZDBlNjkwNzcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTg1MTc5OTk5OTk5OTYsLTc5LjQ2NDc2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzg5NjU0NzViNTc4MzQ1MjlhOTJkZmMxYzYyODRiNjI4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2U5ZjY3Y2ZiYjM0OTRjYWNhN2UwOGEzMjg3MGVlZDMxID0gJCgnPGRpdiBpZD0iaHRtbF9lOWY2N2NmYmIzNDk0Y2FjYTdlMDhhMzI4NzBlZWQzMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGF3cmVuY2UgSGVpZ2h0cywgTGF3cmVuY2UgTWFub3IsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg5NjU0NzViNTc4MzQ1MjlhOTJkZmMxYzYyODRiNjI4LnNldENvbnRlbnQoaHRtbF9lOWY2N2NmYmIzNDk0Y2FjYTdlMDhhMzI4NzBlZWQzMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85OGU4ZjQ3MGJlZTE0MDdlOTMzODczOWFkMGU2OTA3Ny5iaW5kUG9wdXAocG9wdXBfODk2NTQ3NWI1NzgzNDUyOWE5MmRmYzFjNjI4NGI2MjgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmVlYmM0ODI3N2I1NDA5YjkyMjUzNGNhMGVkYzNiZDIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDk1NzcsLTc5LjQ0NTA3MjU5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQxNjZiODFiODI2NjRiODFhMjBlOTI1ZTc1YTY5YzFlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzUxMDc2OTA2ZWU2OTQ3ZGY4NmFjZjgyNDIzNjQxYjQ1ID0gJCgnPGRpdiBpZD0iaHRtbF81MTA3NjkwNmVlNjk0N2RmODZhY2Y4MjQyMzY0MWI0NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R2xlbmNhaXJuLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80MTY2YjgxYjgyNjY0YjgxYTIwZTkyNWU3NWE2OWMxZS5zZXRDb250ZW50KGh0bWxfNTEwNzY5MDZlZTY5NDdkZjg2YWNmODI0MjM2NDFiNDUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmVlYmM0ODI3N2I1NDA5YjkyMjUzNGNhMGVkYzNiZDIuYmluZFBvcHVwKHBvcHVwXzQxNjZiODFiODI2NjRiODFhMjBlOTI1ZTc1YTY5YzFlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzI2NWRiM2Y0YzFkNjRlMDE5YzcwY2VmNjljOGM1MDc4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjkzNzgxMywtNzkuNDI4MTkxNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTBjMmVjZDM0YmJhNGUzODg4OTgxM2M5Nzg1NWYxMDAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfY2NiNDhiOGFhNDE3NDE0NDkzZWZkYTZlMTJkMDJlZjggPSAkKCc8ZGl2IGlkPSJodG1sX2NjYjQ4YjhhYTQxNzQxNDQ5M2VmZGE2ZTEyZDAyZWY4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IdW1ld29vZC1DZWRhcnZhbGUsIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2EwYzJlY2QzNGJiYTRlMzg4ODk4MTNjOTc4NTVmMTAwLnNldENvbnRlbnQoaHRtbF9jY2I0OGI4YWE0MTc0MTQ0OTNlZmRhNmUxMmQwMmVmOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yNjVkYjNmNGMxZDY0ZTAxOWM3MGNlZjY5YzhjNTA3OC5iaW5kUG9wdXAocG9wdXBfYTBjMmVjZDM0YmJhNGUzODg4OTgxM2M5Nzg1NWYxMDApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTJhYTRhZTVlY2JjNDI0ZTgwN2JjYTE3NjJmOTUwMjAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODkwMjU2LC03OS40NTM1MTJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDliMzY3NmI4ZGUzNDcyMjlhOTQ0MGZhNGNmOGMxYmYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzUzODVlYWViYmE3NGNlNGI3MDg5MmMyODcwNGU1ZjIgPSAkKCc8ZGl2IGlkPSJodG1sXzM1Mzg1ZWFlYmJhNzRjZTRiNzA4OTJjMjg3MDRlNWYyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DYWxlZG9uaWEtRmFpcmJhbmtzLCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wOWIzNjc2YjhkZTM0NzIyOWE5NDQwZmE0Y2Y4YzFiZi5zZXRDb250ZW50KGh0bWxfMzUzODVlYWViYmE3NGNlNGI3MDg5MmMyODcwNGU1ZjIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTJhYTRhZTVlY2JjNDI0ZTgwN2JjYTE3NjJmOTUwMjAuYmluZFBvcHVwKHBvcHVwXzA5YjM2NzZiOGRlMzQ3MjI5YTk0NDBmYTRjZjhjMWJmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzUwOGExMjc1NWVmOTRkYjRiY2ZlNjIyZTI1ODY0Y2VjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5NTQyLC03OS40MjI1NjM3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzY1YzBkY2ZmYmJiZjQ0NzJiODUzZDI5OWNiMWQyNjZiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzBhNGQzYThiZWI5ODRjYzNiMDY3NzU2MGI3NjIzYzY5ID0gJCgnPGRpdiBpZD0iaHRtbF8wYTRkM2E4YmViOTg0Y2MzYjA2Nzc1NjBiNzYyM2M2OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2hyaXN0aWUsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY1YzBkY2ZmYmJiZjQ0NzJiODUzZDI5OWNiMWQyNjZiLnNldENvbnRlbnQoaHRtbF8wYTRkM2E4YmViOTg0Y2MzYjA2Nzc1NjBiNzYyM2M2OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81MDhhMTI3NTVlZjk0ZGI0YmNmZTYyMmUyNTg2NGNlYy5iaW5kUG9wdXAocG9wdXBfNjVjMGRjZmZiYmJmNDQ3MmI4NTNkMjk5Y2IxZDI2NmIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjgyNzI1NjRlZThmNGU4N2E2ODVlZjM5YzhlNWFmMGIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjkwMDUxMDAwMDAwMSwtNzkuNDQyMjU5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85ZmZhNmY4MmE2YjU0Njg3YmEwZTA5ZWVjOGZkYTg5YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84YTBkNjczYjUzYzM0ZGQzYjdjZmI4NmViNmU2ZTFhOCA9ICQoJzxkaXYgaWQ9Imh0bWxfOGEwZDY3M2I1M2MzNGRkM2I3Y2ZiODZlYjZlNmUxYTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvdmVyY291cnQgVmlsbGFnZSwgRHVmZmVyaW4sIFdlc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWZmYTZmODJhNmI1NDY4N2JhMGUwOWVlYzhmZGE4OWIuc2V0Q29udGVudChodG1sXzhhMGQ2NzNiNTNjMzRkZDNiN2NmYjg2ZWI2ZTZlMWE4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzY4MjcyNTY0ZWU4ZjRlODdhNjg1ZWYzOWM4ZTVhZjBiLmJpbmRQb3B1cChwb3B1cF85ZmZhNmY4MmE2YjU0Njg3YmEwZTA5ZWVjOGZkYTg5Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80MGMxNzNkODA1ZjI0MGJjODNiMTBkZWJlNGY2OGJmNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzkyNjcwMDAwMDAwNiwtNzkuNDE5NzQ5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yY2Y1ZjIzYjg2ZjI0OWVkOTcxMjA4NzFiZWVhZGY3NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yNzZiYmZmZDNiMzc0ZTQyOGNmYWY1ZWZhNGYzNzRiYiA9ICQoJzxkaXYgaWQ9Imh0bWxfMjc2YmJmZmQzYjM3NGU0MjhjZmFmNWVmYTRmMzc0YmIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxpdHRsZSBQb3J0dWdhbCwgVHJpbml0eSwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yY2Y1ZjIzYjg2ZjI0OWVkOTcxMjA4NzFiZWVhZGY3NS5zZXRDb250ZW50KGh0bWxfMjc2YmJmZmQzYjM3NGU0MjhjZmFmNWVmYTRmMzc0YmIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDBjMTczZDgwNWYyNDBiYzgzYjEwZGViZTRmNjhiZjYuYmluZFBvcHVwKHBvcHVwXzJjZjVmMjNiODZmMjQ5ZWQ5NzEyMDg3MWJlZWFkZjc1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NhNGEzNzcxYzkyZDQ2MGNhNzhhMzBiNGNlNDk0OTQ1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjM2ODQ3MiwtNzkuNDI4MTkxNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGQwODlhNmZiYmI0NDI0ZjhkYzE1NDMxMWM3MDNjOTAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDVmNWI1NTkwOWY1NGIxODk3YTllYTFiN2VmNGM4NWYgPSAkKCc8ZGl2IGlkPSJodG1sX2Q1ZjViNTU5MDlmNTRiMTg5N2E5ZWExYjdlZjRjODVmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ccm9ja3RvbiwgRXhoaWJpdGlvbiBQbGFjZSwgUGFya2RhbGUgVmlsbGFnZSwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80ZDA4OWE2ZmJiYjQ0MjRmOGRjMTU0MzExYzcwM2M5MC5zZXRDb250ZW50KGh0bWxfZDVmNWI1NTkwOWY1NGIxODk3YTllYTFiN2VmNGM4NWYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2E0YTM3NzFjOTJkNDYwY2E3OGEzMGI0Y2U0OTQ5NDUuYmluZFBvcHVwKHBvcHVwXzRkMDg5YTZmYmJiNDQyNGY4ZGMxNTQzMTFjNzAzYzkwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzVlN2ZjMDkyMGY3NDRjYTE4ZGRjYWQ0MzIwYzhiN2QwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzEzNzU2MjAwMDAwMDA2LC03OS40OTAwNzM4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2YzNTI2MjVhMjY2ODQxY2E4OWY5MzZlZWYxMDczNDY4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzFmZmFjMjBmYmRiOTRhZTZiYmQxYTk5ODQwMzY5NmJmID0gJCgnPGRpdiBpZD0iaHRtbF8xZmZhYzIwZmJkYjk0YWU2YmJkMWE5OTg0MDM2OTZiZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnN2aWV3LCBOb3J0aCBQYXJrLCBVcHdvb2QgUGFyaywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjM1MjYyNWEyNjY4NDFjYTg5ZjkzNmVlZjEwNzM0Njguc2V0Q29udGVudChodG1sXzFmZmFjMjBmYmRiOTRhZTZiYmQxYTk5ODQwMzY5NmJmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzVlN2ZjMDkyMGY3NDRjYTE4ZGRjYWQ0MzIwYzhiN2QwLmJpbmRQb3B1cChwb3B1cF9mMzUyNjI1YTI2Njg0MWNhODlmOTM2ZWVmMTA3MzQ2OCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kYjVkYjE0ZGIyYzY0ZjY0Yjc2MTFlODY0YTlhNzczZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5MTExNTgsLTc5LjQ3NjAxMzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2FiNjNjOWVmNWE2NTRlOGU5ZmM1NmI4NmE5ZGMyYmE0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzllZTFjNDg0MGE5NzQzOWZiNjdiNDUwMjVlODkwZGQxID0gJCgnPGRpdiBpZD0iaHRtbF85ZWUxYzQ4NDBhOTc0MzlmYjY3YjQ1MDI1ZTg5MGRkMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGVsIFJheSwgS2VlbGVzZGFsZSwgTW91bnQgRGVubmlzLCBTaWx2ZXJ0aG9ybiwgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYWI2M2M5ZWY1YTY1NGU4ZTlmYzU2Yjg2YTlkYzJiYTQuc2V0Q29udGVudChodG1sXzllZTFjNDg0MGE5NzQzOWZiNjdiNDUwMjVlODkwZGQxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RiNWRiMTRkYjJjNjRmNjRiNzYxMWU4NjRhOWE3NzNmLmJpbmRQb3B1cChwb3B1cF9hYjYzYzllZjVhNjU0ZThlOWZjNTZiODZhOWRjMmJhNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jZGM0ZTJiZDI2NjI0MWZkYTBlMmFhNmNkZGU4YThkYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3MzE4NTI5OTk5OTk5LC03OS40ODcyNjE5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83YmUzNjYwNDBmOWE0ZTA5Yjg2OTZiZWEwZWViNTQzZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iNTg4YTQwZmI1NmY0NmJlYmI1ZTliODY1YzM3ZWFiYyA9ICQoJzxkaXYgaWQ9Imh0bWxfYjU4OGE0MGZiNTZmNDZiZWJiNWU5Yjg2NWMzN2VhYmMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBKdW5jdGlvbiBOb3J0aCwgUnVubnltZWRlLCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83YmUzNjYwNDBmOWE0ZTA5Yjg2OTZiZWEwZWViNTQzZC5zZXRDb250ZW50KGh0bWxfYjU4OGE0MGZiNTZmNDZiZWJiNWU5Yjg2NWMzN2VhYmMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2RjNGUyYmQyNjYyNDFmZGEwZTJhYTZjZGRlOGE4ZGIuYmluZFBvcHVwKHBvcHVwXzdiZTM2NjA0MGY5YTRlMDliODY5NmJlYTBlZWI1NDNkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZiNmYyOWJkM2NiMDQ5YmY4ZjU2MzcwOTEwZGU5NWQ4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYxNjA4MywtNzkuNDY0NzYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYmUzOGVlYzBlYTI3NDBkODk0OGZlNmIwZTUwNjU1NGEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjhhM2JlM2ViYzYwNDIzOWIxNzlhMTQyYzcwODg2NmMgPSAkKCc8ZGl2IGlkPSJodG1sX2Y4YTNiZTNlYmM2MDQyMzliMTc5YTE0MmM3MDg4NjZjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IaWdoIFBhcmssIFRoZSBKdW5jdGlvbiBTb3V0aCwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iZTM4ZWVjMGVhMjc0MGQ4OTQ4ZmU2YjBlNTA2NTU0YS5zZXRDb250ZW50KGh0bWxfZjhhM2JlM2ViYzYwNDIzOWIxNzlhMTQyYzcwODg2NmMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmI2ZjI5YmQzY2IwNDliZjhmNTYzNzA5MTBkZTk1ZDguYmluZFBvcHVwKHBvcHVwX2JlMzhlZWMwZWEyNzQwZDg5NDhmZTZiMGU1MDY1NTRhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2VkMjYzMmQ1MDhhODQxM2FhZjI0MzhkNjk3OWYwNzUyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4OTU5NywtNzkuNDU2MzI1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzMwMGZiOWIyYTJkYzRhODk5ODJjNDRlZmJkNWE0YTIyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2VhYmYyN2FjMWU2YTQ5NTU5MzQ5ZmJhZTNjZWQ3NWE0ID0gJCgnPGRpdiBpZD0iaHRtbF9lYWJmMjdhYzFlNmE0OTU1OTM0OWZiYWUzY2VkNzVhNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFya2RhbGUsIFJvbmNlc3ZhbGxlcywgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zMDBmYjliMmEyZGM0YTg5OTgyYzQ0ZWZiZDVhNGEyMi5zZXRDb250ZW50KGh0bWxfZWFiZjI3YWMxZTZhNDk1NTkzNDlmYmFlM2NlZDc1YTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZWQyNjMyZDUwOGE4NDEzYWFmMjQzOGQ2OTc5ZjA3NTIuYmluZFBvcHVwKHBvcHVwXzMwMGZiOWIyYTJkYzRhODk5ODJjNDRlZmJkNWE0YTIyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E1MzQyYzNkYzUzODQ3ZTg5MDNmZmRmNjMxODU2OTg3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNTcwNiwtNzkuNDg0NDQ5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80Nzg4Mzc5NmZjNTE0NThhOTU1ZTZlZWUyMTY0ZGY2OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jMTBjNjIzMDg0OGM0ZDk2OGE4MTRjNTQ5YWU5OTkxYiA9ICQoJzxkaXYgaWQ9Imh0bWxfYzEwYzYyMzA4NDhjNGQ5NjhhODE0YzU0OWFlOTk5MWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJ1bm55bWVkZSwgU3dhbnNlYSwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80Nzg4Mzc5NmZjNTE0NThhOTU1ZTZlZWUyMTY0ZGY2OC5zZXRDb250ZW50KGh0bWxfYzEwYzYyMzA4NDhjNGQ5NjhhODE0YzU0OWFlOTk5MWIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTUzNDJjM2RjNTM4NDdlODkwM2ZmZGY2MzE4NTY5ODcuYmluZFBvcHVwKHBvcHVwXzQ3ODgzNzk2ZmM1MTQ1OGE5NTVlNmVlZTIxNjRkZjY4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E1NjljYTA2YjE5OTRmZDhhNzQ0ZjI3ZjFmMmE0MGRiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyMzAxNSwtNzkuMzg5NDkzOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iNDNmNjdmZmZiOTU0NTBiOWFiZmRlMDllNDdiMTNiOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iZGRhNjVkOGI3MTM0YTRlYmM0YzljMjU3OTUyM2RlZSA9ICQoJzxkaXYgaWQ9Imh0bWxfYmRkYTY1ZDhiNzEzNGE0ZWJjNGM5YzI1Nzk1MjNkZWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlF1ZWVuJiMzOTtzIFBhcmssIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I0M2Y2N2ZmZmI5NTQ1MGI5YWJmZGUwOWU0N2IxM2I5LnNldENvbnRlbnQoaHRtbF9iZGRhNjVkOGI3MTM0YTRlYmM0YzljMjU3OTUyM2RlZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hNTY5Y2EwNmIxOTk0ZmQ4YTc0NGYyN2YxZjJhNDBkYi5iaW5kUG9wdXAocG9wdXBfYjQzZjY3ZmZmYjk1NDUwYjlhYmZkZTA5ZTQ3YjEzYjkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTgxZTdlMzQ4NDM2NGM0NWE1MzRiMWY5ZDJjODRkMGIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzY5NjU2LC03OS42MTU4MTg5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83NTA3MzQ5MjhlZWM0MzBmYTM5MzI5OTY4NTMwNDVjMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zNTdhNDg3MTM1MTI0NzU3OWFjZDE0NWE3NGY3YTZhNCA9ICQoJzxkaXYgaWQ9Imh0bWxfMzU3YTQ4NzEzNTEyNDc1NzlhY2QxNDVhNzRmN2E2YTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNhbmFkYSBQb3N0IEdhdGV3YXkgUHJvY2Vzc2luZyBDZW50cmUsIE1pc3Npc3NhdWdhPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83NTA3MzQ5MjhlZWM0MzBmYTM5MzI5OTY4NTMwNDVjMi5zZXRDb250ZW50KGh0bWxfMzU3YTQ4NzEzNTEyNDc1NzlhY2QxNDVhNzRmN2E2YTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTgxZTdlMzQ4NDM2NGM0NWE1MzRiMWY5ZDJjODRkMGIuYmluZFBvcHVwKHBvcHVwXzc1MDczNDkyOGVlYzQzMGZhMzkzMjk5Njg1MzA0NWMyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U4ZDhjODZkMGJjYzQwNTM4MDQ0ODkyNmRkOGY2YWQ2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNzQzOSwtNzkuMzIxNTU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZjOGU1ZDk4YmZiNTRkOTU5MDU3NmU2NDU1ZGMyYzA2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzE0MTJjM2E5ZTljMjQ1ZDk4N2I5YjI3MTc1NmRkY2U3ID0gJCgnPGRpdiBpZD0iaHRtbF8xNDEyYzNhOWU5YzI0NWQ5ODdiOWIyNzE3NTZkZGNlNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QnVzaW5lc3MgUmVwbHkgTWFpbCBQcm9jZXNzaW5nIENlbnRyZSA5NjkgRWFzdGVybiwgRWFzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82YzhlNWQ5OGJmYjU0ZDk1OTA1NzZlNjQ1NWRjMmMwNi5zZXRDb250ZW50KGh0bWxfMTQxMmMzYTllOWMyNDVkOTg3YjliMjcxNzU2ZGRjZTcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZThkOGM4NmQwYmNjNDA1MzgwNDQ4OTI2ZGQ4ZjZhZDYuYmluZFBvcHVwKHBvcHVwXzZjOGU1ZDk4YmZiNTRkOTU5MDU3NmU2NDU1ZGMyYzA2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzg2MjQ0ZWRiNWEwMDRiNDc4MDczZjliNWQxYTJhNGY5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjA1NjQ2NiwtNzkuNTAxMzIwNzAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjc5YTE3MjU0ODQyNDFlMThhN2NmYzQyNDczMTNmYzQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmExZDJhMTAwMmE5NGRhZjhhNTk0M2FiYjIzOThiMDAgPSAkKCc8ZGl2IGlkPSJodG1sX2JhMWQyYTEwMDJhOTRkYWY4YTU5NDNhYmIyMzk4YjAwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IdW1iZXIgQmF5IFNob3JlcywgTWltaWNvIFNvdXRoLCBOZXcgVG9yb250bywgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82NzlhMTcyNTQ4NDI0MWUxOGE3Y2ZjNDI0NzMxM2ZjNC5zZXRDb250ZW50KGh0bWxfYmExZDJhMTAwMmE5NGRhZjhhNTk0M2FiYjIzOThiMDApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODYyNDRlZGI1YTAwNGI0NzgwNzNmOWI1ZDFhMmE0ZjkuYmluZFBvcHVwKHBvcHVwXzY3OWExNzI1NDg0MjQxZTE4YTdjZmM0MjQ3MzEzZmM0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzczYmIwZTZhZGVkMDRlZTNhODg0Y2NjMTNlM2FjZWY2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjAyNDEzNzAwMDAwMDEsLTc5LjU0MzQ4NDA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA4NDlhYmUzNTM4MDQ5OThhZWVhOTcyYzQwYmQ3YWZlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2UwYzE3YTY0YWNkODQ5MWE4YmRjZWM2YmU4NDY0M2I4ID0gJCgnPGRpdiBpZD0iaHRtbF9lMGMxN2E2NGFjZDg0OTFhOGJkY2VjNmJlODQ2NDNiOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QWxkZXJ3b29kLCBMb25nIEJyYW5jaCwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wODQ5YWJlMzUzODA0OTk4YWVlYTk3MmM0MGJkN2FmZS5zZXRDb250ZW50KGh0bWxfZTBjMTdhNjRhY2Q4NDkxYThiZGNlYzZiZTg0NjQzYjgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzNiYjBlNmFkZWQwNGVlM2E4ODRjY2MxM2UzYWNlZjYuYmluZFBvcHVwKHBvcHVwXzA4NDlhYmUzNTM4MDQ5OThhZWVhOTcyYzQwYmQ3YWZlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E2MDg3YzJlMjJlYzQ3OWM4N2RmNDM5NjI0NDlhNmRhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUzNjUzNjAwMDAwMDA1LC03OS41MDY5NDM2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzE1YmRhNzZiY2Y0OTQ0ODY4NWZjZmI2NzllMDA3ZDRhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzY3ZmJiOWRhNzgyYzRkYjBhMjRkMGY5N2IxMGI3MzVlID0gJCgnPGRpdiBpZD0iaHRtbF82N2ZiYjlkYTc4MmM0ZGIwYTI0ZDBmOTdiMTBiNzM1ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEtpbmdzd2F5LCBNb250Z29tZXJ5IFJvYWQsIE9sZCBNaWxsIE5vcnRoLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE1YmRhNzZiY2Y0OTQ0ODY4NWZjZmI2NzllMDA3ZDRhLnNldENvbnRlbnQoaHRtbF82N2ZiYjlkYTc4MmM0ZGIwYTI0ZDBmOTdiMTBiNzM1ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hNjA4N2MyZTIyZWM0NzljODdkZjQzOTYyNDQ5YTZkYS5iaW5kUG9wdXAocG9wdXBfMTViZGE3NmJjZjQ5NDQ4Njg1ZmNmYjY3OWUwMDdkNGEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjE1OTMxNjBkZGQ3NGYyMWE3NjQ5NTVmMGU3N2FlYTcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzYyNTc5LC03OS40OTg1MDkwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yY2NjYzgwYWY0MjI0MmYxOWYyMjgzN2UyMTUzYzljMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yNzc4MTZmMmZiYmM0N2RkOGFhOWIwNDAxNWE1YzUzOCA9ICQoJzxkaXYgaWQ9Imh0bWxfMjc3ODE2ZjJmYmJjNDdkZDhhYTliMDQwMTVhNWM1MzgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkh1bWJlciBCYXksIEtpbmcmIzM5O3MgTWlsbCBQYXJrLCBLaW5nc3dheSBQYXJrIFNvdXRoIEVhc3QsIE1pbWljbyBORSwgT2xkIE1pbGwgU291dGgsIFRoZSBRdWVlbnN3YXkgRWFzdCwgUm95YWwgWW9yayBTb3V0aCBFYXN0LCBTdW5ueWxlYSwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yY2NjYzgwYWY0MjI0MmYxOWYyMjgzN2UyMTUzYzljMy5zZXRDb250ZW50KGh0bWxfMjc3ODE2ZjJmYmJjNDdkZDhhYTliMDQwMTVhNWM1MzgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjE1OTMxNjBkZGQ3NGYyMWE3NjQ5NTVmMGU3N2FlYTcuYmluZFBvcHVwKHBvcHVwXzJjY2NjODBhZjQyMjQyZjE5ZjIyODM3ZTIxNTNjOWMzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E1MTY1OWIwOTI3MTQxNjc5ZGYzYTc3MDMzNDVhOTg1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjI4ODQwOCwtNzkuNTIwOTk5NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjNlMTRlYWU4NjhhNGMxM2E4MTk2ZDMxZjdjMTRlZjIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTNkNDQ0YzNkMDllNGRiMmJkMGVlOTk5YmY0MWI3ZDMgPSAkKCc8ZGl2IGlkPSJodG1sXzEzZDQ0NGMzZDA5ZTRkYjJiZDBlZTk5OWJmNDFiN2QzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LaW5nc3dheSBQYXJrIFNvdXRoIFdlc3QsIE1pbWljbyBOVywgVGhlIFF1ZWVuc3dheSBXZXN0LCBSb3lhbCBZb3JrIFNvdXRoIFdlc3QsIFNvdXRoIG9mIEJsb29yLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2IzZTE0ZWFlODY4YTRjMTNhODE5NmQzMWY3YzE0ZWYyLnNldENvbnRlbnQoaHRtbF8xM2Q0NDRjM2QwOWU0ZGIyYmQwZWU5OTliZjQxYjdkMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hNTE2NTliMDkyNzE0MTY3OWRmM2E3NzAzMzQ1YTk4NS5iaW5kUG9wdXAocG9wdXBfYjNlMTRlYWU4NjhhNGMxM2E4MTk2ZDMxZjdjMTRlZjIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTUzMzBhMWMzNTg0NDE4OWI1Zjk5YmNkMTlkYTQ0NmUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njc4NTU2LC03OS41MzIyNDI0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83Zjg1NTI4OGNjMzg0YzUxOGQxNWEwZWFjYjg1ZmRhMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85Y2Y5YWZkOTQyZTE0Zjg5OGViMDQwZmU5ZDI5MTQ2MiA9ICQoJzxkaXYgaWQ9Imh0bWxfOWNmOWFmZDk0MmUxNGY4OThlYjA0MGZlOWQyOTE0NjIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlF1ZWVuJiMzOTtzIFBhcmssIFF1ZWVuJiMzOTtzIFBhcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzdmODU1Mjg4Y2MzODRjNTE4ZDE1YTBlYWNiODVmZGExLnNldENvbnRlbnQoaHRtbF85Y2Y5YWZkOTQyZTE0Zjg5OGViMDQwZmU5ZDI5MTQ2Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xNTMzMGExYzM1ODQ0MTg5YjVmOTliY2QxOWRhNDQ2ZS5iaW5kUG9wdXAocG9wdXBfN2Y4NTUyODhjYzM4NGM1MThkMTVhMGVhY2I4NWZkYTEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGNlOTYyZjUyMmYyNDk2ZWJhNzhlYTliNmU2YTdlZjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTA5NDMyLC03OS41NTQ3MjQ0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wNzY2ZGQ5Mzk5ODM0ZDE1OGU2OWI0MTFjZTc4NzI4OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mZWM0YjE3YmE3ZWE0ZThhYjJjNDk3YzJjOTQ1NGJkNSA9ICQoJzxkaXYgaWQ9Imh0bWxfZmVjNGIxN2JhN2VhNGU4YWIyYzQ5N2MyYzk0NTRiZDUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNsb3ZlcmRhbGUsIElzbGluZ3RvbiwgTWFydGluIEdyb3ZlLCBQcmluY2VzcyBHYXJkZW5zLCBXZXN0IERlYW5lIFBhcmssIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDc2NmRkOTM5OTgzNGQxNThlNjliNDExY2U3ODcyODguc2V0Q29udGVudChodG1sX2ZlYzRiMTdiYTdlYTRlOGFiMmM0OTdjMmM5NDU0YmQ1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RjZTk2MmY1MjJmMjQ5NmViYTc4ZWE5YjZlNmE3ZWYxLmJpbmRQb3B1cChwb3B1cF8wNzY2ZGQ5Mzk5ODM0ZDE1OGU2OWI0MTFjZTc4NzI4OCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kZTg2YmU2ZWViMTA0YmExYjE1ZTY5ZjY0MmZkMDgxOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0MzUxNTIsLTc5LjU3NzIwMDc5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2FlNjRlOTkwNjE3NTRkODI4YmZjNGMyMmFlNzlmYTdlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I5NmE1MDYzODE4YjRiYmNiNzMyNmQxNjczZGI5M2JlID0gJCgnPGRpdiBpZD0iaHRtbF9iOTZhNTA2MzgxOGI0YmJjYjczMjZkMTY3M2RiOTNiZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Qmxvb3JkYWxlIEdhcmRlbnMsIEVyaW5nYXRlLCBNYXJrbGFuZCBXb29kLCBPbGQgQnVybmhhbXRob3JwZSwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hZTY0ZTk5MDYxNzU0ZDgyOGJmYzRjMjJhZTc5ZmE3ZS5zZXRDb250ZW50KGh0bWxfYjk2YTUwNjM4MThiNGJiY2I3MzI2ZDE2NzNkYjkzYmUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZGU4NmJlNmVlYjEwNGJhMWIxNWU2OWY2NDJmZDA4MTkuYmluZFBvcHVwKHBvcHVwX2FlNjRlOTkwNjE3NTRkODI4YmZjNGMyMmFlNzlmYTdlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U3MjZhZDZmYzAwZDQ5ZTRiNGE5MzUwZGViMTBiYWQ3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzU2MzAzMywtNzkuNTY1OTYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGEzOWZhOTA1YWM2NGIyZGJiZTc3ZTViODgxYmQ0ZGEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTc1YTcxZGFmMTVhNDc4ZDgxNTAzZDk2OGIwODQ0YjMgPSAkKCc8ZGl2IGlkPSJodG1sXzE3NWE3MWRhZjE1YTQ3OGQ4MTUwM2Q5NjhiMDg0NGIzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IdW1iZXIgU3VtbWl0LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80YTM5ZmE5MDVhYzY0YjJkYmJlNzdlNWI4ODFiZDRkYS5zZXRDb250ZW50KGh0bWxfMTc1YTcxZGFmMTVhNDc4ZDgxNTAzZDk2OGIwODQ0YjMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTcyNmFkNmZjMDBkNDllNGI0YTkzNTBkZWIxMGJhZDcuYmluZFBvcHVwKHBvcHVwXzRhMzlmYTkwNWFjNjRiMmRiYmU3N2U1Yjg4MWJkNGRhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U2ODlhY2JmOWM3NjQ4NzI5YTVlN2IzOTdlYzU5ZTgxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI0NzY1OSwtNzkuNTMyMjQyNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTEwOTJmODY2ZDg5NDkzN2E5MTA0YzUyY2Y1NzZjNGMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODU5MzAzZTc5MDc5NGQxOTlhNDhhYzQ4Njk2MWUyYjMgPSAkKCc8ZGl2IGlkPSJodG1sXzg1OTMwM2U3OTA3OTRkMTk5YTQ4YWM0ODY5NjFlMmIzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FbWVyeSwgSHVtYmVybGVhLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MTA5MmY4NjZkODk0OTM3YTkxMDRjNTJjZjU3NmM0Yy5zZXRDb250ZW50KGh0bWxfODU5MzAzZTc5MDc5NGQxOTlhNDhhYzQ4Njk2MWUyYjMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTY4OWFjYmY5Yzc2NDg3MjlhNWU3YjM5N2VjNTllODEuYmluZFBvcHVwKHBvcHVwXzkxMDkyZjg2NmQ4OTQ5MzdhOTEwNGM1MmNmNTc2YzRjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzljNzFlYzY0YWNkZDQ3Y2FhZGM4MTgzNWI1NDc4NmM2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA2ODc2LC03OS41MTgxODg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lNzVmYzBmZWUxOWY0NzQwYjllNDM2NDIzOGQ1MTBkYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kNDYzY2E1NWVjODQ0YTA2OWI3OTAyM2JmMmU3MjEzYSA9ICQoJzxkaXYgaWQ9Imh0bWxfZDQ2M2NhNTVlYzg0NGEwNjliNzkwMjNiZjJlNzIxM2EiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3RvbiwgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTc1ZmMwZmVlMTlmNDc0MGI5ZTQzNjQyMzhkNTEwZGIuc2V0Q29udGVudChodG1sX2Q0NjNjYTU1ZWM4NDRhMDY5Yjc5MDIzYmYyZTcyMTNhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzljNzFlYzY0YWNkZDQ3Y2FhZGM4MTgzNWI1NDc4NmM2LmJpbmRQb3B1cChwb3B1cF9lNzVmYzBmZWUxOWY0NzQwYjllNDM2NDIzOGQ1MTBkYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lNTJjMzk3ZDljYTc0MmQyODU3YjE2N2FmYzZmMmMzZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5NjMxOSwtNzkuNTMyMjQyNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYTY5YmFjMDQyZDEwNGQ4Y2E0ZTA2MjMzMTlmNzAzZjUpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTU5ZmFhNGFkNTllNDMxOTg5NWEzZWZjYjgyMTJkNjQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjdlMjYxYjk2YzUzNGNlNmEzNTY1N2Q0OGViN2RlNDAgPSAkKCc8ZGl2IGlkPSJodG1sX2I3ZTI2MWI5NmM1MzRjZTZhMzU2NTdkNDhlYjdkZTQwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XZXN0bW91bnQsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTU5ZmFhNGFkNTllNDMxOTg5NWEzZWZjYjgyMTJkNjQuc2V0Q29udGVudChodG1sX2I3ZTI2MWI5NmM1MzRjZTZhMzU2NTdkNDhlYjdkZTQwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U1MmMzOTdkOWNhNzQyZDI4NTdiMTY3YWZjNmYyYzNlLmJpbmRQb3B1cChwb3B1cF9hNTlmYWE0YWQ1OWU0MzE5ODk1YTNlZmNiODIxMmQ2NCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84MjlmYWU4NzEyNGE0YjdiOTdiOWJmMmVkNzNhY2UwZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4ODkwNTQsLTc5LjU1NDcyNDQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E2OWJhYzA0MmQxMDRkOGNhNGUwNjIzMzE5ZjcwM2Y1KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2M5YjZlOTlmYjI1OTRjZGViNzEwZDkxODNiZDYyZmI0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzFhNzgzY2NlZWQ4YTRkZjZhMmE1YmVhMDViMmE3NmFjID0gJCgnPGRpdiBpZD0iaHRtbF8xYTc4M2NjZWVkOGE0ZGY2YTJhNWJlYTA1YjJhNzZhYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2luZ3N2aWV3IFZpbGxhZ2UsIE1hcnRpbiBHcm92ZSBHYXJkZW5zLCBSaWNodmlldyBHYXJkZW5zLCBTdC4gUGhpbGxpcHMsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzliNmU5OWZiMjU5NGNkZWI3MTBkOTE4M2JkNjJmYjQuc2V0Q29udGVudChodG1sXzFhNzgzY2NlZWQ4YTRkZjZhMmE1YmVhMDViMmE3NmFjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzgyOWZhZTg3MTI0YTRiN2I5N2I5YmYyZWQ3M2FjZTBlLmJpbmRQb3B1cChwb3B1cF9jOWI2ZTk5ZmIyNTk0Y2RlYjcxMGQ5MTgzYmQ2MmZiNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iYmJhNjM2OWIwOWM0OGE5ODUwOTU2NTM3ZTZkZGQ1NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczOTQxNjM5OTk5OTk5NiwtNzkuNTg4NDM2OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zMzVkNjQ2OGZkZWY0MDQzOTczZjBmZTVmYmVjNTJmNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mYWMxOTk3ODcxYjI0NTc3ODEyOGE3ZjNkYTllZDU3YyA9ICQoJzxkaXYgaWQ9Imh0bWxfZmFjMTk5Nzg3MWIyNDU3NzgxMjhhN2YzZGE5ZWQ1N2MiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFsYmlvbiBHYXJkZW5zLCBCZWF1bW9uZCBIZWlnaHRzLCBIdW1iZXJnYXRlLCBKYW1lc3Rvd24sIE1vdW50IE9saXZlLCBTaWx2ZXJzdG9uZSwgU291dGggU3RlZWxlcywgVGhpc3RsZXRvd24sIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzM1ZDY0NjhmZGVmNDA0Mzk3M2YwZmU1ZmJlYzUyZjQuc2V0Q29udGVudChodG1sX2ZhYzE5OTc4NzFiMjQ1Nzc4MTI4YTdmM2RhOWVkNTdjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2JiYmE2MzY5YjA5YzQ4YTk4NTA5NTY1MzdlNmRkZDU1LmJpbmRQb3B1cChwb3B1cF8zMzVkNjQ2OGZkZWY0MDQzOTczZjBmZTVmYmVjNTJmNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mOGYxMTdkODVkZDY0NjhkODc1MmIxMTQwZmIyZmU1MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNjc0ODI5OTk5OTk5NCwtNzkuNTk0MDU0NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hNjliYWMwNDJkMTA0ZDhjYTRlMDYyMzMxOWY3MDNmNSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yOTM2MGM0YTcwMWI0OTI3ODBiMjBhNGVlN2IwMDYzOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80NzQ3NzFiZjY4NTM0MGM2YmMyN2NlNWI0ODA2ODhjYSA9ICQoJzxkaXYgaWQ9Imh0bWxfNDc0NzcxYmY2ODUzNDBjNmJjMjdjZTViNDgwNjg4Y2EiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5vcnRod2VzdCwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yOTM2MGM0YTcwMWI0OTI3ODBiMjBhNGVlN2IwMDYzOS5zZXRDb250ZW50KGh0bWxfNDc0NzcxYmY2ODUzNDBjNmJjMjdjZTViNDgwNjg4Y2EpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjhmMTE3ZDg1ZGQ2NDY4ZDg3NTJiMTE0MGZiMmZlNTIuYmluZFBvcHVwKHBvcHVwXzI5MzYwYzRhNzAxYjQ5Mjc4MGIyMGE0ZWU3YjAwNjM5KTsKCiAgICAgICAgICAgIAogICAgICAgIAo8L3NjcmlwdD4= onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



###  Filter only boroughs that contain the word Toronto 


```python
# filter borough names that contain the word Toronto
borough_names = list(df_merged.Borough.unique())

borough_with_toronto = []

for x in borough_names:
    if "toronto" in x.lower():
        borough_with_toronto.append(x)
        
borough_with_toronto
```




    ['East Toronto', 'Central Toronto', 'Downtown Toronto', 'West Toronto']




```python
# create a new DataFrame with only boroughs that contain the word Toronto
toronto_df_new = df_merged[df_merged['Borough'].isin(borough_with_toronto)].reset_index(drop=True)
print(toronto_df_new.shape)
toronto_df_new.head()
```

    (39, 5)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>The Beaches West, India Bazaar</td>
      <td>43.668999</td>
      <td>-79.315572</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
      <td>43.659526</td>
      <td>-79.340923</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
      <td>43.728020</td>
      <td>-79.388790</td>
    </tr>
  </tbody>
</table>
</div>




```python
# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(toronto_df_new['Latitude'], toronto_df_new['Longitude'], toronto_df_new['Borough'], toronto_df_new['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7).add_to(map_toronto)  
    
map_toronto 
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2JhMTI2ODQ0OTI4MDRkNTg5NDcwYmZkNGY5YTE2NDk2IiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF9iYTEyNjg0NDkyODA0ZDU4OTQ3MGJmZDRmOWExNjQ5NiA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF9iYTEyNjg0NDkyODA0ZDU4OTQ3MGJmZDRmOWExNjQ5NicsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjUzOTYzLC03OS4zODcyMDddLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgem9vbTogMTAsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYXhCb3VuZHM6IGJvdW5kcywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGxheWVyczogW10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3b3JsZENvcHlKdW1wOiBmYWxzZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSk7CiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyX2UxMGZkZWU3YTIwNDRjYjhiNTQ3YzQ4ZjVhZDc5NzFhID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgICAgICAgICAgewogICJhdHRyaWJ1dGlvbiI6IG51bGwsCiAgImRldGVjdFJldGluYSI6IGZhbHNlLAogICJtYXhab29tIjogMTgsCiAgIm1pblpvb20iOiAxLAogICJub1dyYXAiOiBmYWxzZSwKICAic3ViZG9tYWlucyI6ICJhYmMiCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2JhMTI2ODQ0OTI4MDRkNTg5NDcwYmZkNGY5YTE2NDk2KTsKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85OWJmNTM3YWQwZTE0ODMxYjg5M2U2ZjlhMTk5YzdhOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3NjM1NzM5OTk5OTk5LC03OS4yOTMwMzEyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2JhMTI2ODQ0OTI4MDRkNTg5NDcwYmZkNGY5YTE2NDk2KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2MxYmQxMTIyY2RiMzQ1ZWU5ZTI0ZjhmZjg4NDBhMjU0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQyOTA4NDQ1YzM2ZDRmYzBhM2ZkNWQ4YzUyYjFlMmVmID0gJCgnPGRpdiBpZD0iaHRtbF80MjkwODQ0NWMzNmQ0ZmMwYTNmZDVkOGM1MmIxZTJlZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEJlYWNoZXMsIEVhc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzFiZDExMjJjZGIzNDVlZTllMjRmOGZmODg0MGEyNTQuc2V0Q29udGVudChodG1sXzQyOTA4NDQ1YzM2ZDRmYzBhM2ZkNWQ4YzUyYjFlMmVmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk5YmY1MzdhZDBlMTQ4MzFiODkzZTZmOWExOTljN2E4LmJpbmRQb3B1cChwb3B1cF9jMWJkMTEyMmNkYjM0NWVlOWUyNGY4ZmY4ODQwYTI1NCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83NzgwMDE1YTI3MTY0ODhhOTRmNTk5OWE4MzI1OTFlNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3OTU1NzEsLTc5LjM1MjE4OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iYTEyNjg0NDkyODA0ZDU4OTQ3MGJmZDRmOWExNjQ5Nik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82YWY0MDFjM2M0YWY0Zjc4YmNhNDlkM2Q2MjNhMGVmZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83NDA4MzE4Y2I1Y2Y0ZDFiYTIzOWZhNjdiY2I2Y2Y1NiA9ICQoJzxkaXYgaWQ9Imh0bWxfNzQwODMxOGNiNWNmNGQxYmEyMzlmYTY3YmNiNmNmNTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBEYW5mb3J0aCBXZXN0LCBSaXZlcmRhbGUsIEVhc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmFmNDAxYzNjNGFmNGY3OGJjYTQ5ZDNkNjIzYTBlZmQuc2V0Q29udGVudChodG1sXzc0MDgzMThjYjVjZjRkMWJhMjM5ZmE2N2JjYjZjZjU2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzc3ODAwMTVhMjcxNjQ4OGE5NGY1OTk5YTgzMjU5MWU3LmJpbmRQb3B1cChwb3B1cF82YWY0MDFjM2M0YWY0Zjc4YmNhNDlkM2Q2MjNhMGVmZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iZDQyNzkzZWY0OWI0ODY4YmZhMDk0MjNkMDcxYmU3NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2ODk5ODUsLTc5LjMxNTU3MTU5OTk5OTk4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2JhMTI2ODQ0OTI4MDRkNTg5NDcwYmZkNGY5YTE2NDk2KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzM4MTAwOTFmMDA3MTRlNjg5OThmMGRlODNhMjdhZThjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2VmZjVhMDhkNjFlZDRhZjlhZDRlOTc1ODQ0MGYxMjE5ID0gJCgnPGRpdiBpZD0iaHRtbF9lZmY1YTA4ZDYxZWQ0YWY5YWQ0ZTk3NTg0NDBmMTIxOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEJlYWNoZXMgV2VzdCwgSW5kaWEgQmF6YWFyLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM4MTAwOTFmMDA3MTRlNjg5OThmMGRlODNhMjdhZThjLnNldENvbnRlbnQoaHRtbF9lZmY1YTA4ZDYxZWQ0YWY5YWQ0ZTk3NTg0NDBmMTIxOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iZDQyNzkzZWY0OWI0ODY4YmZhMDk0MjNkMDcxYmU3Ni5iaW5kUG9wdXAocG9wdXBfMzgxMDA5MWYwMDcxNGU2ODk5OGYwZGU4M2EyN2FlOGMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzZiZTc4MmQyNDNiNDJiZmJjMGM5MTk1Y2QyZDkzNDggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTk1MjU1LC03OS4zNDA5MjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjMzYTk2ZDJlNTgwNDY3YTkxMTA4YTc4MWEwMjBjOTEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWY3YzcyZGYxZjE2NDZiMWE4YWQwNzU0NDE3NDk5YTEgPSAkKCc8ZGl2IGlkPSJodG1sXzlmN2M3MmRmMWYxNjQ2YjFhOGFkMDc1NDQxNzQ5OWExIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdHVkaW8gRGlzdHJpY3QsIEVhc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjMzYTk2ZDJlNTgwNDY3YTkxMTA4YTc4MWEwMjBjOTEuc2V0Q29udGVudChodG1sXzlmN2M3MmRmMWYxNjQ2YjFhOGFkMDc1NDQxNzQ5OWExKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzc2YmU3ODJkMjQzYjQyYmZiYzBjOTE5NWNkMmQ5MzQ4LmJpbmRQb3B1cChwb3B1cF9iMzNhOTZkMmU1ODA0NjdhOTExMDhhNzgxYTAyMGM5MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yODIyMjg3NDU4Mjk0NmQxYjI3YThlMmFmMzUxOWM3MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyODAyMDUsLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDkzN2E5ZTFlYjNkNDQ4MTgxODBmZDI4OWVlZjI2ODEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzdkNzVjZTJlYWI2NGQ5MmJiZmY3YmYyNjU1Y2YyMzcgPSAkKCc8ZGl2IGlkPSJodG1sXzc3ZDc1Y2UyZWFiNjRkOTJiYmZmN2JmMjY1NWNmMjM3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MYXdyZW5jZSBQYXJrLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ5MzdhOWUxZWIzZDQ0ODE4MTgwZmQyODllZWYyNjgxLnNldENvbnRlbnQoaHRtbF83N2Q3NWNlMmVhYjY0ZDkyYmJmZjdiZjI2NTVjZjIzNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yODIyMjg3NDU4Mjk0NmQxYjI3YThlMmFmMzUxOWM3MS5iaW5kUG9wdXAocG9wdXBfNDkzN2E5ZTFlYjNkNDQ4MTgxODBmZDI4OWVlZjI2ODEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTVlZTI1NWVhYzdmNDAyZGIwYTViNTc3ZDE2OWU1MDYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTI3NTExLC03OS4zOTAxOTc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2JhMTI2ODQ0OTI4MDRkNTg5NDcwYmZkNGY5YTE2NDk2KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzcyOWVhZDdhYmI1NTQ5YWE4ZWU5MGY1Yzk0Y2FhYjU2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzMyMDgxZmY3OWRiZjRjNmI5MWE0NjZkZjE3MGQ0MDk0ID0gJCgnPGRpdiBpZD0iaHRtbF8zMjA4MWZmNzlkYmY0YzZiOTFhNDY2ZGYxNzBkNDA5NCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGF2aXN2aWxsZSBOb3J0aCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83MjllYWQ3YWJiNTU0OWFhOGVlOTBmNWM5NGNhYWI1Ni5zZXRDb250ZW50KGh0bWxfMzIwODFmZjc5ZGJmNGM2YjkxYTQ2NmRmMTcwZDQwOTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTVlZTI1NWVhYzdmNDAyZGIwYTViNTc3ZDE2OWU1MDYuYmluZFBvcHVwKHBvcHVwXzcyOWVhZDdhYmI1NTQ5YWE4ZWU5MGY1Yzk0Y2FhYjU2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc0Yzg3OTYyZDVhZjQ2MzliZTM3OGI4NGM4NGZiNjg1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzE1MzgzNCwtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGI4ZjU1YjMyOTQwNGMwMDk0NTc2NzMwOTRmODIyMGUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODE4ZTkzNjI0MjI4NDhlYWFmMTUyMjkxY2Y3YzMzOWEgPSAkKCc8ZGl2IGlkPSJodG1sXzgxOGU5MzYyNDIyODQ4ZWFhZjE1MjI5MWNmN2MzMzlhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aCBUb3JvbnRvIFdlc3QsIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGI4ZjU1YjMyOTQwNGMwMDk0NTc2NzMwOTRmODIyMGUuc2V0Q29udGVudChodG1sXzgxOGU5MzYyNDIyODQ4ZWFhZjE1MjI5MWNmN2MzMzlhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzc0Yzg3OTYyZDVhZjQ2MzliZTM3OGI4NGM4NGZiNjg1LmJpbmRQb3B1cChwb3B1cF80YjhmNTViMzI5NDA0YzAwOTQ1NzY3MzA5NGY4MjIwZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lZmUxM2UyYTk0ODM0OGU2OGJhMTU3NmFlNjkxODUyNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNDMyNDQsLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTY1MTg0YWNiNTY4NDRjMWE0ZTUwNDZkYmQ2Y2YyNzAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODIxMTAzNmFlNTBlNDg5Njk2OGUzYjM5MWE5NGM3ZTIgPSAkKCc8ZGl2IGlkPSJodG1sXzgyMTEwMzZhZTUwZTQ4OTY5NjhlM2IzOTFhOTRjN2UyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EYXZpc3ZpbGxlLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzU2NTE4NGFjYjU2ODQ0YzFhNGU1MDQ2ZGJkNmNmMjcwLnNldENvbnRlbnQoaHRtbF84MjExMDM2YWU1MGU0ODk2OTY4ZTNiMzkxYTk0YzdlMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lZmUxM2UyYTk0ODM0OGU2OGJhMTU3NmFlNjkxODUyNS5iaW5kUG9wdXAocG9wdXBfNTY1MTg0YWNiNTY4NDRjMWE0ZTUwNDZkYmQ2Y2YyNzApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGU3Y2RhYjJhZWYxNDVmYWE2NmNjZDBkNDJkODQxYjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODk1NzQzLC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iYTEyNjg0NDkyODA0ZDU4OTQ3MGJmZDRmOWExNjQ5Nik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84NTM3N2UzMDAzNWM0OTI0OTg0ZGJjZGNhZGNkMjdlMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80Nzg0YjNkMjM3MDQ0MTZiOGVlOGE1NmMxZDEwYmU4OSA9ICQoJzxkaXYgaWQ9Imh0bWxfNDc4NGIzZDIzNzA0NDE2YjhlZThhNTZjMWQxMGJlODkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1vb3JlIFBhcmssIFN1bW1lcmhpbGwgRWFzdCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84NTM3N2UzMDAzNWM0OTI0OTg0ZGJjZGNhZGNkMjdlMy5zZXRDb250ZW50KGh0bWxfNDc4NGIzZDIzNzA0NDE2YjhlZThhNTZjMWQxMGJlODkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOGU3Y2RhYjJhZWYxNDVmYWE2NmNjZDBkNDJkODQxYjEuYmluZFBvcHVwKHBvcHVwXzg1Mzc3ZTMwMDM1YzQ5MjQ5ODRkYmNkY2FkY2QyN2UzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzYyYjU4MDY3N2E5NTRjYTc5ZDhiZDg5YjlmMjFiMWIwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg2NDEyMjk5OTk5OTksLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjA4MWRiMWFmZTU0NDY1OWFiODk5NjVhMTEyYTg1YWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYWEyZWUzNmQzNmY1NDNmZjhkYWQ3Y2U0MGI4NTI2ZDYgPSAkKCc8ZGl2IGlkPSJodG1sX2FhMmVlMzZkMzZmNTQzZmY4ZGFkN2NlNDBiODUyNmQ2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EZWVyIFBhcmssIEZvcmVzdCBIaWxsIFNFLCBSYXRobmVsbHksIFNvdXRoIEhpbGwsIFN1bW1lcmhpbGwgV2VzdCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iMDgxZGIxYWZlNTQ0NjU5YWI4OTk2NWExMTJhODVhYi5zZXRDb250ZW50KGh0bWxfYWEyZWUzNmQzNmY1NDNmZjhkYWQ3Y2U0MGI4NTI2ZDYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjJiNTgwNjc3YTk1NGNhNzlkOGJkODliOWYyMWIxYjAuYmluZFBvcHVwKHBvcHVwX2IwODFkYjFhZmU1NDQ2NTlhYjg5OTY1YTExMmE4NWFiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc1NjY4YWY1NzE3NTRkMTViYTM1ODcyMjI3YTU1YmM4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc5NTYyNiwtNzkuMzc3NTI5NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmFmN2QyZDA4ODY0NDAwNzg3YmQ2OWMxMTY2ZjA4ZjQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDk5OGMwYWI1ODE2NGUwYjg1YmQyM2IyZWM1ODc3ZWIgPSAkKCc8ZGl2IGlkPSJodG1sXzA5OThjMGFiNTgxNjRlMGI4NWJkMjNiMmVjNTg3N2ViIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb3NlZGFsZSwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmFmN2QyZDA4ODY0NDAwNzg3YmQ2OWMxMTY2ZjA4ZjQuc2V0Q29udGVudChodG1sXzA5OThjMGFiNTgxNjRlMGI4NWJkMjNiMmVjNTg3N2ViKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzc1NjY4YWY1NzE3NTRkMTViYTM1ODcyMjI3YTU1YmM4LmJpbmRQb3B1cChwb3B1cF82YWY3ZDJkMDg4NjQ0MDA3ODdiZDY5YzExNjZmMDhmNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jMDZkZWMwZTQzNjQ0YWYwOWM3ZTI2ZmFiNmE1OWYyMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzk2NywtNzkuMzY3Njc1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iYTEyNjg0NDkyODA0ZDU4OTQ3MGJmZDRmOWExNjQ5Nik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMWU4ODJmN2M3MDQ0MTE2YTU1ZmQ3ZTFjOGM0ODE1ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81YmM3NWJmZWRmZjU0ZGM4OTQxMDZkNTllOTM5ZTg0MiA9ICQoJzxkaXYgaWQ9Imh0bWxfNWJjNzViZmVkZmY1NGRjODk0MTA2ZDU5ZTkzOWU4NDIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNhYmJhZ2V0b3duLCBTdC4gSmFtZXMgVG93biwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTFlODgyZjdjNzA0NDExNmE1NWZkN2UxYzhjNDgxNWUuc2V0Q29udGVudChodG1sXzViYzc1YmZlZGZmNTRkYzg5NDEwNmQ1OWU5MzllODQyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2MwNmRlYzBlNDM2NDRhZjA5YzdlMjZmYWI2YTU5ZjIxLmJpbmRQb3B1cChwb3B1cF9lMWU4ODJmN2M3MDQ0MTE2YTU1ZmQ3ZTFjOGM0ODE1ZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xMGU0MDdiMzU3ZWU0YzUxOWNlMWRlZGNmMjBlYTY3YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2NTg1OTksLTc5LjM4MzE1OTkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2JhMTI2ODQ0OTI4MDRkNTg5NDcwYmZkNGY5YTE2NDk2KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRmZTQwMjdiNDkzYzRlMmViYzFjN2M5YjgzYTAzNGI4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhlODM4MDkzMjI2NjQ2OTNhZDk4ZThmNGMyNDlhNzMxID0gJCgnPGRpdiBpZD0iaHRtbF84ZTgzODA5MzIyNjY0NjkzYWQ5OGU4ZjRjMjQ5YTczMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2h1cmNoIGFuZCBXZWxsZXNsZXksIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRmZTQwMjdiNDkzYzRlMmViYzFjN2M5YjgzYTAzNGI4LnNldENvbnRlbnQoaHRtbF84ZTgzODA5MzIyNjY0NjkzYWQ5OGU4ZjRjMjQ5YTczMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xMGU0MDdiMzU3ZWU0YzUxOWNlMWRlZGNmMjBlYTY3Yy5iaW5kUG9wdXAocG9wdXBfNGZlNDAyN2I0OTNjNGUyZWJjMWM3YzliODNhMDM0YjgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmUwY2YzMjA5YTA0NDQ5OTgzNzkwMjY4MzhhYzhiZmQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTQyNTk5LC03OS4zNjA2MzU5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2JhMTI2ODQ0OTI4MDRkNTg5NDcwYmZkNGY5YTE2NDk2KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U5N2FmNzE4NzNjNzRhOWFhY2FiNjhhNGY5MmUxNzU3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzY5MmNhZWQ0YTRhZTQ0MTViYmE2ZDkxMThjODgyMTM5ID0gJCgnPGRpdiBpZD0iaHRtbF82OTJjYWVkNGE0YWU0NDE1YmJhNmQ5MTE4Yzg4MjEzOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFyYm91cmZyb250LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lOTdhZjcxODczYzc0YTlhYWNhYjY4YTRmOTJlMTc1Ny5zZXRDb250ZW50KGh0bWxfNjkyY2FlZDRhNGFlNDQxNWJiYTZkOTExOGM4ODIxMzkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNmUwY2YzMjA5YTA0NDQ5OTgzNzkwMjY4MzhhYzhiZmQuYmluZFBvcHVwKHBvcHVwX2U5N2FmNzE4NzNjNzRhOWFhY2FiNjhhNGY5MmUxNzU3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E5NWJkMWFhMjg2ZjQ3MmZiMzhlN2JkZmY5OTg0ZTE5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3MTYxOCwtNzkuMzc4OTM3MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGI3MjQxMTEyZDY2NDcxOGJjMTcxZjljMDVmMmZkMzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDJhZTQxMDcxMzY3NGU1ZDk3MDA0NzQ5YjY5MjFhZDAgPSAkKCc8ZGl2IGlkPSJodG1sXzAyYWU0MTA3MTM2NzRlNWQ5NzAwNDc0OWI2OTIxYWQwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SeWVyc29uLCBHYXJkZW4gRGlzdHJpY3QsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRiNzI0MTExMmQ2NjQ3MThiYzE3MWY5YzA1ZjJmZDMzLnNldENvbnRlbnQoaHRtbF8wMmFlNDEwNzEzNjc0ZTVkOTcwMDQ3NDliNjkyMWFkMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hOTViZDFhYTI4NmY0NzJmYjM4ZTdiZGZmOTk4NGUxOS5iaW5kUG9wdXAocG9wdXBfNGI3MjQxMTEyZDY2NDcxOGJjMTcxZjljMDVmMmZkMzMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODY0MDU4ODYwYTA0NDZiN2FhODM4YTkyZmMwNjFjMTQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE0OTM5LC03OS4zNzU0MTc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2JhMTI2ODQ0OTI4MDRkNTg5NDcwYmZkNGY5YTE2NDk2KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E0YzI0OGY0YTFkYjQwODU4N2I0OTA4NTEzMzc2YmQ1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzE3YTIwNmVjNmUxODQ1ZTc4ZTZhOGM0NmJkNjZhYTQ3ID0gJCgnPGRpdiBpZD0iaHRtbF8xN2EyMDZlYzZlMTg0NWU3OGU2YThjNDZiZDY2YWE0NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3QuIEphbWVzIFRvd24sIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E0YzI0OGY0YTFkYjQwODU4N2I0OTA4NTEzMzc2YmQ1LnNldENvbnRlbnQoaHRtbF8xN2EyMDZlYzZlMTg0NWU3OGU2YThjNDZiZDY2YWE0Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84NjQwNTg4NjBhMDQ0NmI3YWE4MzhhOTJmYzA2MWMxNC5iaW5kUG9wdXAocG9wdXBfYTRjMjQ4ZjRhMWRiNDA4NTg3YjQ5MDg1MTMzNzZiZDUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTU5OGVmOWYwYjE3NGQxNGFmYWRjMzc2NjQ1NTI2MGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDQ3NzA3OTk5OTk5OTYsLTc5LjM3MzMwNjRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMGU5MDY3MTFkOTc1NDRjYTgwYzYxMzU5OTVlNzc5MTAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTRlYmNlNjZlOGU5NGQzMzgyZjNjYjFlNTZlMWJkYzIgPSAkKCc8ZGl2IGlkPSJodG1sX2U0ZWJjZTY2ZThlOTRkMzM4MmYzY2IxZTU2ZTFiZGMyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CZXJjenkgUGFyaywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMGU5MDY3MTFkOTc1NDRjYTgwYzYxMzU5OTVlNzc5MTAuc2V0Q29udGVudChodG1sX2U0ZWJjZTY2ZThlOTRkMzM4MmYzY2IxZTU2ZTFiZGMyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU1OThlZjlmMGIxNzRkMTRhZmFkYzM3NjY0NTUyNjBmLmJpbmRQb3B1cChwb3B1cF8wZTkwNjcxMWQ5NzU0NGNhODBjNjEzNTk5NWU3NzkxMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82ZjU1MGI3ZmNlZDU0YWU2Yjk0MGU3OTEwNDZmZjhlNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1Nzk1MjQsLTc5LjM4NzM4MjZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDNkYjNkNDQxYmM1NGU5Yjk3MTc0OTgyMDAwZWM3MGEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjA4YThkYjA2MzQ1NDc4Y2E3YjA4NzgwYTUxZGE0NzcgPSAkKCc8ZGl2IGlkPSJodG1sX2YwOGE4ZGIwNjM0NTQ3OGNhN2IwODc4MGE1MWRhNDc3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZW50cmFsIEJheSBTdHJlZXQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQzZGIzZDQ0MWJjNTRlOWI5NzE3NDk4MjAwMGVjNzBhLnNldENvbnRlbnQoaHRtbF9mMDhhOGRiMDYzNDU0NzhjYTdiMDg3ODBhNTFkYTQ3Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82ZjU1MGI3ZmNlZDU0YWU2Yjk0MGU3OTEwNDZmZjhlNy5iaW5kUG9wdXAocG9wdXBfNDNkYjNkNDQxYmM1NGU5Yjk3MTc0OTgyMDAwZWM3MGEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmQyYmM4NThhZjI5NGNkNjkwNzU5MDBlYjczYzE5NzggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTA1NzEyMDAwMDAwMSwtNzkuMzg0NTY3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iYTEyNjg0NDkyODA0ZDU4OTQ3MGJmZDRmOWExNjQ5Nik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kNDJmNjNiYTMzMmY0ZTU4OWM5ZmE1YmRmYmM5NDdlNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hZDY3MjdmZWRhZDM0MjZiYTJjMDBmYjlmODZjOGE2ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfYWQ2NzI3ZmVkYWQzNDI2YmEyYzAwZmI5Zjg2YzhhNmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFkZWxhaWRlLCBLaW5nLCBSaWNobW9uZCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDQyZjYzYmEzMzJmNGU1ODljOWZhNWJkZmJjOTQ3ZTcuc2V0Q29udGVudChodG1sX2FkNjcyN2ZlZGFkMzQyNmJhMmMwMGZiOWY4NmM4YTZkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZkMmJjODU4YWYyOTRjZDY5MDc1OTAwZWI3M2MxOTc4LmJpbmRQb3B1cChwb3B1cF9kNDJmNjNiYTMzMmY0ZTU4OWM5ZmE1YmRmYmM5NDdlNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iZmRmYjdmNTY4NzE0NDgwOWY3NjIyZTRjOTdjNjZiMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0MDgxNTcsLTc5LjM4MTc1MjI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2JhMTI2ODQ0OTI4MDRkNTg5NDcwYmZkNGY5YTE2NDk2KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzBkZjMwOGM0ZTIyYTQ0MTJhN2E4MmFkYjk5YmI4NDJiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q5MzcwZDEwYjA3YTQ0YjhhZjRlYzI1ZDRkYjIxMWJiID0gJCgnPGRpdiBpZD0iaHRtbF9kOTM3MGQxMGIwN2E0NGI4YWY0ZWMyNWQ0ZGIyMTFiYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFyYm91cmZyb250IEVhc3QsIFRvcm9udG8gSXNsYW5kcywgVW5pb24gU3RhdGlvbiwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMGRmMzA4YzRlMjJhNDQxMmE3YTgyYWRiOTliYjg0MmIuc2V0Q29udGVudChodG1sX2Q5MzcwZDEwYjA3YTQ0YjhhZjRlYzI1ZDRkYjIxMWJiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2JmZGZiN2Y1Njg3MTQ0ODA5Zjc2MjJlNGM5N2M2NmIzLmJpbmRQb3B1cChwb3B1cF8wZGYzMDhjNGUyMmE0NDEyYTdhODJhZGI5OWJiODQyYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84Mjk5M2ZjYTQ1ZTQ0M2UyYjdlODM3MTA3ZjNiZWNiOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzE3NjgsLTc5LjM4MTU3NjQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2JhMTI2ODQ0OTI4MDRkNTg5NDcwYmZkNGY5YTE2NDk2KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzY5MGJlZGVjYWI2ODRiZGVhNjdjOGNlN2Y4YzUzYmVhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2NmYjY5NzI1NGQxNTQ3ZTFhZjQ1NTcyYmQ1Y2YzYWVhID0gJCgnPGRpdiBpZD0iaHRtbF9jZmI2OTcyNTRkMTU0N2UxYWY0NTU3MmJkNWNmM2FlYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGVzaWduIEV4Y2hhbmdlLCBUb3JvbnRvIERvbWluaW9uIENlbnRyZSwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjkwYmVkZWNhYjY4NGJkZWE2N2M4Y2U3ZjhjNTNiZWEuc2V0Q29udGVudChodG1sX2NmYjY5NzI1NGQxNTQ3ZTFhZjQ1NTcyYmQ1Y2YzYWVhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzgyOTkzZmNhNDVlNDQzZTJiN2U4MzcxMDdmM2JlY2I5LmJpbmRQb3B1cChwb3B1cF82OTBiZWRlY2FiNjg0YmRlYTY3YzhjZTdmOGM1M2JlYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jYzAzNmM3NTZhMjQ0NDEzODQwNmZjZDVmNjdlZTkxZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODE5ODUsLTc5LjM3OTgxNjkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2JhMTI2ODQ0OTI4MDRkNTg5NDcwYmZkNGY5YTE2NDk2KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzhhMjBmM2Y1ZjlmMTQyZmRiNDM4ZmIzMzVjZTMzZmMyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzYzZDcxMTEyMjY0NTQ5OGQ4NDEyMjUxMzRmYjc0YmE4ID0gJCgnPGRpdiBpZD0iaHRtbF82M2Q3MTExMjI2NDU0OThkODQxMjI1MTM0ZmI3NGJhOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q29tbWVyY2UgQ291cnQsIFZpY3RvcmlhIEhvdGVsLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84YTIwZjNmNWY5ZjE0MmZkYjQzOGZiMzM1Y2UzM2ZjMi5zZXRDb250ZW50KGh0bWxfNjNkNzExMTIyNjQ1NDk4ZDg0MTIyNTEzNGZiNzRiYTgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2MwMzZjNzU2YTI0NDQxMzg0MDZmY2Q1ZjY3ZWU5MWUuYmluZFBvcHVwKHBvcHVwXzhhMjBmM2Y1ZjlmMTQyZmRiNDM4ZmIzMzVjZTMzZmMyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2JjY2JmYTUzMzg5NDRiMjE4MDBhNTE4NzFjMjYzYTVkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzExNjk0OCwtNzkuNDE2OTM1NTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzRjYzYyMTExMWI2NDExOWE5ZDhjOTk3ZjdhMGRjNzEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzgwZjg0OGRhOWM4NGU0Mjk0NGQ5NTAwNGEwY2RlMzYgPSAkKCc8ZGl2IGlkPSJodG1sX2M4MGY4NDhkYTljODRlNDI5NDRkOTUwMDRhMGNkZTM2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb3NlbGF3biwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zNGNjNjIxMTExYjY0MTE5YTlkOGM5OTdmN2EwZGM3MS5zZXRDb250ZW50KGh0bWxfYzgwZjg0OGRhOWM4NGU0Mjk0NGQ5NTAwNGEwY2RlMzYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYmNjYmZhNTMzODk0NGIyMTgwMGE1MTg3MWMyNjNhNWQuYmluZFBvcHVwKHBvcHVwXzM0Y2M2MjExMTFiNjQxMTlhOWQ4Yzk5N2Y3YTBkYzcxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Q0ZDlmMGM0Yzk4MTQ2MjFhMGVlNDZmZTQ1ODc4ZTRiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjk2OTQ3NiwtNzkuNDExMzA3MjAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjJkYWM3MWUwYWRlNGMzMGFjNTdhODc2M2E2ZDAxZmIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTA2YWNlN2U1NmFkNGFlOWI4Y2YwNDRiYzBjM2RiNzMgPSAkKCc8ZGl2IGlkPSJodG1sXzEwNmFjZTdlNTZhZDRhZTliOGNmMDQ0YmMwYzNkYjczIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Gb3Jlc3QgSGlsbCBOb3J0aCwgRm9yZXN0IEhpbGwgV2VzdCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iMmRhYzcxZTBhZGU0YzMwYWM1N2E4NzYzYTZkMDFmYi5zZXRDb250ZW50KGh0bWxfMTA2YWNlN2U1NmFkNGFlOWI4Y2YwNDRiYzBjM2RiNzMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDRkOWYwYzRjOTgxNDYyMWEwZWU0NmZlNDU4NzhlNGIuYmluZFBvcHVwKHBvcHVwX2IyZGFjNzFlMGFkZTRjMzBhYzU3YTg3NjNhNmQwMWZiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFmZTc4Zjk0NjlkYzQ3YTk5ODY2M2ZlMmNiMTU2MDBmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjcyNzA5NywtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjlkMzY2ODMzYmIyNGNkNzgyNzNlNTJkNjY0N2I2MzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODk2MWUxMGIyZTg2NGNjY2FhNjI3MDA4YmQ3YmZlNjUgPSAkKCc8ZGl2IGlkPSJodG1sXzg5NjFlMTBiMmU4NjRjY2NhYTYyNzAwOGJkN2JmZTY1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgQW5uZXgsIE5vcnRoIE1pZHRvd24sIFlvcmt2aWxsZSwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mOWQzNjY4MzNiYjI0Y2Q3ODI3M2U1MmQ2NjQ3YjYzMy5zZXRDb250ZW50KGh0bWxfODk2MWUxMGIyZTg2NGNjY2FhNjI3MDA4YmQ3YmZlNjUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMWZlNzhmOTQ2OWRjNDdhOTk4NjYzZmUyY2IxNTYwMGYuYmluZFBvcHVwKHBvcHVwX2Y5ZDM2NjgzM2JiMjRjZDc4MjczZTUyZDY2NDdiNjMzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I5NjQ2NjE0NTVkZjQ4ZWViYWNkZDQ0OTdkN2IzMjNjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNjk1NiwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iYTEyNjg0NDkyODA0ZDU4OTQ3MGJmZDRmOWExNjQ5Nik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jMjUwNzM2MmZmMTk0MDMyYTlhYTU0ODFjYTZmOWFmMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85ZmM1NDJlODFiMDA0ZDY0YWZkZTExMzAxOWJhMWQyMSA9ICQoJzxkaXYgaWQ9Imh0bWxfOWZjNTQyZTgxYjAwNGQ2NGFmZGUxMTMwMTliYTFkMjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhhcmJvcmQsIFVuaXZlcnNpdHkgb2YgVG9yb250bywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzI1MDczNjJmZjE5NDAzMmE5YWE1NDgxY2E2ZjlhZjEuc2V0Q29udGVudChodG1sXzlmYzU0MmU4MWIwMDRkNjRhZmRlMTEzMDE5YmExZDIxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I5NjQ2NjE0NTVkZjQ4ZWViYWNkZDQ0OTdkN2IzMjNjLmJpbmRQb3B1cChwb3B1cF9jMjUwNzM2MmZmMTk0MDMyYTlhYTU0ODFjYTZmOWFmMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kMDc3MzJmNDc1OTU0Y2IzYWE0YjcyYTg1NzVkMWJlOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MzIwNTcsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzM3NmJmZmFkMGE4NGJmNWJlODAyMjIzMmQ3OWY4ZWEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjdiMzAzNjY5MjcyNDhiMmJmYmM0YTRiODczMWJkMmQgPSAkKCc8ZGl2IGlkPSJodG1sXzY3YjMwMzY2OTI3MjQ4YjJiZmJjNGE0Yjg3MzFiZDJkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DaGluYXRvd24sIEdyYW5nZSBQYXJrLCBLZW5zaW5ndG9uIE1hcmtldCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzM3NmJmZmFkMGE4NGJmNWJlODAyMjIzMmQ3OWY4ZWEuc2V0Q29udGVudChodG1sXzY3YjMwMzY2OTI3MjQ4YjJiZmJjNGE0Yjg3MzFiZDJkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2QwNzczMmY0NzU5NTRjYjNhYTRiNzJhODU3NWQxYmU5LmJpbmRQb3B1cChwb3B1cF8zMzc2YmZmYWQwYTg0YmY1YmU4MDIyMjMyZDc5ZjhlYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82ZDFkYjEyZWIxMGM0NmVmOWYxZTdkYWMyMDEzYTBlNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjA4MTdmMGEzNmMzNDc2NmI2NzFhOGFiM2IxODMwM2UgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWYwNTI1NjczNjVhNGNlYmE5NWRiYThiZTRmMDBjZjUgPSAkKCc8ZGl2IGlkPSJodG1sXzVmMDUyNTY3MzY1YTRjZWJhOTVkYmE4YmU0ZjAwY2Y1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DTiBUb3dlciwgQmF0aHVyc3QgUXVheSwgSXNsYW5kIGFpcnBvcnQsIEhhcmJvdXJmcm9udCBXZXN0LCBLaW5nIGFuZCBTcGFkaW5hLCBSYWlsd2F5IExhbmRzLCBTb3V0aCBOaWFnYXJhLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82MDgxN2YwYTM2YzM0NzY2YjY3MWE4YWIzYjE4MzAzZS5zZXRDb250ZW50KGh0bWxfNWYwNTI1NjczNjVhNGNlYmE5NWRiYThiZTRmMDBjZjUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNmQxZGIxMmViMTBjNDZlZjlmMWU3ZGFjMjAxM2EwZTQuYmluZFBvcHVwKHBvcHVwXzYwODE3ZjBhMzZjMzQ3NjZiNjcxYThhYjNiMTgzMDNlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZiY2MyZWQyZWY1OTQwNjY4YjY0MzdhMWY0ZWRmNTMzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ2NDM1MiwtNzkuMzc0ODQ1OTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGUzZmE1Y2RjNGU5NDZkMWI1OTUxNzU0NDYwMzdhYzQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTBiMWNhNmY3YTdjNDkxYjhmMjdkOTI0ZGExYWYyYjcgPSAkKCc8ZGl2IGlkPSJodG1sXzUwYjFjYTZmN2E3YzQ5MWI4ZjI3ZDkyNGRhMWFmMmI3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdG4gQSBQTyBCb3hlcyAyNSBUaGUgRXNwbGFuYWRlLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80ZTNmYTVjZGM0ZTk0NmQxYjU5NTE3NTQ0NjAzN2FjNC5zZXRDb250ZW50KGh0bWxfNTBiMWNhNmY3YTdjNDkxYjhmMjdkOTI0ZGExYWYyYjcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNmJjYzJlZDJlZjU5NDA2NjhiNjQzN2ExZjRlZGY1MzMuYmluZFBvcHVwKHBvcHVwXzRlM2ZhNWNkYzRlOTQ2ZDFiNTk1MTc1NDQ2MDM3YWM0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFjMGRiZWFlMDA3MjRlZThhNTliY2ZhMzFjNTQwOTE5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4NDI5MiwtNzkuMzgyMjgwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iYTEyNjg0NDkyODA0ZDU4OTQ3MGJmZDRmOWExNjQ5Nik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83ZjI1NDcwODM2NzU0ZTc0YTI4Y2VjODA2M2RkMzMxZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hNGUwOTJkZTNkYjE0MTBhYjgxMGNlN2FhNjFiMTAzNCA9ICQoJzxkaXYgaWQ9Imh0bWxfYTRlMDkyZGUzZGIxNDEwYWI4MTBjZTdhYTYxYjEwMzQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZpcnN0IENhbmFkaWFuIFBsYWNlLCBVbmRlcmdyb3VuZCBjaXR5LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83ZjI1NDcwODM2NzU0ZTc0YTI4Y2VjODA2M2RkMzMxZS5zZXRDb250ZW50KGh0bWxfYTRlMDkyZGUzZGIxNDEwYWI4MTBjZTdhYTYxYjEwMzQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMWMwZGJlYWUwMDcyNGVlOGE1OWJjZmEzMWM1NDA5MTkuYmluZFBvcHVwKHBvcHVwXzdmMjU0NzA4MzY3NTRlNzRhMjhjZWM4MDYzZGQzMzFlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU1MGFjMjUzZmNmMTQ0YTBiNTVmYWM5Y2M5MWZkM2M3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5NTQyLC03OS40MjI1NjM3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2JhMTI2ODQ0OTI4MDRkNTg5NDcwYmZkNGY5YTE2NDk2KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU0ZjZkMzVlNjE2YjQ5MjI5ODgwYThjYjRhN2I4ZTk3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2YxOWUwMWI3ZDI0YzRmOWU5MzVhMjlkMTY0MWUwZDU3ID0gJCgnPGRpdiBpZD0iaHRtbF9mMTllMDFiN2QyNGM0ZjllOTM1YTI5ZDE2NDFlMGQ1NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2hyaXN0aWUsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzU0ZjZkMzVlNjE2YjQ5MjI5ODgwYThjYjRhN2I4ZTk3LnNldENvbnRlbnQoaHRtbF9mMTllMDFiN2QyNGM0ZjllOTM1YTI5ZDE2NDFlMGQ1Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81NTBhYzI1M2ZjZjE0NGEwYjU1ZmFjOWNjOTFmZDNjNy5iaW5kUG9wdXAocG9wdXBfNTRmNmQzNWU2MTZiNDkyMjk4ODBhOGNiNGE3YjhlOTcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWY1NWRiNjFlMDcwNDljN2E1ZDA0NDc1NWM5M2E3MTQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjkwMDUxMDAwMDAwMSwtNzkuNDQyMjU5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iYTEyNjg0NDkyODA0ZDU4OTQ3MGJmZDRmOWExNjQ5Nik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84OGUwZTdiZDZkYjY0M2VlOTU5NDk2MGQ2MjMxMTc2NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82YzM4NWYwZjc5OWI0MWQ3YWFlNjVkNTc4MTJlZmJkYyA9ICQoJzxkaXYgaWQ9Imh0bWxfNmMzODVmMGY3OTliNDFkN2FhZTY1ZDU3ODEyZWZiZGMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvdmVyY291cnQgVmlsbGFnZSwgRHVmZmVyaW4sIFdlc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODhlMGU3YmQ2ZGI2NDNlZTk1OTQ5NjBkNjIzMTE3Njcuc2V0Q29udGVudChodG1sXzZjMzg1ZjBmNzk5YjQxZDdhYWU2NWQ1NzgxMmVmYmRjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2VmNTVkYjYxZTA3MDQ5YzdhNWQwNDQ3NTVjOTNhNzE0LmJpbmRQb3B1cChwb3B1cF84OGUwZTdiZDZkYjY0M2VlOTU5NDk2MGQ2MjMxMTc2Nyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xZGZiMTllNDcwM2Q0NjI2ODAyOTA4YzVjNzUyYjJhMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzkyNjcwMDAwMDAwNiwtNzkuNDE5NzQ5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iYTEyNjg0NDkyODA0ZDU4OTQ3MGJmZDRmOWExNjQ5Nik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83ZTNhMjk3ZWFhNGM0OWIwYTdiZGE5MzRhNTU5M2NhMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84MTVmMGJlYTJlYzc0YTJjOTU0ODYzZTQwNmNmNmJmYyA9ICQoJzxkaXYgaWQ9Imh0bWxfODE1ZjBiZWEyZWM3NGEyYzk1NDg2M2U0MDZjZjZiZmMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxpdHRsZSBQb3J0dWdhbCwgVHJpbml0eSwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83ZTNhMjk3ZWFhNGM0OWIwYTdiZGE5MzRhNTU5M2NhMy5zZXRDb250ZW50KGh0bWxfODE1ZjBiZWEyZWM3NGEyYzk1NDg2M2U0MDZjZjZiZmMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMWRmYjE5ZTQ3MDNkNDYyNjgwMjkwOGM1Yzc1MmIyYTMuYmluZFBvcHVwKHBvcHVwXzdlM2EyOTdlYWE0YzQ5YjBhN2JkYTkzNGE1NTkzY2EzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE4NWFhOGZkOTQ0YTQ5NzBiODNjNjkyNGI2YTJkM2U3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjM2ODQ3MiwtNzkuNDI4MTkxNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOWRhNGE0ZjNjZjBmNDNhNTgyNGQ3ZmQxZmRjYTRlY2UgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjFjOTliMjU0Y2MyNDExNDk2Njk4OTgzMzJkNDMzODQgPSAkKCc8ZGl2IGlkPSJodG1sX2IxYzk5YjI1NGNjMjQxMTQ5NjY5ODk4MzMyZDQzMzg0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ccm9ja3RvbiwgRXhoaWJpdGlvbiBQbGFjZSwgUGFya2RhbGUgVmlsbGFnZSwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85ZGE0YTRmM2NmMGY0M2E1ODI0ZDdmZDFmZGNhNGVjZS5zZXRDb250ZW50KGh0bWxfYjFjOTliMjU0Y2MyNDExNDk2Njk4OTgzMzJkNDMzODQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTg1YWE4ZmQ5NDRhNDk3MGI4M2M2OTI0YjZhMmQzZTcuYmluZFBvcHVwKHBvcHVwXzlkYTRhNGYzY2YwZjQzYTU4MjRkN2ZkMWZkY2E0ZWNlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M4MjQxMTM0ZjI2YTQ5ODBhNDk5ODg1NDNjOGRiYThmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYxNjA4MywtNzkuNDY0NzYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDU3NTM5YmU3Nzc5NDJjMTliZTY1MjViNDM2ZjllY2MgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzBlMzgxM2M2YWFkNDQwMDg4YTUzZTY1Yjc1YzBkZTAgPSAkKCc8ZGl2IGlkPSJodG1sXzMwZTM4MTNjNmFhZDQ0MDA4OGE1M2U2NWI3NWMwZGUwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IaWdoIFBhcmssIFRoZSBKdW5jdGlvbiBTb3V0aCwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wNTc1MzliZTc3Nzk0MmMxOWJlNjUyNWI0MzZmOWVjYy5zZXRDb250ZW50KGh0bWxfMzBlMzgxM2M2YWFkNDQwMDg4YTUzZTY1Yjc1YzBkZTApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzgyNDExMzRmMjZhNDk4MGE0OTk4ODU0M2M4ZGJhOGYuYmluZFBvcHVwKHBvcHVwXzA1NzUzOWJlNzc3OTQyYzE5YmU2NTI1YjQzNmY5ZWNjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZjMjQ2OTA3MDRjZTRlM2Q4NTQxZDYwN2ExMGUwNTQ1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4OTU5NywtNzkuNDU2MzI1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2JhMTI2ODQ0OTI4MDRkNTg5NDcwYmZkNGY5YTE2NDk2KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2EwMDI4YTFkODM5NTQ4NmM5NzllM2I0ODNkZjBkNDA5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2YyZmEyMDRjYjMzYjQxZjY5N2MzZDg0MDBhMmYxYmYxID0gJCgnPGRpdiBpZD0iaHRtbF9mMmZhMjA0Y2IzM2I0MWY2OTdjM2Q4NDAwYTJmMWJmMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFya2RhbGUsIFJvbmNlc3ZhbGxlcywgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hMDAyOGExZDgzOTU0ODZjOTc5ZTNiNDgzZGYwZDQwOS5zZXRDb250ZW50KGh0bWxfZjJmYTIwNGNiMzNiNDFmNjk3YzNkODQwMGEyZjFiZjEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNmMyNDY5MDcwNGNlNGUzZDg1NDFkNjA3YTEwZTA1NDUuYmluZFBvcHVwKHBvcHVwX2EwMDI4YTFkODM5NTQ4NmM5NzllM2I0ODNkZjBkNDA5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Q3NmQ5Nzc1MGZhODQzNWI5OWZjNmI0ZWIyNGQ2ZGI5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNTcwNiwtNzkuNDg0NDQ5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iYTEyNjg0NDkyODA0ZDU4OTQ3MGJmZDRmOWExNjQ5Nik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zYTRhYjdkNWY0MDI0ZDZjODA5NWNhYTljNTdmYjRmMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80MmNkZTRmMGNkYTk0NWIyYWNkMmJkMGM5MDZjNjY2MiA9ICQoJzxkaXYgaWQ9Imh0bWxfNDJjZGU0ZjBjZGE5NDViMmFjZDJiZDBjOTA2YzY2NjIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJ1bm55bWVkZSwgU3dhbnNlYSwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zYTRhYjdkNWY0MDI0ZDZjODA5NWNhYTljNTdmYjRmMi5zZXRDb250ZW50KGh0bWxfNDJjZGU0ZjBjZGE5NDViMmFjZDJiZDBjOTA2YzY2NjIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDc2ZDk3NzUwZmE4NDM1Yjk5ZmM2YjRlYjI0ZDZkYjkuYmluZFBvcHVwKHBvcHVwXzNhNGFiN2Q1ZjQwMjRkNmM4MDk1Y2FhOWM1N2ZiNGYyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2QyMTc0NGM2ZmRiNDQ5ZWE5MWUwZTk2MmVlYWNhM2JmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyMzAxNSwtNzkuMzg5NDkzOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9iYTEyNjg0NDkyODA0ZDU4OTQ3MGJmZDRmOWExNjQ5Nik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jMGRhMDgyOTVjNjc0MDdkODU2MWY5MTc2NmM0NGVhYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kODg1NjMwNGQ3ZTU0OTY1YWI1ZTU1NzRiMDUxNWFiMyA9ICQoJzxkaXYgaWQ9Imh0bWxfZDg4NTYzMDRkN2U1NDk2NWFiNWU1NTc0YjA1MTVhYjMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlF1ZWVuJiMzOTtzIFBhcmssIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MwZGEwODI5NWM2NzQwN2Q4NTYxZjkxNzY2YzQ0ZWFhLnNldENvbnRlbnQoaHRtbF9kODg1NjMwNGQ3ZTU0OTY1YWI1ZTU1NzRiMDUxNWFiMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kMjE3NDRjNmZkYjQ0OWVhOTFlMGU5NjJlZWFjYTNiZi5iaW5kUG9wdXAocG9wdXBfYzBkYTA4Mjk1YzY3NDA3ZDg1NjFmOTE3NjZjNDRlYWEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjc5M2I0MGIxNGMzNGMyZGIwM2E4OGU3NjQyYjdkMWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI3NDM5LC03OS4zMjE1NThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYmExMjY4NDQ5MjgwNGQ1ODk0NzBiZmQ0ZjlhMTY0OTYpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTk1MjVmMTdhYzg0NGU3NDliMjczMGU3MGUzYjUyYmIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTljMjQxOGUzYmMzNGNlYjgzYWNiY2Y2MjE1YjBlMjEgPSAkKCc8ZGl2IGlkPSJodG1sX2E5YzI0MThlM2JjMzRjZWI4M2FjYmNmNjIxNWIwZTIxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CdXNpbmVzcyBSZXBseSBNYWlsIFByb2Nlc3NpbmcgQ2VudHJlIDk2OSBFYXN0ZXJuLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U5NTI1ZjE3YWM4NDRlNzQ5YjI3MzBlNzBlM2I1MmJiLnNldENvbnRlbnQoaHRtbF9hOWMyNDE4ZTNiYzM0Y2ViODNhY2JjZjYyMTViMGUyMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82NzkzYjQwYjE0YzM0YzJkYjAzYTg4ZTc2NDJiN2QxYi5iaW5kUG9wdXAocG9wdXBfZTk1MjVmMTdhYzg0NGU3NDliMjczMGU3MGUzYjUyYmIpOwoKICAgICAgICAgICAgCiAgICAgICAgCjwvc2NyaXB0Pg== onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### Use the Foursquare API to explore the neighborhoods


```python
# define Foursquare Credentials and Version
CLIENT_ID = 'HUEV0BQM13VF5INPHAJEWZHB5KMTK5TR02EOM2IOH34CEYHH' # your Foursquare ID
CLIENT_SECRET = 'SQXXLXAB5IVVYODNC1AGL1EXK1ZAELOYBNGRTLP4FZCNVIQN' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
```

    Your credentails:
    CLIENT_ID: HUEV0BQM13VF5INPHAJEWZHB5KMTK5TR02EOM2IOH34CEYHH
    CLIENT_SECRET:SQXXLXAB5IVVYODNC1AGL1EXK1ZAELOYBNGRTLP4FZCNVIQN


**Get the top 100 venues that are within a radius of 500 meters.**


```python
radius = 500
LIMIT = 100

venues = []

for lat, long, post, borough, neighborhood in zip(toronto_df_new['Latitude'], toronto_df_new['Longitude'], toronto_df_new['PostalCode'], toronto_df_new['Borough'], toronto_df_new['Neighborhood']):
    url = "https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}".format(
        CLIENT_ID,
        CLIENT_SECRET,
        VERSION,
        lat,
        long,
        radius, 
        LIMIT)
    
    results = requests.get(url).json()["response"]['groups'][0]['items']
    
    for venue in results:
        venues.append((
            post, 
            borough,
            neighborhood,
            lat, 
            long, 
            venue['venue']['name'], 
            venue['venue']['location']['lat'], 
            venue['venue']['location']['lng'],  
            venue['venue']['categories'][0]['name']))
```


```python
# convert the venues list into a new DataFrame
venues_df = pd.DataFrame(venues)

# define the column names
venues_df.columns = ['PostalCode', 'Borough', 'Neighborhood', 'BoroughLatitude', 'BoroughLongitude', 'VenueName', 'VenueLatitude', 'VenueLongitude', 'VenueCategory']

print(venues_df.shape)
venues_df.head()
```

    (1721, 9)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>BoroughLatitude</th>
      <th>BoroughLongitude</th>
      <th>VenueName</th>
      <th>VenueLatitude</th>
      <th>VenueLongitude</th>
      <th>VenueCategory</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Glen Manor Ravine</td>
      <td>43.676821</td>
      <td>-79.293942</td>
      <td>Trail</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Glen Stewart Ravine</td>
      <td>43.676300</td>
      <td>-79.294784</td>
      <td>Other Great Outdoors</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>The Big Carrot Natural Food Market</td>
      <td>43.678879</td>
      <td>-79.297734</td>
      <td>Health Food Store</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Grover Pub and Grub</td>
      <td>43.679181</td>
      <td>-79.297215</td>
      <td>Pub</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>Upper Beaches</td>
      <td>43.680563</td>
      <td>-79.292869</td>
      <td>Neighborhood</td>
    </tr>
  </tbody>
</table>
</div>



**Check how many venues were returned for each PostalCode**


```python
venues_df.groupby(["PostalCode", "Borough", "Neighborhood"]).count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>BoroughLatitude</th>
      <th>BoroughLongitude</th>
      <th>VenueName</th>
      <th>VenueLatitude</th>
      <th>VenueLongitude</th>
      <th>VenueCategory</th>
    </tr>
    <tr>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>M4E</th>
      <th>East Toronto</th>
      <th>The Beaches</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>M4K</th>
      <th>East Toronto</th>
      <th>The Danforth West, Riverdale</th>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
      <td>41</td>
    </tr>
    <tr>
      <th>M4L</th>
      <th>East Toronto</th>
      <th>The Beaches West, India Bazaar</th>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <th>M4M</th>
      <th>East Toronto</th>
      <th>Studio District</th>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
    </tr>
    <tr>
      <th>M4N</th>
      <th>Central Toronto</th>
      <th>Lawrence Park</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>M4P</th>
      <th>Central Toronto</th>
      <th>Davisville North</th>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>M4R</th>
      <th>Central Toronto</th>
      <th>North Toronto West</th>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
      <td>23</td>
    </tr>
    <tr>
      <th>M4S</th>
      <th>Central Toronto</th>
      <th>Davisville</th>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>M4T</th>
      <th>Central Toronto</th>
      <th>Moore Park, Summerhill East</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>M4V</th>
      <th>Central Toronto</th>
      <th>Deer Park, Forest Hill SE, Rathnelly, South Hill, Summerhill West</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>M4W</th>
      <th>Downtown Toronto</th>
      <th>Rosedale</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>M4X</th>
      <th>Downtown Toronto</th>
      <th>Cabbagetown, St. James Town</th>
      <td>49</td>
      <td>49</td>
      <td>49</td>
      <td>49</td>
      <td>49</td>
      <td>49</td>
    </tr>
    <tr>
      <th>M4Y</th>
      <th>Downtown Toronto</th>
      <th>Church and Wellesley</th>
      <td>83</td>
      <td>83</td>
      <td>83</td>
      <td>83</td>
      <td>83</td>
      <td>83</td>
    </tr>
    <tr>
      <th>M5A</th>
      <th>Downtown Toronto</th>
      <th>Harbourfront</th>
      <td>46</td>
      <td>46</td>
      <td>46</td>
      <td>46</td>
      <td>46</td>
      <td>46</td>
    </tr>
    <tr>
      <th>M5B</th>
      <th>Downtown Toronto</th>
      <th>Ryerson, Garden District</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>M5C</th>
      <th>Downtown Toronto</th>
      <th>St. James Town</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>M5E</th>
      <th>Downtown Toronto</th>
      <th>Berczy Park</th>
      <td>56</td>
      <td>56</td>
      <td>56</td>
      <td>56</td>
      <td>56</td>
      <td>56</td>
    </tr>
    <tr>
      <th>M5G</th>
      <th>Downtown Toronto</th>
      <th>Central Bay Street</th>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
    </tr>
    <tr>
      <th>M5H</th>
      <th>Downtown Toronto</th>
      <th>Adelaide, King, Richmond</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>M5J</th>
      <th>Downtown Toronto</th>
      <th>Harbourfront East, Toronto Islands, Union Station</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>M5K</th>
      <th>Downtown Toronto</th>
      <th>Design Exchange, Toronto Dominion Centre</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>M5L</th>
      <th>Downtown Toronto</th>
      <th>Commerce Court, Victoria Hotel</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>M5N</th>
      <th>Central Toronto</th>
      <th>Roselawn</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>M5P</th>
      <th>Central Toronto</th>
      <th>Forest Hill North, Forest Hill West</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>M5R</th>
      <th>Central Toronto</th>
      <th>The Annex, North Midtown, Yorkville</th>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
    </tr>
    <tr>
      <th>M5S</th>
      <th>Downtown Toronto</th>
      <th>Harbord, University of Toronto</th>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
      <td>36</td>
    </tr>
    <tr>
      <th>M5T</th>
      <th>Downtown Toronto</th>
      <th>Chinatown, Grange Park, Kensington Market</th>
      <td>84</td>
      <td>84</td>
      <td>84</td>
      <td>84</td>
      <td>84</td>
      <td>84</td>
    </tr>
    <tr>
      <th>M5V</th>
      <th>Downtown Toronto</th>
      <th>CN Tower, Bathurst Quay, Island airport, Harbourfront West, King and Spadina, Railway Lands, South Niagara</th>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
      <td>17</td>
    </tr>
    <tr>
      <th>M5W</th>
      <th>Downtown Toronto</th>
      <th>Stn A PO Boxes 25 The Esplanade</th>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
    </tr>
    <tr>
      <th>M5X</th>
      <th>Downtown Toronto</th>
      <th>First Canadian Place, Underground city</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>M6G</th>
      <th>Downtown Toronto</th>
      <th>Christie</th>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
    </tr>
    <tr>
      <th>M6H</th>
      <th>West Toronto</th>
      <th>Dovercourt Village, Dufferin</th>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
    </tr>
    <tr>
      <th>M6J</th>
      <th>West Toronto</th>
      <th>Little Portugal, Trinity</th>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
      <td>53</td>
    </tr>
    <tr>
      <th>M6K</th>
      <th>West Toronto</th>
      <th>Brockton, Exhibition Place, Parkdale Village</th>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
    </tr>
    <tr>
      <th>M6P</th>
      <th>West Toronto</th>
      <th>High Park, The Junction South</th>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
    </tr>
    <tr>
      <th>M6R</th>
      <th>West Toronto</th>
      <th>Parkdale, Roncesvalles</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>M6S</th>
      <th>West Toronto</th>
      <th>Runnymede, Swansea</th>
      <td>39</td>
      <td>39</td>
      <td>39</td>
      <td>39</td>
      <td>39</td>
      <td>39</td>
    </tr>
    <tr>
      <th>M7A</th>
      <th>Downtown Toronto</th>
      <th>Queen's Park</th>
      <td>38</td>
      <td>38</td>
      <td>38</td>
      <td>38</td>
      <td>38</td>
      <td>38</td>
    </tr>
    <tr>
      <th>M7Y</th>
      <th>East Toronto</th>
      <th>Business Reply Mail Processing Centre 969 Eastern</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('There are {} uniques categories.'.format(len(venues_df['VenueCategory'].unique())))
```

    There are 241 uniques categories.


###  Analyze Each Area


```python
# one hot encoding
toronto_onehot = pd.get_dummies(venues_df[['VenueCategory']], prefix="", prefix_sep="")

# add postal, borough and neighborhood column back to dataframe
toronto_onehot['PostalCode'] = venues_df['PostalCode'] 
toronto_onehot['Borough'] = venues_df['Borough'] 
toronto_onehot['Neighborhoods'] = venues_df['Neighborhood'] 

# move postal, borough and neighborhood column to the first column
fixed_columns = list(toronto_onehot.columns[-3:]) + list(toronto_onehot.columns[:-3])
toronto_onehot = toronto_onehot[fixed_columns]

print(toronto_onehot.shape)
toronto_onehot.head()
```

    (1721, 244)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhoods</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>Aquarium</th>
      <th>Art Gallery</th>
      <th>Arts &amp; Crafts Store</th>
      <th>Asian Restaurant</th>
      <th>Athletics &amp; Sports</th>
      <th>Auto Workshop</th>
      <th>BBQ Joint</th>
      <th>Baby Store</th>
      <th>Bagel Shop</th>
      <th>Bakery</th>
      <th>Bank</th>
      <th>Bar</th>
      <th>Baseball Stadium</th>
      <th>Basketball Stadium</th>
      <th>Beach</th>
      <th>Bed &amp; Breakfast</th>
      <th>Beer Bar</th>
      <th>Beer Store</th>
      <th>Belgian Restaurant</th>
      <th>Bistro</th>
      <th>Boat or Ferry</th>
      <th>Bookstore</th>
      <th>Boutique</th>
      <th>Brazilian Restaurant</th>
      <th>Breakfast Spot</th>
      <th>Brewery</th>
      <th>Bubble Tea Shop</th>
      <th>Building</th>
      <th>Burger Joint</th>
      <th>Burrito Place</th>
      <th>Bus Line</th>
      <th>Bus Stop</th>
      <th>Butcher</th>
      <th>Café</th>
      <th>Cajun / Creole Restaurant</th>
      <th>Camera Store</th>
      <th>Candy Store</th>
      <th>Caribbean Restaurant</th>
      <th>Cheese Shop</th>
      <th>Chinese Restaurant</th>
      <th>Chocolate Shop</th>
      <th>Church</th>
      <th>Climbing Gym</th>
      <th>Clothing Store</th>
      <th>Cocktail Bar</th>
      <th>Coffee Shop</th>
      <th>College Arts Building</th>
      <th>College Auditorium</th>
      <th>College Gym</th>
      <th>College Rec Center</th>
      <th>Colombian Restaurant</th>
      <th>Comfort Food Restaurant</th>
      <th>Comic Shop</th>
      <th>Concert Hall</th>
      <th>Construction &amp; Landscaping</th>
      <th>Convenience Store</th>
      <th>Cosmetics Shop</th>
      <th>Coworking Space</th>
      <th>Creperie</th>
      <th>Cuban Restaurant</th>
      <th>Cupcake Shop</th>
      <th>Dance Studio</th>
      <th>Deli / Bodega</th>
      <th>Department Store</th>
      <th>Dessert Shop</th>
      <th>Dim Sum Restaurant</th>
      <th>Diner</th>
      <th>Discount Store</th>
      <th>Dog Run</th>
      <th>Doner Restaurant</th>
      <th>Donut Shop</th>
      <th>Dumpling Restaurant</th>
      <th>Eastern European Restaurant</th>
      <th>Electronics Store</th>
      <th>Ethiopian Restaurant</th>
      <th>Event Space</th>
      <th>Falafel Restaurant</th>
      <th>Farmers Market</th>
      <th>Fast Food Restaurant</th>
      <th>Filipino Restaurant</th>
      <th>Fish &amp; Chips Shop</th>
      <th>Fish Market</th>
      <th>Flea Market</th>
      <th>Flower Shop</th>
      <th>Food</th>
      <th>Food &amp; Drink Shop</th>
      <th>Food Court</th>
      <th>Food Truck</th>
      <th>Fountain</th>
      <th>French Restaurant</th>
      <th>Fried Chicken Joint</th>
      <th>Frozen Yogurt Shop</th>
      <th>Fruit &amp; Vegetable Store</th>
      <th>Furniture / Home Store</th>
      <th>Gaming Cafe</th>
      <th>Garden</th>
      <th>Garden Center</th>
      <th>Gas Station</th>
      <th>Gastropub</th>
      <th>Gay Bar</th>
      <th>General Entertainment</th>
      <th>General Travel</th>
      <th>Gift Shop</th>
      <th>Gluten-free Restaurant</th>
      <th>Gourmet Shop</th>
      <th>Greek Restaurant</th>
      <th>Grocery Store</th>
      <th>Gym</th>
      <th>Gym / Fitness Center</th>
      <th>Harbor / Marina</th>
      <th>Health &amp; Beauty Service</th>
      <th>Health Food Store</th>
      <th>Historic Site</th>
      <th>History Museum</th>
      <th>Hospital</th>
      <th>Hostel</th>
      <th>Hotel</th>
      <th>Hotel Bar</th>
      <th>Hotpot Restaurant</th>
      <th>IT Services</th>
      <th>Ice Cream Shop</th>
      <th>Indian Restaurant</th>
      <th>Indie Movie Theater</th>
      <th>Indoor Play Area</th>
      <th>Intersection</th>
      <th>Irish Pub</th>
      <th>Italian Restaurant</th>
      <th>Japanese Restaurant</th>
      <th>Jazz Club</th>
      <th>Jewelry Store</th>
      <th>Juice Bar</th>
      <th>Korean Restaurant</th>
      <th>Lake</th>
      <th>Latin American Restaurant</th>
      <th>Light Rail Station</th>
      <th>Lingerie Store</th>
      <th>Liquor Store</th>
      <th>Lounge</th>
      <th>Mac &amp; Cheese Joint</th>
      <th>Market</th>
      <th>Massage Studio</th>
      <th>Mediterranean Restaurant</th>
      <th>Men's Store</th>
      <th>Metro Station</th>
      <th>Mexican Restaurant</th>
      <th>Middle Eastern Restaurant</th>
      <th>Miscellaneous Shop</th>
      <th>Modern European Restaurant</th>
      <th>Molecular Gastronomy Restaurant</th>
      <th>Monument / Landmark</th>
      <th>Movie Theater</th>
      <th>Moving Target</th>
      <th>Museum</th>
      <th>Music Venue</th>
      <th>Neighborhood</th>
      <th>New American Restaurant</th>
      <th>Nightclub</th>
      <th>Noodle House</th>
      <th>Office</th>
      <th>Opera House</th>
      <th>Optical Shop</th>
      <th>Organic Grocery</th>
      <th>Other Great Outdoors</th>
      <th>Outdoor Sculpture</th>
      <th>Park</th>
      <th>Performing Arts Venue</th>
      <th>Pet Store</th>
      <th>Pharmacy</th>
      <th>Photography Studio</th>
      <th>Pizza Place</th>
      <th>Plane</th>
      <th>Playground</th>
      <th>Plaza</th>
      <th>Poke Place</th>
      <th>Pool</th>
      <th>Portuguese Restaurant</th>
      <th>Poutine Place</th>
      <th>Pub</th>
      <th>Ramen Restaurant</th>
      <th>Record Shop</th>
      <th>Rental Car Location</th>
      <th>Restaurant</th>
      <th>Roof Deck</th>
      <th>Sake Bar</th>
      <th>Salad Place</th>
      <th>Salon / Barbershop</th>
      <th>Sandwich Place</th>
      <th>Scenic Lookout</th>
      <th>School</th>
      <th>Sculpture Garden</th>
      <th>Seafood Restaurant</th>
      <th>Shoe Store</th>
      <th>Shopping Mall</th>
      <th>Skate Park</th>
      <th>Skating Rink</th>
      <th>Smoke Shop</th>
      <th>Snack Place</th>
      <th>Soup Place</th>
      <th>Southern / Soul Food Restaurant</th>
      <th>Spa</th>
      <th>Speakeasy</th>
      <th>Sporting Goods Shop</th>
      <th>Sports Bar</th>
      <th>Stadium</th>
      <th>Stationery Store</th>
      <th>Steakhouse</th>
      <th>Strip Club</th>
      <th>Supermarket</th>
      <th>Supplement Shop</th>
      <th>Sushi Restaurant</th>
      <th>Swim School</th>
      <th>Taco Place</th>
      <th>Tailor Shop</th>
      <th>Taiwanese Restaurant</th>
      <th>Tanning Salon</th>
      <th>Tea Room</th>
      <th>Tennis Court</th>
      <th>Thai Restaurant</th>
      <th>Theater</th>
      <th>Theme Restaurant</th>
      <th>Thrift / Vintage Store</th>
      <th>Toy / Game Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
      <th>Yoga Studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



**Group rows by neighborhood and by taking the mean of the frequency of occurrence of each category**


```python
toronto_grouped = toronto_onehot.groupby(["PostalCode", "Borough", "Neighborhoods"]).mean().reset_index()

print(toronto_grouped.shape)
toronto_grouped
```

    (39, 244)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhoods</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>Aquarium</th>
      <th>Art Gallery</th>
      <th>Arts &amp; Crafts Store</th>
      <th>Asian Restaurant</th>
      <th>Athletics &amp; Sports</th>
      <th>Auto Workshop</th>
      <th>BBQ Joint</th>
      <th>Baby Store</th>
      <th>Bagel Shop</th>
      <th>Bakery</th>
      <th>Bank</th>
      <th>Bar</th>
      <th>Baseball Stadium</th>
      <th>Basketball Stadium</th>
      <th>Beach</th>
      <th>Bed &amp; Breakfast</th>
      <th>Beer Bar</th>
      <th>Beer Store</th>
      <th>Belgian Restaurant</th>
      <th>Bistro</th>
      <th>Boat or Ferry</th>
      <th>Bookstore</th>
      <th>Boutique</th>
      <th>Brazilian Restaurant</th>
      <th>Breakfast Spot</th>
      <th>Brewery</th>
      <th>Bubble Tea Shop</th>
      <th>Building</th>
      <th>Burger Joint</th>
      <th>Burrito Place</th>
      <th>Bus Line</th>
      <th>Bus Stop</th>
      <th>Butcher</th>
      <th>Café</th>
      <th>Cajun / Creole Restaurant</th>
      <th>Camera Store</th>
      <th>Candy Store</th>
      <th>Caribbean Restaurant</th>
      <th>Cheese Shop</th>
      <th>Chinese Restaurant</th>
      <th>Chocolate Shop</th>
      <th>Church</th>
      <th>Climbing Gym</th>
      <th>Clothing Store</th>
      <th>Cocktail Bar</th>
      <th>Coffee Shop</th>
      <th>College Arts Building</th>
      <th>College Auditorium</th>
      <th>College Gym</th>
      <th>College Rec Center</th>
      <th>Colombian Restaurant</th>
      <th>Comfort Food Restaurant</th>
      <th>Comic Shop</th>
      <th>Concert Hall</th>
      <th>Construction &amp; Landscaping</th>
      <th>Convenience Store</th>
      <th>Cosmetics Shop</th>
      <th>Coworking Space</th>
      <th>Creperie</th>
      <th>Cuban Restaurant</th>
      <th>Cupcake Shop</th>
      <th>Dance Studio</th>
      <th>Deli / Bodega</th>
      <th>Department Store</th>
      <th>Dessert Shop</th>
      <th>Dim Sum Restaurant</th>
      <th>Diner</th>
      <th>Discount Store</th>
      <th>Dog Run</th>
      <th>Doner Restaurant</th>
      <th>Donut Shop</th>
      <th>Dumpling Restaurant</th>
      <th>Eastern European Restaurant</th>
      <th>Electronics Store</th>
      <th>Ethiopian Restaurant</th>
      <th>Event Space</th>
      <th>Falafel Restaurant</th>
      <th>Farmers Market</th>
      <th>Fast Food Restaurant</th>
      <th>Filipino Restaurant</th>
      <th>Fish &amp; Chips Shop</th>
      <th>Fish Market</th>
      <th>Flea Market</th>
      <th>Flower Shop</th>
      <th>Food</th>
      <th>Food &amp; Drink Shop</th>
      <th>Food Court</th>
      <th>Food Truck</th>
      <th>Fountain</th>
      <th>French Restaurant</th>
      <th>Fried Chicken Joint</th>
      <th>Frozen Yogurt Shop</th>
      <th>Fruit &amp; Vegetable Store</th>
      <th>Furniture / Home Store</th>
      <th>Gaming Cafe</th>
      <th>Garden</th>
      <th>Garden Center</th>
      <th>Gas Station</th>
      <th>Gastropub</th>
      <th>Gay Bar</th>
      <th>General Entertainment</th>
      <th>General Travel</th>
      <th>Gift Shop</th>
      <th>Gluten-free Restaurant</th>
      <th>Gourmet Shop</th>
      <th>Greek Restaurant</th>
      <th>Grocery Store</th>
      <th>Gym</th>
      <th>Gym / Fitness Center</th>
      <th>Harbor / Marina</th>
      <th>Health &amp; Beauty Service</th>
      <th>Health Food Store</th>
      <th>Historic Site</th>
      <th>History Museum</th>
      <th>Hospital</th>
      <th>Hostel</th>
      <th>Hotel</th>
      <th>Hotel Bar</th>
      <th>Hotpot Restaurant</th>
      <th>IT Services</th>
      <th>Ice Cream Shop</th>
      <th>Indian Restaurant</th>
      <th>Indie Movie Theater</th>
      <th>Indoor Play Area</th>
      <th>Intersection</th>
      <th>Irish Pub</th>
      <th>Italian Restaurant</th>
      <th>Japanese Restaurant</th>
      <th>Jazz Club</th>
      <th>Jewelry Store</th>
      <th>Juice Bar</th>
      <th>Korean Restaurant</th>
      <th>Lake</th>
      <th>Latin American Restaurant</th>
      <th>Light Rail Station</th>
      <th>Lingerie Store</th>
      <th>Liquor Store</th>
      <th>Lounge</th>
      <th>Mac &amp; Cheese Joint</th>
      <th>Market</th>
      <th>Massage Studio</th>
      <th>Mediterranean Restaurant</th>
      <th>Men's Store</th>
      <th>Metro Station</th>
      <th>Mexican Restaurant</th>
      <th>Middle Eastern Restaurant</th>
      <th>Miscellaneous Shop</th>
      <th>Modern European Restaurant</th>
      <th>Molecular Gastronomy Restaurant</th>
      <th>Monument / Landmark</th>
      <th>Movie Theater</th>
      <th>Moving Target</th>
      <th>Museum</th>
      <th>Music Venue</th>
      <th>Neighborhood</th>
      <th>New American Restaurant</th>
      <th>Nightclub</th>
      <th>Noodle House</th>
      <th>Office</th>
      <th>Opera House</th>
      <th>Optical Shop</th>
      <th>Organic Grocery</th>
      <th>Other Great Outdoors</th>
      <th>Outdoor Sculpture</th>
      <th>Park</th>
      <th>Performing Arts Venue</th>
      <th>Pet Store</th>
      <th>Pharmacy</th>
      <th>Photography Studio</th>
      <th>Pizza Place</th>
      <th>Plane</th>
      <th>Playground</th>
      <th>Plaza</th>
      <th>Poke Place</th>
      <th>Pool</th>
      <th>Portuguese Restaurant</th>
      <th>Poutine Place</th>
      <th>Pub</th>
      <th>Ramen Restaurant</th>
      <th>Record Shop</th>
      <th>Rental Car Location</th>
      <th>Restaurant</th>
      <th>Roof Deck</th>
      <th>Sake Bar</th>
      <th>Salad Place</th>
      <th>Salon / Barbershop</th>
      <th>Sandwich Place</th>
      <th>Scenic Lookout</th>
      <th>School</th>
      <th>Sculpture Garden</th>
      <th>Seafood Restaurant</th>
      <th>Shoe Store</th>
      <th>Shopping Mall</th>
      <th>Skate Park</th>
      <th>Skating Rink</th>
      <th>Smoke Shop</th>
      <th>Snack Place</th>
      <th>Soup Place</th>
      <th>Southern / Soul Food Restaurant</th>
      <th>Spa</th>
      <th>Speakeasy</th>
      <th>Sporting Goods Shop</th>
      <th>Sports Bar</th>
      <th>Stadium</th>
      <th>Stationery Store</th>
      <th>Steakhouse</th>
      <th>Strip Club</th>
      <th>Supermarket</th>
      <th>Supplement Shop</th>
      <th>Sushi Restaurant</th>
      <th>Swim School</th>
      <th>Taco Place</th>
      <th>Tailor Shop</th>
      <th>Taiwanese Restaurant</th>
      <th>Tanning Salon</th>
      <th>Tea Room</th>
      <th>Tennis Court</th>
      <th>Thai Restaurant</th>
      <th>Theater</th>
      <th>Theme Restaurant</th>
      <th>Thrift / Vintage Store</th>
      <th>Toy / Game Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
      <th>Yoga Studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.20000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.20</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.20000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.048780</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.024390</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.073171</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.02439</td>
      <td>0.02439</td>
      <td>0.048780</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.219512</td>
      <td>0.024390</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.048780</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.073171</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.024390</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.024390</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.02439</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.024390</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>The Beaches West, India Bazaar</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.050000</td>
      <td>0.050000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.050000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.100000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.023810</td>
      <td>0.023810</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.095238</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.02381</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.02381</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.023810</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.02381</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.02381</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.023810</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.2</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.2</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.2</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.2</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M4P</td>
      <td>Central Toronto</td>
      <td>Davisville North</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M4R</td>
      <td>Central Toronto</td>
      <td>North Toronto West</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.130435</td>
      <td>0.000000</td>
      <td>0.086957</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.043478</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.043478</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M4S</td>
      <td>Central Toronto</td>
      <td>Davisville</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.027778</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.0</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.027778</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M4T</td>
      <td>Central Toronto</td>
      <td>Moore Park, Summerhill East</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.25</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M4V</td>
      <td>Central Toronto</td>
      <td>Deer Park, Forest Hill SE, Rathnelly, South Hi...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.00</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M4W</td>
      <td>Downtown Toronto</td>
      <td>Rosedale</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.25000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M4X</td>
      <td>Downtown Toronto</td>
      <td>Cabbagetown, St. James Town</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040816</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.061224</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.061224</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.00</td>
      <td>0.020408</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040816</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.020408</td>
      <td>0.061224</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.020408</td>
      <td>0.0</td>
      <td>0.040816</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040816</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.061224</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M4Y</td>
      <td>Downtown Toronto</td>
      <td>Church and Wellesley</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.00</td>
      <td>0.012048</td>
      <td>0.012048</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024096</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.072289</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024096</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024096</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.024096</td>
      <td>0.048193</td>
      <td>0.012048</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.024096</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024096</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.012048</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.060241</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024096</td>
      <td>0.024096</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024096</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.036145</td>
      <td>0.00</td>
      <td>0.012048</td>
      <td>0.00000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.012048</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.048193</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.012048</td>
      <td>0.012048</td>
      <td>0.012048</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012048</td>
      <td>0.00</td>
      <td>0.012048</td>
    </tr>
    <tr>
      <th>13</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.021739</td>
      <td>0.00</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.065217</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.152174</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.021739</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.021739</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.065217</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.065217</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.043478</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>M5B</td>
      <td>Downtown Toronto</td>
      <td>Ryerson, Garden District</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.070000</td>
      <td>0.000000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.02</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.020000</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.010000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>M5C</td>
      <td>Downtown Toronto</td>
      <td>St. James Town</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.060000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.020000</td>
      <td>0.070000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.00000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.02</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>M5E</td>
      <td>Downtown Toronto</td>
      <td>Berczy Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.017857</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.017857</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.017857</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.089286</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.017857</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.017857</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.017857</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.017857</td>
      <td>0.017857</td>
      <td>0.017857</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.017857</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.017857</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.017857</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.017857</td>
      <td>0.00</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.017857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>M5G</td>
      <td>Downtown Toronto</td>
      <td>Central Bay Street</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.00</td>
      <td>0.036585</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.170732</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.012195</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.024390</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.036585</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.048780</td>
      <td>0.036585</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.036585</td>
      <td>0.012195</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.02439</td>
      <td>0.000000</td>
      <td>0.036585</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.012195</td>
      <td>0.00</td>
      <td>0.024390</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.012195</td>
    </tr>
    <tr>
      <th>18</th>
      <td>M5H</td>
      <td>Downtown Toronto</td>
      <td>Adelaide, King, Richmond</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.030000</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.060000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.01000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.040000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>M5J</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront East, Toronto Islands, Union Station</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.05</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.02</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.010000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.120000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.01000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.02</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.01000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.03</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>M5K</td>
      <td>Downtown Toronto</td>
      <td>Design Exchange, Toronto Dominion Centre</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.110000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.060000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.02000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>M5L</td>
      <td>Downtown Toronto</td>
      <td>Commerce Court, Victoria Hotel</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.070000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.120000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.030000</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.01000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>M5N</td>
      <td>Central Toronto</td>
      <td>Roselawn</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.5000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>M5P</td>
      <td>Central Toronto</td>
      <td>Forest Hill North, Forest Hill West</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.25000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>M5R</td>
      <td>Central Toronto</td>
      <td>The Annex, North Midtown, Yorkville</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.136364</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.0</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.136364</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>M5S</td>
      <td>Downtown Toronto</td>
      <td>Harbord, University of Toronto</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.138889</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.027778</td>
    </tr>
    <tr>
      <th>26</th>
      <td>M5T</td>
      <td>Downtown Toronto</td>
      <td>Chinatown, Grange Park, Kensington Market</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.011905</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011905</td>
      <td>0.011905</td>
      <td>0.000000</td>
      <td>0.011905</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.011905</td>
      <td>0.011905</td>
      <td>0.011905</td>
      <td>0.00</td>
      <td>0.023810</td>
      <td>0.011905</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.059524</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.011905</td>
      <td>0.011905</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023810</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023810</td>
      <td>0.011905</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011905</td>
      <td>0.011905</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.011905</td>
      <td>0.000000</td>
      <td>0.011905</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011905</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.011905</td>
      <td>0.011905</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.011905</td>
      <td>0.000000</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011905</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011905</td>
      <td>0.011905</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011905</td>
      <td>0.011905</td>
      <td>0.000000</td>
      <td>0.011905</td>
      <td>0.011905</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011905</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.011905</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.023810</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011905</td>
      <td>0.0</td>
      <td>0.011905</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011905</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011905</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.011905</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.059524</td>
      <td>0.011905</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>M5V</td>
      <td>Downtown Toronto</td>
      <td>CN Tower, Bathurst Quay, Island airport, Harbo...</td>
      <td>0.000000</td>
      <td>0.058824</td>
      <td>0.058824</td>
      <td>0.058824</td>
      <td>0.117647</td>
      <td>0.176471</td>
      <td>0.117647</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.058824</td>
      <td>0.000000</td>
      <td>0.058824</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.058824</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.058824</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.058824</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.058824</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.058824</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>M5W</td>
      <td>Downtown Toronto</td>
      <td>Stn A PO Boxes 25 The Esplanade</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.010417</td>
      <td>0.00</td>
      <td>0.020833</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.020833</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020833</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020833</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.00000</td>
      <td>0.020833</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020833</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.010417</td>
      <td>0.020833</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020833</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.031250</td>
      <td>0.031250</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010417</td>
      <td>0.020833</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.00</td>
      <td>0.010417</td>
      <td>0.0000</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.020833</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.031250</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010417</td>
      <td>0.00</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>M5X</td>
      <td>Downtown Toronto</td>
      <td>First Canadian Place, Underground city</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.070000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.110000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.01000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>M6G</td>
      <td>Downtown Toronto</td>
      <td>Christie</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.222222</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.111111</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>M6H</td>
      <td>West Toronto</td>
      <td>Dovercourt Village, Dufferin</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.105263</td>
      <td>0.052632</td>
      <td>0.052632</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.105263</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>M6J</td>
      <td>West Toronto</td>
      <td>Little Portugal, Trinity</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.056604</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.113208</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037736</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.056604</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.018868</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.018868</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.037736</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.00000</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.037736</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.056604</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.037736</td>
      <td>0.018868</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.018868</td>
    </tr>
    <tr>
      <th>33</th>
      <td>M6K</td>
      <td>West Toronto</td>
      <td>Brockton, Exhibition Place, Parkdale Village</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.136364</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.090909</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>M6P</td>
      <td>West Toronto</td>
      <td>High Park, The Junction South</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.041667</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>M6R</td>
      <td>West Toronto</td>
      <td>Parkdale, Roncesvalles</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.142857</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.071429</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>M6S</td>
      <td>West Toronto</td>
      <td>Runnymede, Swansea</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.051282</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.076923</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.00</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025641</td>
      <td>0.051282</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.025641</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.025641</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.025641</td>
    </tr>
    <tr>
      <th>37</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.026316</td>
      <td>0.026316</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.263158</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.052632</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.00</td>
      <td>0.026316</td>
    </tr>
    <tr>
      <th>38</th>
      <td>M7Y</td>
      <td>East Toronto</td>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0625</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0625</td>
      <td>0.0625</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.0625</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0625</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.062500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Create the new dataframe and display the top 10 venues for each PostalCode.**


```python
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
areaColumns = ['PostalCode', 'Borough', 'Neighborhoods']
freqColumns = []
for ind in np.arange(num_top_venues):
    try:
        freqColumns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        freqColumns.append('{}th Most Common Venue'.format(ind+1))
columns = areaColumns+freqColumns

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['PostalCode'] = toronto_grouped['PostalCode']
neighborhoods_venues_sorted['Borough'] = toronto_grouped['Borough']
neighborhoods_venues_sorted['Neighborhoods'] = toronto_grouped['Neighborhoods']

for ind in np.arange(toronto_grouped.shape[0]):
    row_categories = toronto_grouped.iloc[ind, :].iloc[3:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    neighborhoods_venues_sorted.iloc[ind, 3:] = row_categories_sorted.index.values[0:num_top_venues]

# neighborhoods_venues_sorted.sort_values(freqColumns, inplace=True)
print(neighborhoods_venues_sorted.shape)
neighborhoods_venues_sorted
```

    (39, 13)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhoods</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>Health Food Store</td>
      <td>Other Great Outdoors</td>
      <td>Trail</td>
      <td>Neighborhood</td>
      <td>Pub</td>
      <td>Eastern European Restaurant</td>
      <td>Dumpling Restaurant</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>Greek Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Bookstore</td>
      <td>Furniture / Home Store</td>
      <td>Ice Cream Shop</td>
      <td>Fruit &amp; Vegetable Store</td>
      <td>Brewery</td>
      <td>Bubble Tea Shop</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>The Beaches West, India Bazaar</td>
      <td>Sandwich Place</td>
      <td>Pizza Place</td>
      <td>Park</td>
      <td>Ice Cream Shop</td>
      <td>Burger Joint</td>
      <td>Burrito Place</td>
      <td>Liquor Store</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Steakhouse</td>
      <td>Food &amp; Drink Shop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Brewery</td>
      <td>Gastropub</td>
      <td>American Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Middle Eastern Restaurant</td>
      <td>Bar</td>
      <td>Clothing Store</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
      <td>Photography Studio</td>
      <td>Construction &amp; Landscaping</td>
      <td>Park</td>
      <td>Bus Line</td>
      <td>Swim School</td>
      <td>Diner</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M4P</td>
      <td>Central Toronto</td>
      <td>Davisville North</td>
      <td>Breakfast Spot</td>
      <td>Park</td>
      <td>Department Store</td>
      <td>Food &amp; Drink Shop</td>
      <td>Hotel</td>
      <td>Sandwich Place</td>
      <td>Gym</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M4R</td>
      <td>Central Toronto</td>
      <td>North Toronto West</td>
      <td>Clothing Store</td>
      <td>Coffee Shop</td>
      <td>Yoga Studio</td>
      <td>Bagel Shop</td>
      <td>Pet Store</td>
      <td>Chinese Restaurant</td>
      <td>Dessert Shop</td>
      <td>Restaurant</td>
      <td>Diner</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M4S</td>
      <td>Central Toronto</td>
      <td>Davisville</td>
      <td>Sandwich Place</td>
      <td>Pizza Place</td>
      <td>Dessert Shop</td>
      <td>Gym</td>
      <td>Café</td>
      <td>Italian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Sushi Restaurant</td>
      <td>Pharmacy</td>
      <td>Deli / Bodega</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M4T</td>
      <td>Central Toronto</td>
      <td>Moore Park, Summerhill East</td>
      <td>Tennis Court</td>
      <td>Playground</td>
      <td>Park</td>
      <td>Restaurant</td>
      <td>Doner Restaurant</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Dog Run</td>
      <td>Dumpling Restaurant</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M4V</td>
      <td>Central Toronto</td>
      <td>Deer Park, Forest Hill SE, Rathnelly, South Hi...</td>
      <td>Coffee Shop</td>
      <td>Pub</td>
      <td>Pizza Place</td>
      <td>Fried Chicken Joint</td>
      <td>Restaurant</td>
      <td>Liquor Store</td>
      <td>Sports Bar</td>
      <td>Supermarket</td>
      <td>Sushi Restaurant</td>
      <td>Light Rail Station</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M4W</td>
      <td>Downtown Toronto</td>
      <td>Rosedale</td>
      <td>Park</td>
      <td>Playground</td>
      <td>Trail</td>
      <td>Yoga Studio</td>
      <td>Dumpling Restaurant</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M4X</td>
      <td>Downtown Toronto</td>
      <td>Cabbagetown, St. James Town</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Park</td>
      <td>Restaurant</td>
      <td>Pub</td>
      <td>Bakery</td>
      <td>Pizza Place</td>
      <td>Italian Restaurant</td>
      <td>Pharmacy</td>
      <td>Butcher</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M4Y</td>
      <td>Downtown Toronto</td>
      <td>Church and Wellesley</td>
      <td>Coffee Shop</td>
      <td>Japanese Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Gay Bar</td>
      <td>Restaurant</td>
      <td>Mediterranean Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Gym</td>
      <td>Hotel</td>
      <td>Dance Studio</td>
    </tr>
    <tr>
      <th>13</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Park</td>
      <td>Pub</td>
      <td>Mexican Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Theater</td>
      <td>Restaurant</td>
      <td>Café</td>
      <td>Chocolate Shop</td>
    </tr>
    <tr>
      <th>14</th>
      <td>M5B</td>
      <td>Downtown Toronto</td>
      <td>Ryerson, Garden District</td>
      <td>Coffee Shop</td>
      <td>Clothing Store</td>
      <td>Japanese Restaurant</td>
      <td>Café</td>
      <td>Cosmetics Shop</td>
      <td>Middle Eastern Restaurant</td>
      <td>Electronics Store</td>
      <td>Bakery</td>
      <td>Bookstore</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>15</th>
      <td>M5C</td>
      <td>Downtown Toronto</td>
      <td>St. James Town</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Clothing Store</td>
      <td>Beer Bar</td>
      <td>Hotel</td>
      <td>American Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Diner</td>
      <td>Cosmetics Shop</td>
    </tr>
    <tr>
      <th>16</th>
      <td>M5E</td>
      <td>Downtown Toronto</td>
      <td>Berczy Park</td>
      <td>Coffee Shop</td>
      <td>Farmers Market</td>
      <td>Beer Bar</td>
      <td>Steakhouse</td>
      <td>Bakery</td>
      <td>Cocktail Bar</td>
      <td>Cheese Shop</td>
      <td>Seafood Restaurant</td>
      <td>Café</td>
      <td>French Restaurant</td>
    </tr>
    <tr>
      <th>17</th>
      <td>M5G</td>
      <td>Downtown Toronto</td>
      <td>Central Bay Street</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Juice Bar</td>
      <td>Sandwich Place</td>
      <td>Burger Joint</td>
      <td>Ice Cream Shop</td>
      <td>Bakery</td>
      <td>Chinese Restaurant</td>
      <td>Bar</td>
    </tr>
    <tr>
      <th>18</th>
      <td>M5H</td>
      <td>Downtown Toronto</td>
      <td>Adelaide, King, Richmond</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Thai Restaurant</td>
      <td>Cosmetics Shop</td>
      <td>Burger Joint</td>
      <td>Restaurant</td>
      <td>Steakhouse</td>
      <td>Bakery</td>
      <td>Sushi Restaurant</td>
    </tr>
    <tr>
      <th>19</th>
      <td>M5J</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront East, Toronto Islands, Union Station</td>
      <td>Coffee Shop</td>
      <td>Aquarium</td>
      <td>Hotel</td>
      <td>Italian Restaurant</td>
      <td>Café</td>
      <td>Brewery</td>
      <td>Restaurant</td>
      <td>Scenic Lookout</td>
      <td>Fried Chicken Joint</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>20</th>
      <td>M5K</td>
      <td>Downtown Toronto</td>
      <td>Design Exchange, Toronto Dominion Centre</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Bakery</td>
      <td>Steakhouse</td>
      <td>Seafood Restaurant</td>
      <td>American Restaurant</td>
      <td>Bar</td>
    </tr>
    <tr>
      <th>21</th>
      <td>M5L</td>
      <td>Downtown Toronto</td>
      <td>Commerce Court, Victoria Hotel</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Restaurant</td>
      <td>Gym</td>
      <td>Deli / Bodega</td>
      <td>American Restaurant</td>
      <td>Bakery</td>
      <td>Steakhouse</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>22</th>
      <td>M5N</td>
      <td>Central Toronto</td>
      <td>Roselawn</td>
      <td>Garden</td>
      <td>Pool</td>
      <td>Dessert Shop</td>
      <td>Farmers Market</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
      <td>Electronics Store</td>
      <td>Eastern European Restaurant</td>
      <td>Dumpling Restaurant</td>
    </tr>
    <tr>
      <th>23</th>
      <td>M5P</td>
      <td>Central Toronto</td>
      <td>Forest Hill North, Forest Hill West</td>
      <td>Mexican Restaurant</td>
      <td>Trail</td>
      <td>Sushi Restaurant</td>
      <td>Jewelry Store</td>
      <td>Yoga Studio</td>
      <td>Diner</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>24</th>
      <td>M5R</td>
      <td>Central Toronto</td>
      <td>The Annex, North Midtown, Yorkville</td>
      <td>Sandwich Place</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Liquor Store</td>
      <td>BBQ Joint</td>
      <td>Pharmacy</td>
      <td>Cosmetics Shop</td>
      <td>Pizza Place</td>
      <td>Pub</td>
      <td>Middle Eastern Restaurant</td>
    </tr>
    <tr>
      <th>25</th>
      <td>M5S</td>
      <td>Downtown Toronto</td>
      <td>Harbord, University of Toronto</td>
      <td>Café</td>
      <td>Bakery</td>
      <td>Bookstore</td>
      <td>Japanese Restaurant</td>
      <td>Restaurant</td>
      <td>Bar</td>
      <td>Yoga Studio</td>
      <td>Chinese Restaurant</td>
      <td>Beer Store</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>26</th>
      <td>M5T</td>
      <td>Downtown Toronto</td>
      <td>Chinatown, Grange Park, Kensington Market</td>
      <td>Bar</td>
      <td>Café</td>
      <td>Vietnamese Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Coffee Shop</td>
      <td>Dumpling Restaurant</td>
      <td>Bakery</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Mexican Restaurant</td>
      <td>Farmers Market</td>
    </tr>
    <tr>
      <th>27</th>
      <td>M5V</td>
      <td>Downtown Toronto</td>
      <td>CN Tower, Bathurst Quay, Island airport, Harbo...</td>
      <td>Airport Service</td>
      <td>Airport Lounge</td>
      <td>Airport Terminal</td>
      <td>Plane</td>
      <td>Sculpture Garden</td>
      <td>Coffee Shop</td>
      <td>Boat or Ferry</td>
      <td>Harbor / Marina</td>
      <td>Boutique</td>
      <td>Airport Gate</td>
    </tr>
    <tr>
      <th>28</th>
      <td>M5W</td>
      <td>Downtown Toronto</td>
      <td>Stn A PO Boxes 25 The Esplanade</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Seafood Restaurant</td>
      <td>Hotel</td>
      <td>Beer Bar</td>
      <td>Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Farmers Market</td>
      <td>Creperie</td>
    </tr>
    <tr>
      <th>29</th>
      <td>M5X</td>
      <td>Downtown Toronto</td>
      <td>First Canadian Place, Underground city</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Burger Joint</td>
      <td>Deli / Bodega</td>
      <td>Gym</td>
      <td>Steakhouse</td>
      <td>American Restaurant</td>
      <td>Bakery</td>
      <td>Seafood Restaurant</td>
    </tr>
    <tr>
      <th>30</th>
      <td>M6G</td>
      <td>Downtown Toronto</td>
      <td>Christie</td>
      <td>Grocery Store</td>
      <td>Café</td>
      <td>Park</td>
      <td>Baby Store</td>
      <td>Gas Station</td>
      <td>Italian Restaurant</td>
      <td>Restaurant</td>
      <td>Candy Store</td>
      <td>Nightclub</td>
      <td>Athletics &amp; Sports</td>
    </tr>
    <tr>
      <th>31</th>
      <td>M6H</td>
      <td>West Toronto</td>
      <td>Dovercourt Village, Dufferin</td>
      <td>Bakery</td>
      <td>Pharmacy</td>
      <td>Brewery</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Bank</td>
      <td>Middle Eastern Restaurant</td>
      <td>Portuguese Restaurant</td>
      <td>Pool</td>
      <td>Music Venue</td>
    </tr>
    <tr>
      <th>32</th>
      <td>M6J</td>
      <td>West Toronto</td>
      <td>Little Portugal, Trinity</td>
      <td>Bar</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Asian Restaurant</td>
      <td>Vietnamese Restaurant</td>
      <td>Pizza Place</td>
      <td>Café</td>
      <td>Men's Store</td>
      <td>Yoga Studio</td>
      <td>Record Shop</td>
    </tr>
    <tr>
      <th>33</th>
      <td>M6K</td>
      <td>West Toronto</td>
      <td>Brockton, Exhibition Place, Parkdale Village</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Breakfast Spot</td>
      <td>Grocery Store</td>
      <td>Intersection</td>
      <td>Performing Arts Venue</td>
      <td>Pet Store</td>
      <td>Gym</td>
      <td>Nightclub</td>
      <td>Climbing Gym</td>
    </tr>
    <tr>
      <th>34</th>
      <td>M6P</td>
      <td>West Toronto</td>
      <td>High Park, The Junction South</td>
      <td>Café</td>
      <td>Mexican Restaurant</td>
      <td>Bar</td>
      <td>Thai Restaurant</td>
      <td>Grocery Store</td>
      <td>Furniture / Home Store</td>
      <td>Music Venue</td>
      <td>Diner</td>
      <td>Cajun / Creole Restaurant</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>35</th>
      <td>M6R</td>
      <td>West Toronto</td>
      <td>Parkdale, Roncesvalles</td>
      <td>Gift Shop</td>
      <td>Italian Restaurant</td>
      <td>Dessert Shop</td>
      <td>Bookstore</td>
      <td>Eastern European Restaurant</td>
      <td>Movie Theater</td>
      <td>Bar</td>
      <td>Bank</td>
      <td>Dog Run</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>36</th>
      <td>M6S</td>
      <td>West Toronto</td>
      <td>Runnymede, Swansea</td>
      <td>Pizza Place</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Sushi Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Yoga Studio</td>
      <td>Gastropub</td>
      <td>Restaurant</td>
      <td>Pub</td>
      <td>Latin American Restaurant</td>
    </tr>
    <tr>
      <th>37</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park</td>
      <td>Coffee Shop</td>
      <td>Gym</td>
      <td>Park</td>
      <td>Yoga Studio</td>
      <td>Japanese Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Portuguese Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Bar</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>38</th>
      <td>M7Y</td>
      <td>East Toronto</td>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>Pizza Place</td>
      <td>Auto Workshop</td>
      <td>Moving Target</td>
      <td>Restaurant</td>
      <td>Butcher</td>
      <td>Burrito Place</td>
      <td>Skate Park</td>
      <td>Brewery</td>
      <td>Farmers Market</td>
      <td>Comic Shop</td>
    </tr>
  </tbody>
</table>
</div>



### Cluster Areas
Run k-means to cluster the Toronto areas into 5 clusters.


```python
# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop(["PostalCode", "Borough", "Neighborhoods"], 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
```




    array([3, 1, 1, 1, 0, 1, 1, 1, 0, 1], dtype=int32)




```python
# create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.
toronto_merged = toronto_df_new.copy()

# add clustering labels
toronto_merged["Cluster Labels"] = kmeans.labels_

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.drop(["Borough", "Neighborhoods"], 1).set_index("PostalCode"), on="PostalCode")

print(toronto_merged.shape)
toronto_merged.head() # check the last columns!
```

    (39, 16)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>3</td>
      <td>Health Food Store</td>
      <td>Other Great Outdoors</td>
      <td>Trail</td>
      <td>Neighborhood</td>
      <td>Pub</td>
      <td>Eastern European Restaurant</td>
      <td>Dumpling Restaurant</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>1</td>
      <td>Greek Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Bookstore</td>
      <td>Furniture / Home Store</td>
      <td>Ice Cream Shop</td>
      <td>Fruit &amp; Vegetable Store</td>
      <td>Brewery</td>
      <td>Bubble Tea Shop</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>The Beaches West, India Bazaar</td>
      <td>43.668999</td>
      <td>-79.315572</td>
      <td>1</td>
      <td>Sandwich Place</td>
      <td>Pizza Place</td>
      <td>Park</td>
      <td>Ice Cream Shop</td>
      <td>Burger Joint</td>
      <td>Burrito Place</td>
      <td>Liquor Store</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Steakhouse</td>
      <td>Food &amp; Drink Shop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
      <td>43.659526</td>
      <td>-79.340923</td>
      <td>1</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Brewery</td>
      <td>Gastropub</td>
      <td>American Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Middle Eastern Restaurant</td>
      <td>Bar</td>
      <td>Clothing Store</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
      <td>43.728020</td>
      <td>-79.388790</td>
      <td>0</td>
      <td>Photography Studio</td>
      <td>Construction &amp; Landscaping</td>
      <td>Park</td>
      <td>Bus Line</td>
      <td>Swim School</td>
      <td>Diner</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
      <td>Electronics Store</td>
    </tr>
  </tbody>
</table>
</div>




```python
# sort the results by Cluster Labels
print(toronto_merged.shape)
toronto_merged.sort_values(["Cluster Labels"], inplace=True)
toronto_merged
```

    (39, 16)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>M4N</td>
      <td>Central Toronto</td>
      <td>Lawrence Park</td>
      <td>43.728020</td>
      <td>-79.388790</td>
      <td>0</td>
      <td>Photography Studio</td>
      <td>Construction &amp; Landscaping</td>
      <td>Park</td>
      <td>Bus Line</td>
      <td>Swim School</td>
      <td>Diner</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M4T</td>
      <td>Central Toronto</td>
      <td>Moore Park, Summerhill East</td>
      <td>43.689574</td>
      <td>-79.383160</td>
      <td>0</td>
      <td>Tennis Court</td>
      <td>Playground</td>
      <td>Park</td>
      <td>Restaurant</td>
      <td>Doner Restaurant</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Dog Run</td>
      <td>Dumpling Restaurant</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M4W</td>
      <td>Downtown Toronto</td>
      <td>Rosedale</td>
      <td>43.679563</td>
      <td>-79.377529</td>
      <td>0</td>
      <td>Park</td>
      <td>Playground</td>
      <td>Trail</td>
      <td>Yoga Studio</td>
      <td>Dumpling Restaurant</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
    </tr>
    <tr>
      <th>19</th>
      <td>M5J</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront East, Toronto Islands, Union Station</td>
      <td>43.640816</td>
      <td>-79.381752</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Aquarium</td>
      <td>Hotel</td>
      <td>Italian Restaurant</td>
      <td>Café</td>
      <td>Brewery</td>
      <td>Restaurant</td>
      <td>Scenic Lookout</td>
      <td>Fried Chicken Joint</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>24</th>
      <td>M5R</td>
      <td>Central Toronto</td>
      <td>The Annex, North Midtown, Yorkville</td>
      <td>43.672710</td>
      <td>-79.405678</td>
      <td>1</td>
      <td>Sandwich Place</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Liquor Store</td>
      <td>BBQ Joint</td>
      <td>Pharmacy</td>
      <td>Cosmetics Shop</td>
      <td>Pizza Place</td>
      <td>Pub</td>
      <td>Middle Eastern Restaurant</td>
    </tr>
    <tr>
      <th>25</th>
      <td>M5S</td>
      <td>Downtown Toronto</td>
      <td>Harbord, University of Toronto</td>
      <td>43.662696</td>
      <td>-79.400049</td>
      <td>1</td>
      <td>Café</td>
      <td>Bakery</td>
      <td>Bookstore</td>
      <td>Japanese Restaurant</td>
      <td>Restaurant</td>
      <td>Bar</td>
      <td>Yoga Studio</td>
      <td>Chinese Restaurant</td>
      <td>Beer Store</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>26</th>
      <td>M5T</td>
      <td>Downtown Toronto</td>
      <td>Chinatown, Grange Park, Kensington Market</td>
      <td>43.653206</td>
      <td>-79.400049</td>
      <td>1</td>
      <td>Bar</td>
      <td>Café</td>
      <td>Vietnamese Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Coffee Shop</td>
      <td>Dumpling Restaurant</td>
      <td>Bakery</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Mexican Restaurant</td>
      <td>Farmers Market</td>
    </tr>
    <tr>
      <th>27</th>
      <td>M5V</td>
      <td>Downtown Toronto</td>
      <td>CN Tower, Bathurst Quay, Island airport, Harbo...</td>
      <td>43.628947</td>
      <td>-79.394420</td>
      <td>1</td>
      <td>Airport Service</td>
      <td>Airport Lounge</td>
      <td>Airport Terminal</td>
      <td>Plane</td>
      <td>Sculpture Garden</td>
      <td>Coffee Shop</td>
      <td>Boat or Ferry</td>
      <td>Harbor / Marina</td>
      <td>Boutique</td>
      <td>Airport Gate</td>
    </tr>
    <tr>
      <th>28</th>
      <td>M5W</td>
      <td>Downtown Toronto</td>
      <td>Stn A PO Boxes 25 The Esplanade</td>
      <td>43.646435</td>
      <td>-79.374846</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Seafood Restaurant</td>
      <td>Hotel</td>
      <td>Beer Bar</td>
      <td>Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Farmers Market</td>
      <td>Creperie</td>
    </tr>
    <tr>
      <th>29</th>
      <td>M5X</td>
      <td>Downtown Toronto</td>
      <td>First Canadian Place, Underground city</td>
      <td>43.648429</td>
      <td>-79.382280</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Burger Joint</td>
      <td>Deli / Bodega</td>
      <td>Gym</td>
      <td>Steakhouse</td>
      <td>American Restaurant</td>
      <td>Bakery</td>
      <td>Seafood Restaurant</td>
    </tr>
    <tr>
      <th>30</th>
      <td>M6G</td>
      <td>Downtown Toronto</td>
      <td>Christie</td>
      <td>43.669542</td>
      <td>-79.422564</td>
      <td>1</td>
      <td>Grocery Store</td>
      <td>Café</td>
      <td>Park</td>
      <td>Baby Store</td>
      <td>Gas Station</td>
      <td>Italian Restaurant</td>
      <td>Restaurant</td>
      <td>Candy Store</td>
      <td>Nightclub</td>
      <td>Athletics &amp; Sports</td>
    </tr>
    <tr>
      <th>31</th>
      <td>M6H</td>
      <td>West Toronto</td>
      <td>Dovercourt Village, Dufferin</td>
      <td>43.669005</td>
      <td>-79.442259</td>
      <td>1</td>
      <td>Bakery</td>
      <td>Pharmacy</td>
      <td>Brewery</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Bank</td>
      <td>Middle Eastern Restaurant</td>
      <td>Portuguese Restaurant</td>
      <td>Pool</td>
      <td>Music Venue</td>
    </tr>
    <tr>
      <th>32</th>
      <td>M6J</td>
      <td>West Toronto</td>
      <td>Little Portugal, Trinity</td>
      <td>43.647927</td>
      <td>-79.419750</td>
      <td>1</td>
      <td>Bar</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Asian Restaurant</td>
      <td>Vietnamese Restaurant</td>
      <td>Pizza Place</td>
      <td>Café</td>
      <td>Men's Store</td>
      <td>Yoga Studio</td>
      <td>Record Shop</td>
    </tr>
    <tr>
      <th>33</th>
      <td>M6K</td>
      <td>West Toronto</td>
      <td>Brockton, Exhibition Place, Parkdale Village</td>
      <td>43.636847</td>
      <td>-79.428191</td>
      <td>1</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Breakfast Spot</td>
      <td>Grocery Store</td>
      <td>Intersection</td>
      <td>Performing Arts Venue</td>
      <td>Pet Store</td>
      <td>Gym</td>
      <td>Nightclub</td>
      <td>Climbing Gym</td>
    </tr>
    <tr>
      <th>34</th>
      <td>M6P</td>
      <td>West Toronto</td>
      <td>High Park, The Junction South</td>
      <td>43.661608</td>
      <td>-79.464763</td>
      <td>1</td>
      <td>Café</td>
      <td>Mexican Restaurant</td>
      <td>Bar</td>
      <td>Thai Restaurant</td>
      <td>Grocery Store</td>
      <td>Furniture / Home Store</td>
      <td>Music Venue</td>
      <td>Diner</td>
      <td>Cajun / Creole Restaurant</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>35</th>
      <td>M6R</td>
      <td>West Toronto</td>
      <td>Parkdale, Roncesvalles</td>
      <td>43.648960</td>
      <td>-79.456325</td>
      <td>1</td>
      <td>Gift Shop</td>
      <td>Italian Restaurant</td>
      <td>Dessert Shop</td>
      <td>Bookstore</td>
      <td>Eastern European Restaurant</td>
      <td>Movie Theater</td>
      <td>Bar</td>
      <td>Bank</td>
      <td>Dog Run</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>36</th>
      <td>M6S</td>
      <td>West Toronto</td>
      <td>Runnymede, Swansea</td>
      <td>43.651571</td>
      <td>-79.484450</td>
      <td>1</td>
      <td>Pizza Place</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Sushi Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Yoga Studio</td>
      <td>Gastropub</td>
      <td>Restaurant</td>
      <td>Pub</td>
      <td>Latin American Restaurant</td>
    </tr>
    <tr>
      <th>21</th>
      <td>M5L</td>
      <td>Downtown Toronto</td>
      <td>Commerce Court, Victoria Hotel</td>
      <td>43.648198</td>
      <td>-79.379817</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Restaurant</td>
      <td>Gym</td>
      <td>Deli / Bodega</td>
      <td>American Restaurant</td>
      <td>Bakery</td>
      <td>Steakhouse</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>20</th>
      <td>M5K</td>
      <td>Downtown Toronto</td>
      <td>Design Exchange, Toronto Dominion Centre</td>
      <td>43.647177</td>
      <td>-79.381576</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Bakery</td>
      <td>Steakhouse</td>
      <td>Seafood Restaurant</td>
      <td>American Restaurant</td>
      <td>Bar</td>
    </tr>
    <tr>
      <th>38</th>
      <td>M7Y</td>
      <td>East Toronto</td>
      <td>Business Reply Mail Processing Centre 969 Eastern</td>
      <td>43.662744</td>
      <td>-79.321558</td>
      <td>1</td>
      <td>Pizza Place</td>
      <td>Auto Workshop</td>
      <td>Moving Target</td>
      <td>Restaurant</td>
      <td>Butcher</td>
      <td>Burrito Place</td>
      <td>Skate Park</td>
      <td>Brewery</td>
      <td>Farmers Market</td>
      <td>Comic Shop</td>
    </tr>
    <tr>
      <th>18</th>
      <td>M5H</td>
      <td>Downtown Toronto</td>
      <td>Adelaide, King, Richmond</td>
      <td>43.650571</td>
      <td>-79.384568</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Thai Restaurant</td>
      <td>Cosmetics Shop</td>
      <td>Burger Joint</td>
      <td>Restaurant</td>
      <td>Steakhouse</td>
      <td>Bakery</td>
      <td>Sushi Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
      <td>1</td>
      <td>Greek Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Bookstore</td>
      <td>Furniture / Home Store</td>
      <td>Ice Cream Shop</td>
      <td>Fruit &amp; Vegetable Store</td>
      <td>Brewery</td>
      <td>Bubble Tea Shop</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>The Beaches West, India Bazaar</td>
      <td>43.668999</td>
      <td>-79.315572</td>
      <td>1</td>
      <td>Sandwich Place</td>
      <td>Pizza Place</td>
      <td>Park</td>
      <td>Ice Cream Shop</td>
      <td>Burger Joint</td>
      <td>Burrito Place</td>
      <td>Liquor Store</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Steakhouse</td>
      <td>Food &amp; Drink Shop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4M</td>
      <td>East Toronto</td>
      <td>Studio District</td>
      <td>43.659526</td>
      <td>-79.340923</td>
      <td>1</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Brewery</td>
      <td>Gastropub</td>
      <td>American Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Middle Eastern Restaurant</td>
      <td>Bar</td>
      <td>Clothing Store</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M4P</td>
      <td>Central Toronto</td>
      <td>Davisville North</td>
      <td>43.712751</td>
      <td>-79.390197</td>
      <td>1</td>
      <td>Breakfast Spot</td>
      <td>Park</td>
      <td>Department Store</td>
      <td>Food &amp; Drink Shop</td>
      <td>Hotel</td>
      <td>Sandwich Place</td>
      <td>Gym</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M4R</td>
      <td>Central Toronto</td>
      <td>North Toronto West</td>
      <td>43.715383</td>
      <td>-79.405678</td>
      <td>1</td>
      <td>Clothing Store</td>
      <td>Coffee Shop</td>
      <td>Yoga Studio</td>
      <td>Bagel Shop</td>
      <td>Pet Store</td>
      <td>Chinese Restaurant</td>
      <td>Dessert Shop</td>
      <td>Restaurant</td>
      <td>Diner</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M4S</td>
      <td>Central Toronto</td>
      <td>Davisville</td>
      <td>43.704324</td>
      <td>-79.388790</td>
      <td>1</td>
      <td>Sandwich Place</td>
      <td>Pizza Place</td>
      <td>Dessert Shop</td>
      <td>Gym</td>
      <td>Café</td>
      <td>Italian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Sushi Restaurant</td>
      <td>Pharmacy</td>
      <td>Deli / Bodega</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M4V</td>
      <td>Central Toronto</td>
      <td>Deer Park, Forest Hill SE, Rathnelly, South Hi...</td>
      <td>43.686412</td>
      <td>-79.400049</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Pub</td>
      <td>Pizza Place</td>
      <td>Fried Chicken Joint</td>
      <td>Restaurant</td>
      <td>Liquor Store</td>
      <td>Sports Bar</td>
      <td>Supermarket</td>
      <td>Sushi Restaurant</td>
      <td>Light Rail Station</td>
    </tr>
    <tr>
      <th>37</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park</td>
      <td>43.662301</td>
      <td>-79.389494</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Gym</td>
      <td>Park</td>
      <td>Yoga Studio</td>
      <td>Japanese Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Portuguese Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Bar</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M4X</td>
      <td>Downtown Toronto</td>
      <td>Cabbagetown, St. James Town</td>
      <td>43.667967</td>
      <td>-79.367675</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Park</td>
      <td>Restaurant</td>
      <td>Pub</td>
      <td>Bakery</td>
      <td>Pizza Place</td>
      <td>Italian Restaurant</td>
      <td>Pharmacy</td>
      <td>Butcher</td>
    </tr>
    <tr>
      <th>13</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
      <td>43.654260</td>
      <td>-79.360636</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Park</td>
      <td>Pub</td>
      <td>Mexican Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Theater</td>
      <td>Restaurant</td>
      <td>Café</td>
      <td>Chocolate Shop</td>
    </tr>
    <tr>
      <th>14</th>
      <td>M5B</td>
      <td>Downtown Toronto</td>
      <td>Ryerson, Garden District</td>
      <td>43.657162</td>
      <td>-79.378937</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Clothing Store</td>
      <td>Japanese Restaurant</td>
      <td>Café</td>
      <td>Cosmetics Shop</td>
      <td>Middle Eastern Restaurant</td>
      <td>Electronics Store</td>
      <td>Bakery</td>
      <td>Bookstore</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>15</th>
      <td>M5C</td>
      <td>Downtown Toronto</td>
      <td>St. James Town</td>
      <td>43.651494</td>
      <td>-79.375418</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Clothing Store</td>
      <td>Beer Bar</td>
      <td>Hotel</td>
      <td>American Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Diner</td>
      <td>Cosmetics Shop</td>
    </tr>
    <tr>
      <th>16</th>
      <td>M5E</td>
      <td>Downtown Toronto</td>
      <td>Berczy Park</td>
      <td>43.644771</td>
      <td>-79.373306</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Farmers Market</td>
      <td>Beer Bar</td>
      <td>Steakhouse</td>
      <td>Bakery</td>
      <td>Cocktail Bar</td>
      <td>Cheese Shop</td>
      <td>Seafood Restaurant</td>
      <td>Café</td>
      <td>French Restaurant</td>
    </tr>
    <tr>
      <th>17</th>
      <td>M5G</td>
      <td>Downtown Toronto</td>
      <td>Central Bay Street</td>
      <td>43.657952</td>
      <td>-79.387383</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Juice Bar</td>
      <td>Sandwich Place</td>
      <td>Burger Joint</td>
      <td>Ice Cream Shop</td>
      <td>Bakery</td>
      <td>Chinese Restaurant</td>
      <td>Bar</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M4Y</td>
      <td>Downtown Toronto</td>
      <td>Church and Wellesley</td>
      <td>43.665860</td>
      <td>-79.383160</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Japanese Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Gay Bar</td>
      <td>Restaurant</td>
      <td>Mediterranean Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Gym</td>
      <td>Hotel</td>
      <td>Dance Studio</td>
    </tr>
    <tr>
      <th>22</th>
      <td>M5N</td>
      <td>Central Toronto</td>
      <td>Roselawn</td>
      <td>43.711695</td>
      <td>-79.416936</td>
      <td>2</td>
      <td>Garden</td>
      <td>Pool</td>
      <td>Dessert Shop</td>
      <td>Farmers Market</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
      <td>Electronics Store</td>
      <td>Eastern European Restaurant</td>
      <td>Dumpling Restaurant</td>
    </tr>
    <tr>
      <th>0</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
      <td>3</td>
      <td>Health Food Store</td>
      <td>Other Great Outdoors</td>
      <td>Trail</td>
      <td>Neighborhood</td>
      <td>Pub</td>
      <td>Eastern European Restaurant</td>
      <td>Dumpling Restaurant</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
    </tr>
    <tr>
      <th>23</th>
      <td>M5P</td>
      <td>Central Toronto</td>
      <td>Forest Hill North, Forest Hill West</td>
      <td>43.696948</td>
      <td>-79.411307</td>
      <td>4</td>
      <td>Mexican Restaurant</td>
      <td>Trail</td>
      <td>Sushi Restaurant</td>
      <td>Jewelry Store</td>
      <td>Yoga Studio</td>
      <td>Diner</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
      <td>Electronics Store</td>
    </tr>
  </tbody>
</table>
</div>



**Visualize the resulting clusters**


```python
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, post, bor, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['PostalCode'], toronto_merged['Borough'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup('{} ({}): {} - Cluster {}'.format(bor, post, poi, cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfYzM3NDcyNGRmOGQwNGE4MDk0NDllYjA4M2ExNGNmMTcgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2MzNzQ3MjRkZjhkMDRhODA5NDQ5ZWIwODNhMTRjZjE3IiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF9jMzc0NzI0ZGY4ZDA0YTgwOTQ0OWViMDgzYTE0Y2YxNyA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF9jMzc0NzI0ZGY4ZDA0YTgwOTQ0OWViMDgzYTE0Y2YxNycsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjUzOTYzLC03OS4zODcyMDddLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgem9vbTogMTEsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYXhCb3VuZHM6IGJvdW5kcywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGxheWVyczogW10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3b3JsZENvcHlKdW1wOiBmYWxzZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSk7CiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyXzI2NzVlYThkOTRkYjQ4NTJhN2UxYjdjZmE5MWEzMjBmID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgICAgICAgICAgewogICJhdHRyaWJ1dGlvbiI6IG51bGwsCiAgImRldGVjdFJldGluYSI6IGZhbHNlLAogICJtYXhab29tIjogMTgsCiAgIm1pblpvb20iOiAxLAogICJub1dyYXAiOiBmYWxzZSwKICAic3ViZG9tYWlucyI6ICJhYmMiCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2MzNzQ3MjRkZjhkMDRhODA5NDQ5ZWIwODNhMTRjZjE3KTsKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mOTc3YTU0NDAwZDY0ODRjOTdmN2JlZjg3MmFhOTYxNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyODAyMDUsLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYzM3NDcyNGRmOGQwNGE4MDk0NDllYjA4M2ExNGNmMTcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzBiZGQxNTExODhhNGZhYjg2MmYwNmFiYWRjNTljN2EgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfM2UzOTg3YjhmNWM4NDkzOTg5MTkxYmQzMzJmNDM4NWQgPSAkKCc8ZGl2IGlkPSJodG1sXzNlMzk4N2I4ZjVjODQ5Mzk4OTE5MWJkMzMyZjQzODVkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZW50cmFsIFRvcm9udG8gKE00Tik6IExhd3JlbmNlIFBhcmsgLSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MwYmRkMTUxMTg4YTRmYWI4NjJmMDZhYmFkYzU5YzdhLnNldENvbnRlbnQoaHRtbF8zZTM5ODdiOGY1Yzg0OTM5ODkxOTFiZDMzMmY0Mzg1ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mOTc3YTU0NDAwZDY0ODRjOTdmN2JlZjg3MmFhOTYxNC5iaW5kUG9wdXAocG9wdXBfYzBiZGQxNTExODhhNGZhYjg2MmYwNmFiYWRjNTljN2EpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTBhZDY5ODdkYzIxNGE3M2E0MTYzYTdlZjUwYmEzNjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODk1NzQzLC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jMzc0NzI0ZGY4ZDA0YTgwOTQ0OWViMDgzYTE0Y2YxNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81OTFmZWE1MWU5ODk0OTMxYWNjZTVhNWVhNDJjYTMwYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wMjZhZWRlOTQ5NzI0ZmJmYmUzOGQ4ZDMwZDIyMzgyMCA9ICQoJzxkaXYgaWQ9Imh0bWxfMDI2YWVkZTk0OTcyNGZiZmJlMzhkOGQzMGQyMjM4MjAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgVG9yb250byAoTTRUKTogTW9vcmUgUGFyaywgU3VtbWVyaGlsbCBFYXN0IC0gQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81OTFmZWE1MWU5ODk0OTMxYWNjZTVhNWVhNDJjYTMwYy5zZXRDb250ZW50KGh0bWxfMDI2YWVkZTk0OTcyNGZiZmJlMzhkOGQzMGQyMjM4MjApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTBhZDY5ODdkYzIxNGE3M2E0MTYzYTdlZjUwYmEzNjguYmluZFBvcHVwKHBvcHVwXzU5MWZlYTUxZTk4OTQ5MzFhY2NlNWE1ZWE0MmNhMzBjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzg1MWUwMjdlY2JmMDRmNTY4ZWUzNmEzYmEyNzMxZDY2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc5NTYyNiwtNzkuMzc3NTI5NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYzM3NDcyNGRmOGQwNGE4MDk0NDllYjA4M2ExNGNmMTcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTRiNjRlNjZjYjAxNDc3MjkyMDdlZDViZDhiOTBkYmUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDdhMDhlOWZiMTdlNGNlYWE3OWVmYjE3YzllZTI2NzEgPSAkKCc8ZGl2IGlkPSJodG1sXzA3YTA4ZTlmYjE3ZTRjZWFhNzllZmIxN2M5ZWUyNjcxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvIChNNFcpOiBSb3NlZGFsZSAtIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTRiNjRlNjZjYjAxNDc3MjkyMDdlZDViZDhiOTBkYmUuc2V0Q29udGVudChodG1sXzA3YTA4ZTlmYjE3ZTRjZWFhNzllZmIxN2M5ZWUyNjcxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg1MWUwMjdlY2JmMDRmNTY4ZWUzNmEzYmEyNzMxZDY2LmJpbmRQb3B1cChwb3B1cF9hNGI2NGU2NmNiMDE0NzcyOTIwN2VkNWJkOGI5MGRiZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mZDg4NjIxNDM5OTY0Yjk4YWE1YjFhNjg0MTNjNzk5ZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0MDgxNTcsLTc5LjM4MTc1MjI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2MzNzQ3MjRkZjhkMDRhODA5NDQ5ZWIwODNhMTRjZjE3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2QwZmYxYmIxMjA0NjQ0MDI5Yzk4YjJhODY5YmY4OWMxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRmZWNjMTRmNzg1YTRlNjRhZjg5MDgxY2ZjODkwYWQ1ID0gJCgnPGRpdiBpZD0iaHRtbF80ZmVjYzE0Zjc4NWE0ZTY0YWY4OTA4MWNmYzg5MGFkNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250byAoTTVKKTogSGFyYm91cmZyb250IEVhc3QsIFRvcm9udG8gSXNsYW5kcywgVW5pb24gU3RhdGlvbiAtIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDBmZjFiYjEyMDQ2NDQwMjljOThiMmE4NjliZjg5YzEuc2V0Q29udGVudChodG1sXzRmZWNjMTRmNzg1YTRlNjRhZjg5MDgxY2ZjODkwYWQ1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZkODg2MjE0Mzk5NjRiOThhYTViMWE2ODQxM2M3OTlmLmJpbmRQb3B1cChwb3B1cF9kMGZmMWJiMTIwNDY0NDAyOWM5OGIyYTg2OWJmODljMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yYzY2Mzc2YjE4YmQ0ZmU5YWI2NDk4YTZiMjk4ZTczYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3MjcwOTcsLTc5LjQwNTY3ODQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2MzNzQ3MjRkZjhkMDRhODA5NDQ5ZWIwODNhMTRjZjE3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I5OTBkNjgxNWFmOTRlZGU4MWIxNGU5ZDljYzI4NzI2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk5ZDIzY2VjMjg5YTQ0MTBiNjQ0ZTI0OWUwOThlNzQ3ID0gJCgnPGRpdiBpZD0iaHRtbF85OWQyM2NlYzI4OWE0NDEwYjY0NGUyNDllMDk4ZTc0NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBUb3JvbnRvIChNNVIpOiBUaGUgQW5uZXgsIE5vcnRoIE1pZHRvd24sIFlvcmt2aWxsZSAtIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjk5MGQ2ODE1YWY5NGVkZTgxYjE0ZTlkOWNjMjg3MjYuc2V0Q29udGVudChodG1sXzk5ZDIzY2VjMjg5YTQ0MTBiNjQ0ZTI0OWUwOThlNzQ3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJjNjYzNzZiMThiZDRmZTlhYjY0OThhNmIyOThlNzNiLmJpbmRQb3B1cChwb3B1cF9iOTkwZDY4MTVhZjk0ZWRlODFiMTRlOWQ5Y2MyODcyNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84Yzg3YWYwZjgzOGM0ODJhYWI4ZGFjNjNmOTUwYWY2YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MjY5NTYsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYzM3NDcyNGRmOGQwNGE4MDk0NDllYjA4M2ExNGNmMTcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzFlMjcyNmQ1OWNjNGQ2ZDliMTUxMTA5NzIwZWNlNDMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzcxOTM3ZWIyNTBhNGQwN2JhZjg4NDA5NWZmODExMWEgPSAkKCc8ZGl2IGlkPSJodG1sXzM3MTkzN2ViMjUwYTRkMDdiYWY4ODQwOTVmZjgxMTFhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvIChNNVMpOiBIYXJib3JkLCBVbml2ZXJzaXR5IG9mIFRvcm9udG8gLSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MxZTI3MjZkNTljYzRkNmQ5YjE1MTEwOTcyMGVjZTQzLnNldENvbnRlbnQoaHRtbF8zNzE5MzdlYjI1MGE0ZDA3YmFmODg0MDk1ZmY4MTExYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84Yzg3YWYwZjgzOGM0ODJhYWI4ZGFjNjNmOTUwYWY2Yi5iaW5kUG9wdXAocG9wdXBfYzFlMjcyNmQ1OWNjNGQ2ZDliMTUxMTA5NzIwZWNlNDMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDZmNjUyODg1NmYxNDJlNzg3OWI5ZTM1NzA4YjRhYmYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTMyMDU3LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2MzNzQ3MjRkZjhkMDRhODA5NDQ5ZWIwODNhMTRjZjE3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y1MTg0OWM5YzgzNTQxZDZiZDgxNTFkNjUzN2UzYmZiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2U2ZWQyNzdiNGI0MzQzMTI4N2Y0YzRjN2FjZmQyOWYzID0gJCgnPGRpdiBpZD0iaHRtbF9lNmVkMjc3YjRiNDM0MzEyODdmNGM0YzdhY2ZkMjlmMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250byAoTTVUKTogQ2hpbmF0b3duLCBHcmFuZ2UgUGFyaywgS2Vuc2luZ3RvbiBNYXJrZXQgLSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y1MTg0OWM5YzgzNTQxZDZiZDgxNTFkNjUzN2UzYmZiLnNldENvbnRlbnQoaHRtbF9lNmVkMjc3YjRiNDM0MzEyODdmNGM0YzdhY2ZkMjlmMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wNmY2NTI4ODU2ZjE0MmU3ODc5YjllMzU3MDhiNGFiZi5iaW5kUG9wdXAocG9wdXBfZjUxODQ5YzljODM1NDFkNmJkODE1MWQ2NTM3ZTNiZmIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjU4YWM1OTAyNDQ2NDNmMzg5Mjk5NjI2OWI3MjJiNmEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg5NDY3LC03OS4zOTQ0MTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2MzNzQ3MjRkZjhkMDRhODA5NDQ5ZWIwODNhMTRjZjE3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2NkZDcyNDJmNTdmZTQ1MWZiNmYyOTIyZmJmYWYxNDk4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2VjZGYyY2ZmYWViNzQ5NTZiMTViNDIxMjQ2NzE1ZGEzID0gJCgnPGRpdiBpZD0iaHRtbF9lY2RmMmNmZmFlYjc0OTU2YjE1YjQyMTI0NjcxNWRhMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250byAoTTVWKTogQ04gVG93ZXIsIEJhdGh1cnN0IFF1YXksIElzbGFuZCBhaXJwb3J0LCBIYXJib3VyZnJvbnQgV2VzdCwgS2luZyBhbmQgU3BhZGluYSwgUmFpbHdheSBMYW5kcywgU291dGggTmlhZ2FyYSAtIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2RkNzI0MmY1N2ZlNDUxZmI2ZjI5MjJmYmZhZjE0OTguc2V0Q29udGVudChodG1sX2VjZGYyY2ZmYWViNzQ5NTZiMTViNDIxMjQ2NzE1ZGEzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I1OGFjNTkwMjQ0NjQzZjM4OTI5OTYyNjliNzIyYjZhLmJpbmRQb3B1cChwb3B1cF9jZGQ3MjQyZjU3ZmU0NTFmYjZmMjkyMmZiZmFmMTQ5OCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xNTJjZDdiNTgyMTI0ODMzYWJmZWFjNTM2YzA3MTViOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NjQzNTIsLTc5LjM3NDg0NTk5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2MzNzQ3MjRkZjhkMDRhODA5NDQ5ZWIwODNhMTRjZjE3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzM2OTEyYzYwNDllODQ4MWI4YjM3NDJhMmE3ZTViNzRiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzljMTFlMmZiYjNhYTQ1OGY4NGZiNjM0ZmE0MDFiYmUxID0gJCgnPGRpdiBpZD0iaHRtbF85YzExZTJmYmIzYWE0NThmODRmYjYzNGZhNDAxYmJlMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250byAoTTVXKTogU3RuIEEgUE8gQm94ZXMgMjUgVGhlIEVzcGxhbmFkZSAtIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzY5MTJjNjA0OWU4NDgxYjhiMzc0MmEyYTdlNWI3NGIuc2V0Q29udGVudChodG1sXzljMTFlMmZiYjNhYTQ1OGY4NGZiNjM0ZmE0MDFiYmUxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE1MmNkN2I1ODIxMjQ4MzNhYmZlYWM1MzZjMDcxNWI4LmJpbmRQb3B1cChwb3B1cF8zNjkxMmM2MDQ5ZTg0ODFiOGIzNzQyYTJhN2U1Yjc0Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yZTYyYmRkYjYwZDg0NmYzYTE0NDZhYjBiMzlkZjg2YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODQyOTIsLTc5LjM4MjI4MDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYzM3NDcyNGRmOGQwNGE4MDk0NDllYjA4M2ExNGNmMTcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzVjZWM2ZWE5MGZlNDNkZGE4NTJjNTYzZTAyNTdkYTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODI1YjhlNDExNjQ2NGNkYjlmZTQ4YWE1ZGFmMzA2NWUgPSAkKCc8ZGl2IGlkPSJodG1sXzgyNWI4ZTQxMTY0NjRjZGI5ZmU0OGFhNWRhZjMwNjVlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvIChNNVgpOiBGaXJzdCBDYW5hZGlhbiBQbGFjZSwgVW5kZXJncm91bmQgY2l0eSAtIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzVjZWM2ZWE5MGZlNDNkZGE4NTJjNTYzZTAyNTdkYTguc2V0Q29udGVudChodG1sXzgyNWI4ZTQxMTY0NjRjZGI5ZmU0OGFhNWRhZjMwNjVlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJlNjJiZGRiNjBkODQ2ZjNhMTQ0NmFiMGIzOWRmODZjLmJpbmRQb3B1cChwb3B1cF83NWNlYzZlYTkwZmU0M2RkYTg1MmM1NjNlMDI1N2RhOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wODgyOGE1MTY3OGE0YzEyYjRhZWRhMDgzOWFiMjhlYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTU0MiwtNzkuNDIyNTYzN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jMzc0NzI0ZGY4ZDA0YTgwOTQ0OWViMDgzYTE0Y2YxNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kZmNiYzNiMjA4Y2E0YjYzYTkzMGRjYjEwOWIyYTFiYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jMTQ4ZWUzMzQxODQ0NjBiOWJhMDA4MzIzMmVkMmUyNiA9ICQoJzxkaXYgaWQ9Imh0bWxfYzE0OGVlMzM0MTg0NDYwYjliYTAwODMyMzJlZDJlMjYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG8gKE02Ryk6IENocmlzdGllIC0gQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kZmNiYzNiMjA4Y2E0YjYzYTkzMGRjYjEwOWIyYTFiYy5zZXRDb250ZW50KGh0bWxfYzE0OGVlMzM0MTg0NDYwYjliYTAwODMyMzJlZDJlMjYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDg4MjhhNTE2NzhhNGMxMmI0YWVkYTA4MzlhYjI4ZWEuYmluZFBvcHVwKHBvcHVwX2RmY2JjM2IyMDhjYTRiNjNhOTMwZGNiMTA5YjJhMWJjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc0MTkwZjc2OGY2OTRlOTRhYjBmYWJkODZlMDdjZTdiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5MDA1MTAwMDAwMDEsLTc5LjQ0MjI1OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYzM3NDcyNGRmOGQwNGE4MDk0NDllYjA4M2ExNGNmMTcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjg1M2M1MmNlY2U5NDYyOTgwMGZhNWY5YTBlZWNjYTYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNmVkZTM5YWI2ODNiNDRlNTlkMDA4ZTlhNjAxYjY3NGYgPSAkKCc8ZGl2IGlkPSJodG1sXzZlZGUzOWFiNjgzYjQ0ZTU5ZDAwOGU5YTYwMWI2NzRmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XZXN0IFRvcm9udG8gKE02SCk6IERvdmVyY291cnQgVmlsbGFnZSwgRHVmZmVyaW4gLSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY4NTNjNTJjZWNlOTQ2Mjk4MDBmYTVmOWEwZWVjY2E2LnNldENvbnRlbnQoaHRtbF82ZWRlMzlhYjY4M2I0NGU1OWQwMDhlOWE2MDFiNjc0Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83NDE5MGY3NjhmNjk0ZTk0YWIwZmFiZDg2ZTA3Y2U3Yi5iaW5kUG9wdXAocG9wdXBfNjg1M2M1MmNlY2U5NDYyOTgwMGZhNWY5YTBlZWNjYTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNWZjMWQzOGNhMDZiNDE1NTk4Nzc0NzFmNmIyZWUyMTMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDc5MjY3MDAwMDAwMDYsLTc5LjQxOTc0OTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYzM3NDcyNGRmOGQwNGE4MDk0NDllYjA4M2ExNGNmMTcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzA4NTdjNzhjYzFjNDk2ZDg4NzMyYTdmNDk1NmRiNzUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTRhMGZmYTU0MWFmNGZiMjhjMzU4NTViOTI2OGRjY2MgPSAkKCc8ZGl2IGlkPSJodG1sX2U0YTBmZmE1NDFhZjRmYjI4YzM1ODU1YjkyNjhkY2NjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XZXN0IFRvcm9udG8gKE02Sik6IExpdHRsZSBQb3J0dWdhbCwgVHJpbml0eSAtIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzA4NTdjNzhjYzFjNDk2ZDg4NzMyYTdmNDk1NmRiNzUuc2V0Q29udGVudChodG1sX2U0YTBmZmE1NDFhZjRmYjI4YzM1ODU1YjkyNjhkY2NjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzVmYzFkMzhjYTA2YjQxNTU5ODc3NDcxZjZiMmVlMjEzLmJpbmRQb3B1cChwb3B1cF9jMDg1N2M3OGNjMWM0OTZkODg3MzJhN2Y0OTU2ZGI3NSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNTFiNzQ3ZmU2YTE0MDk0ODcwYmYzMDdiYWMxODNjMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYzNjg0NzIsLTc5LjQyODE5MTQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2MzNzQ3MjRkZjhkMDRhODA5NDQ5ZWIwODNhMTRjZjE3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQyODhkYmYxODY0ZDQ4MmU5M2Q5NTdlYTRhOWI1NDQ5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2IxYzU4YTg4ZDBmYzRhZWZiZjA3NTcxNGIzNGVlZmEwID0gJCgnPGRpdiBpZD0iaHRtbF9iMWM1OGE4OGQwZmM0YWVmYmYwNzU3MTRiMzRlZWZhMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdCBUb3JvbnRvIChNNkspOiBCcm9ja3RvbiwgRXhoaWJpdGlvbiBQbGFjZSwgUGFya2RhbGUgVmlsbGFnZSAtIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDI4OGRiZjE4NjRkNDgyZTkzZDk1N2VhNGE5YjU0NDkuc2V0Q29udGVudChodG1sX2IxYzU4YTg4ZDBmYzRhZWZiZjA3NTcxNGIzNGVlZmEwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y1MWI3NDdmZTZhMTQwOTQ4NzBiZjMwN2JhYzE4M2MzLmJpbmRQb3B1cChwb3B1cF80Mjg4ZGJmMTg2NGQ0ODJlOTNkOTU3ZWE0YTliNTQ0OSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kMjNmNjZmYzIwOTA0ZWJjOGVjNDIwNDY1NGViZTE5MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MTYwODMsLTc5LjQ2NDc2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2MzNzQ3MjRkZjhkMDRhODA5NDQ5ZWIwODNhMTRjZjE3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2YwOGQ1YTA3Njc3OTQ1ZjBhODQyY2E4MmM4MzkzOGQxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2U3ZmU1NjMxZTVkNjQyMjM5NjU5NjhhYWExY2ExNmUxID0gJCgnPGRpdiBpZD0iaHRtbF9lN2ZlNTYzMWU1ZDY0MjIzOTY1OTY4YWFhMWNhMTZlMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdCBUb3JvbnRvIChNNlApOiBIaWdoIFBhcmssIFRoZSBKdW5jdGlvbiBTb3V0aCAtIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjA4ZDVhMDc2Nzc5NDVmMGE4NDJjYTgyYzgzOTM4ZDEuc2V0Q29udGVudChodG1sX2U3ZmU1NjMxZTVkNjQyMjM5NjU5NjhhYWExY2ExNmUxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2QyM2Y2NmZjMjA5MDRlYmM4ZWM0MjA0NjU0ZWJlMTkyLmJpbmRQb3B1cChwb3B1cF9mMDhkNWEwNzY3Nzk0NWYwYTg0MmNhODJjODM5MzhkMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80MzBjMzY4MjU3N2M0NDc3YWU2MzBkMDI2MzQ0OTgwOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODk1OTcsLTc5LjQ1NjMyNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jMzc0NzI0ZGY4ZDA0YTgwOTQ0OWViMDgzYTE0Y2YxNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xYzEwMDRkMTYyMmU0ZTI0ODBkNzc0N2ViZDRkOWI2NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jNDVkMjg2MzRhYTQ0YmE4OTM1MjBmMjAwOTNkYWFkOCA9ICQoJzxkaXYgaWQ9Imh0bWxfYzQ1ZDI4NjM0YWE0NGJhODkzNTIwZjIwMDkzZGFhZDgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3QgVG9yb250byAoTTZSKTogUGFya2RhbGUsIFJvbmNlc3ZhbGxlcyAtIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMWMxMDA0ZDE2MjJlNGUyNDgwZDc3NDdlYmQ0ZDliNjYuc2V0Q29udGVudChodG1sX2M0NWQyODYzNGFhNDRiYTg5MzUyMGYyMDA5M2RhYWQ4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQzMGMzNjgyNTc3YzQ0NzdhZTYzMGQwMjYzNDQ5ODA4LmJpbmRQb3B1cChwb3B1cF8xYzEwMDRkMTYyMmU0ZTI0ODBkNzc0N2ViZDRkOWI2Nik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lOWNiZDA5NzgyMmQ0YjY0ODYxMjI2NThkN2QzMTQyMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MTU3MDYsLTc5LjQ4NDQ0OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYzM3NDcyNGRmOGQwNGE4MDk0NDllYjA4M2ExNGNmMTcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjVkZTE4MGQwNmIwNDlmOTliZjU5MDRjOGJkNDY5YzggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDI0MjhkNWEwM2U3NDMxNzg3YzViYjhjNDdhZjhkOGIgPSAkKCc8ZGl2IGlkPSJodG1sXzAyNDI4ZDVhMDNlNzQzMTc4N2M1YmI4YzQ3YWY4ZDhiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XZXN0IFRvcm9udG8gKE02Uyk6IFJ1bm55bWVkZSwgU3dhbnNlYSAtIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjVkZTE4MGQwNmIwNDlmOTliZjU5MDRjOGJkNDY5Yzguc2V0Q29udGVudChodG1sXzAyNDI4ZDVhMDNlNzQzMTc4N2M1YmI4YzQ3YWY4ZDhiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U5Y2JkMDk3ODIyZDRiNjQ4NjEyMjY1OGQ3ZDMxNDIxLmJpbmRQb3B1cChwb3B1cF9iNWRlMTgwZDA2YjA0OWY5OWJmNTkwNGM4YmQ0NjljOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xMGFiYzVhNzI1ZDg0ZDFiYTA1MzBkNDBmNmI4YWU3OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODE5ODUsLTc5LjM3OTgxNjkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2MzNzQ3MjRkZjhkMDRhODA5NDQ5ZWIwODNhMTRjZjE3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JmZWZhM2EzMmVhOTQ3Mzk5OWYwZTZlNzI3ODBiMzliID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2VlM2U3ZTNhYzgwYTRhNjRhYTFlMWQ2ZDdkNDdkOWM0ID0gJCgnPGRpdiBpZD0iaHRtbF9lZTNlN2UzYWM4MGE0YTY0YWExZTFkNmQ3ZDQ3ZDljNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250byAoTTVMKTogQ29tbWVyY2UgQ291cnQsIFZpY3RvcmlhIEhvdGVsIC0gQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iZmVmYTNhMzJlYTk0NzM5OTlmMGU2ZTcyNzgwYjM5Yi5zZXRDb250ZW50KGh0bWxfZWUzZTdlM2FjODBhNGE2NGFhMWUxZDZkN2Q0N2Q5YzQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTBhYmM1YTcyNWQ4NGQxYmEwNTMwZDQwZjZiOGFlNzkuYmluZFBvcHVwKHBvcHVwX2JmZWZhM2EzMmVhOTQ3Mzk5OWYwZTZlNzI3ODBiMzliKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzIzMjI0N2VmZDExZTRlNWNiOGQ3ZjdiNTRhYzY3MWY0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ3MTc2OCwtNzkuMzgxNTc2NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYzM3NDcyNGRmOGQwNGE4MDk0NDllYjA4M2ExNGNmMTcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTgzZTQyNWRhNGVjNGY0YWE0MzEwYzA2ZGEzYzNhNTcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjlkN2IzYjIyMTU3NDVlMWE5MzM5NWRiODQ5MGRlMzYgPSAkKCc8ZGl2IGlkPSJodG1sX2I5ZDdiM2IyMjE1NzQ1ZTFhOTMzOTVkYjg0OTBkZTM2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvIChNNUspOiBEZXNpZ24gRXhjaGFuZ2UsIFRvcm9udG8gRG9taW5pb24gQ2VudHJlIC0gQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85ODNlNDI1ZGE0ZWM0ZjRhYTQzMTBjMDZkYTNjM2E1Ny5zZXRDb250ZW50KGh0bWxfYjlkN2IzYjIyMTU3NDVlMWE5MzM5NWRiODQ5MGRlMzYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjMyMjQ3ZWZkMTFlNGU1Y2I4ZDdmN2I1NGFjNjcxZjQuYmluZFBvcHVwKHBvcHVwXzk4M2U0MjVkYTRlYzRmNGFhNDMxMGMwNmRhM2MzYTU3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzg1YTJjMjAzMDY1YTRiY2ZhOTFiMGFkMDZhZDA4ODk2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNzQzOSwtNzkuMzIxNTU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2MzNzQ3MjRkZjhkMDRhODA5NDQ5ZWIwODNhMTRjZjE3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzAxN2JmN2MyMGMxNzRhZmRhNDFlOWRkNzU3ZjJkMjcxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhiZTViMzI4ZTUxNDRlYmZiYzg3MDZhNDY0YjNhM2E1ID0gJCgnPGRpdiBpZD0iaHRtbF84YmU1YjMyOGU1MTQ0ZWJmYmM4NzA2YTQ2NGIzYTNhNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RWFzdCBUb3JvbnRvIChNN1kpOiBCdXNpbmVzcyBSZXBseSBNYWlsIFByb2Nlc3NpbmcgQ2VudHJlIDk2OSBFYXN0ZXJuIC0gQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wMTdiZjdjMjBjMTc0YWZkYTQxZTlkZDc1N2YyZDI3MS5zZXRDb250ZW50KGh0bWxfOGJlNWIzMjhlNTE0NGViZmJjODcwNmE0NjRiM2EzYTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODVhMmMyMDMwNjVhNGJjZmE5MWIwYWQwNmFkMDg4OTYuYmluZFBvcHVwKHBvcHVwXzAxN2JmN2MyMGMxNzRhZmRhNDFlOWRkNzU3ZjJkMjcxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzk3NDE1YmRlMzgwMzQ2ODVhYzU5Y2UzZGQ4NTUyZDQ2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUwNTcxMjAwMDAwMDEsLTc5LjM4NDU2NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYzM3NDcyNGRmOGQwNGE4MDk0NDllYjA4M2ExNGNmMTcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMGYwYjVhNjkyMTVjNDgxNGFmZDA5NGVlZWJkMDk2ZjEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNGI5YTIwNWI4ZTRkNGI3MmJiZGM0YWQ0ZWZiMDc0NzkgPSAkKCc8ZGl2IGlkPSJodG1sXzRiOWEyMDViOGU0ZDRiNzJiYmRjNGFkNGVmYjA3NDc5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvIChNNUgpOiBBZGVsYWlkZSwgS2luZywgUmljaG1vbmQgLSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzBmMGI1YTY5MjE1YzQ4MTRhZmQwOTRlZWViZDA5NmYxLnNldENvbnRlbnQoaHRtbF80YjlhMjA1YjhlNGQ0YjcyYmJkYzRhZDRlZmIwNzQ3OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85NzQxNWJkZTM4MDM0Njg1YWM1OWNlM2RkODU1MmQ0Ni5iaW5kUG9wdXAocG9wdXBfMGYwYjVhNjkyMTVjNDgxNGFmZDA5NGVlZWJkMDk2ZjEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGEzYWE0ZmIzNzhiNDRkNGE3YTNiYTY0OTUwYzk5ZDUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NTcxLC03OS4zNTIxODhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYzM3NDcyNGRmOGQwNGE4MDk0NDllYjA4M2ExNGNmMTcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWRjNGFmMDFkZTBkNGJlOTgzODZiMmQyNDE5NGI5MmQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2ZkYWVjZjhjMjhmNGE4NGIyOWMzZDhkNjg4MTkyYjYgPSAkKCc8ZGl2IGlkPSJodG1sXzdmZGFlY2Y4YzI4ZjRhODRiMjljM2Q4ZDY4ODE5MmI2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FYXN0IFRvcm9udG8gKE00Syk6IFRoZSBEYW5mb3J0aCBXZXN0LCBSaXZlcmRhbGUgLSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFkYzRhZjAxZGUwZDRiZTk4Mzg2YjJkMjQxOTRiOTJkLnNldENvbnRlbnQoaHRtbF83ZmRhZWNmOGMyOGY0YTg0YjI5YzNkOGQ2ODgxOTJiNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kYTNhYTRmYjM3OGI0NGQ0YTdhM2JhNjQ5NTBjOTlkNS5iaW5kUG9wdXAocG9wdXBfMWRjNGFmMDFkZTBkNGJlOTgzODZiMmQyNDE5NGI5MmQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODM4YmVmMDVmODc2NDRkNmE3YzRlNmNjN2ZiYWQwZDQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njg5OTg1LC03OS4zMTU1NzE1OTk5OTk5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jMzc0NzI0ZGY4ZDA0YTgwOTQ0OWViMDgzYTE0Y2YxNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wNTE0NDk3MDIwNDM0ZGI1YjdmZTIzNTE5YzQzYjAyMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80M2U1Zjc1MzdkYzE0ZTlmYTg1OWYyOTZiYjY3M2JhZSA9ICQoJzxkaXYgaWQ9Imh0bWxfNDNlNWY3NTM3ZGMxNGU5ZmE4NTlmMjk2YmI2NzNiYWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkVhc3QgVG9yb250byAoTTRMKTogVGhlIEJlYWNoZXMgV2VzdCwgSW5kaWEgQmF6YWFyIC0gQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wNTE0NDk3MDIwNDM0ZGI1YjdmZTIzNTE5YzQzYjAyMi5zZXRDb250ZW50KGh0bWxfNDNlNWY3NTM3ZGMxNGU5ZmE4NTlmMjk2YmI2NzNiYWUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODM4YmVmMDVmODc2NDRkNmE3YzRlNmNjN2ZiYWQwZDQuYmluZFBvcHVwKHBvcHVwXzA1MTQ0OTcwMjA0MzRkYjViN2ZlMjM1MTljNDNiMDIyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU0ZTFiYjM0YmQxYzQxYTE5NmFjNjA0Mjk0Mjk2OWFlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU5NTI1NSwtNzkuMzQwOTIzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2MzNzQ3MjRkZjhkMDRhODA5NDQ5ZWIwODNhMTRjZjE3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2UyZTM3M2Y4NTc5NzRjODFiNmQ3MWU0MmQwZThlMWQ5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc5ZDEzYjVhNzNmYTQ1ZGY5ZGEzMzcwNTE2NjIyMzEyID0gJCgnPGRpdiBpZD0iaHRtbF83OWQxM2I1YTczZmE0NWRmOWRhMzM3MDUxNjYyMjMxMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RWFzdCBUb3JvbnRvIChNNE0pOiBTdHVkaW8gRGlzdHJpY3QgLSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2UyZTM3M2Y4NTc5NzRjODFiNmQ3MWU0MmQwZThlMWQ5LnNldENvbnRlbnQoaHRtbF83OWQxM2I1YTczZmE0NWRmOWRhMzM3MDUxNjYyMjMxMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81NGUxYmIzNGJkMWM0MWExOTZhYzYwNDI5NDI5NjlhZS5iaW5kUG9wdXAocG9wdXBfZTJlMzczZjg1Nzk3NGM4MWI2ZDcxZTQyZDBlOGUxZDkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzhhYTUwYTBkN2ViNDBjNWI4ODc5MWQ3MDU0OGE5MmQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTI3NTExLC03OS4zOTAxOTc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2MzNzQ3MjRkZjhkMDRhODA5NDQ5ZWIwODNhMTRjZjE3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzIxMGU0MjUzYTJlODRkYjc4MzZjMzMyODVkZjBiZTQ5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzlhMGM3YjlkNWRjNDRlMTZhZWQ3MWFiMzdkYTdhYjI5ID0gJCgnPGRpdiBpZD0iaHRtbF85YTBjN2I5ZDVkYzQ0ZTE2YWVkNzFhYjM3ZGE3YWIyOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBUb3JvbnRvIChNNFApOiBEYXZpc3ZpbGxlIE5vcnRoIC0gQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yMTBlNDI1M2EyZTg0ZGI3ODM2YzMzMjg1ZGYwYmU0OS5zZXRDb250ZW50KGh0bWxfOWEwYzdiOWQ1ZGM0NGUxNmFlZDcxYWIzN2RhN2FiMjkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzhhYTUwYTBkN2ViNDBjNWI4ODc5MWQ3MDU0OGE5MmQuYmluZFBvcHVwKHBvcHVwXzIxMGU0MjUzYTJlODRkYjc4MzZjMzMyODVkZjBiZTQ5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzdmNWQ3ZDg5MGUzNTRmNmQ5NmU5NzJhMmUzZTNmZWJmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzE1MzgzNCwtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYzM3NDcyNGRmOGQwNGE4MDk0NDllYjA4M2ExNGNmMTcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjJhMDJkMGRjNGRmNDk4M2JjNjFhZTM4YjAxOGRjMzEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTAzNWE2YjI5MWZjNDk2YjhhZGU0YjE3YTYxZmQxYzEgPSAkKCc8ZGl2IGlkPSJodG1sXzEwMzVhNmIyOTFmYzQ5NmI4YWRlNGIxN2E2MWZkMWMxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZW50cmFsIFRvcm9udG8gKE00Uik6IE5vcnRoIFRvcm9udG8gV2VzdCAtIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjJhMDJkMGRjNGRmNDk4M2JjNjFhZTM4YjAxOGRjMzEuc2V0Q29udGVudChodG1sXzEwMzVhNmIyOTFmYzQ5NmI4YWRlNGIxN2E2MWZkMWMxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdmNWQ3ZDg5MGUzNTRmNmQ5NmU5NzJhMmUzZTNmZWJmLmJpbmRQb3B1cChwb3B1cF8yMmEwMmQwZGM0ZGY0OTgzYmM2MWFlMzhiMDE4ZGMzMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iM2U3NjIzZWViNjU0OGMyODNhOGM1ZjA0NDUwYjkxNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNDMyNDQsLTc5LjM4ODc5MDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYzM3NDcyNGRmOGQwNGE4MDk0NDllYjA4M2ExNGNmMTcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDMyZmY2MTFlOGNlNDEwYTk4MTQxOGQxOGEwY2I4YWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDgzNmE1MWIwMzE0NGVjNWJiNjg5YzFmNTIzMDhmYzggPSAkKCc8ZGl2IGlkPSJodG1sXzQ4MzZhNTFiMDMxNDRlYzViYjY4OWMxZjUyMzA4ZmM4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZW50cmFsIFRvcm9udG8gKE00Uyk6IERhdmlzdmlsbGUgLSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQzMmZmNjExZThjZTQxMGE5ODE0MThkMThhMGNiOGFiLnNldENvbnRlbnQoaHRtbF80ODM2YTUxYjAzMTQ0ZWM1YmI2ODljMWY1MjMwOGZjOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iM2U3NjIzZWViNjU0OGMyODNhOGM1ZjA0NDUwYjkxNy5iaW5kUG9wdXAocG9wdXBfNDMyZmY2MTFlOGNlNDEwYTk4MTQxOGQxOGEwY2I4YWIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzhiMTg0ZDQ1ZTk0NGQwM2JhYWIxM2YzNGJhY2UwNGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODY0MTIyOTk5OTk5OSwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jMzc0NzI0ZGY4ZDA0YTgwOTQ0OWViMDgzYTE0Y2YxNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80OGU4NGEyOTNhOTY0NDE2OWFiNjI2ZTU1OTIwNDk5ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82MzhlNjk5ODUwZjA0NThkOGUxYTUxY2ZhNzU4MTQ0YyA9ICQoJzxkaXYgaWQ9Imh0bWxfNjM4ZTY5OTg1MGYwNDU4ZDhlMWE1MWNmYTc1ODE0NGMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgVG9yb250byAoTTRWKTogRGVlciBQYXJrLCBGb3Jlc3QgSGlsbCBTRSwgUmF0aG5lbGx5LCBTb3V0aCBIaWxsLCBTdW1tZXJoaWxsIFdlc3QgLSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ4ZTg0YTI5M2E5NjQ0MTY5YWI2MjZlNTU5MjA0OTlkLnNldENvbnRlbnQoaHRtbF82MzhlNjk5ODUwZjA0NThkOGUxYTUxY2ZhNzU4MTQ0Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jOGIxODRkNDVlOTQ0ZDAzYmFhYjEzZjM0YmFjZTA0Zi5iaW5kUG9wdXAocG9wdXBfNDhlODRhMjkzYTk2NDQxNjlhYjYyNmU1NTkyMDQ5OWQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYmQ5Njk0YTdjNmY4NDUzMDk4MjRjMjJiMjA4MjZkNWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjIzMDE1LC03OS4zODk0OTM4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2MzNzQ3MjRkZjhkMDRhODA5NDQ5ZWIwODNhMTRjZjE3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzJhMWZjMTllOGM3ZTQ4M2NhOGNmMDg4N2YyNGE5NzU5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2M0ZTExMjE4NWZmNjRlYmI5YjFlNGQ0ZDc5ZmJmZjY4ID0gJCgnPGRpdiBpZD0iaHRtbF9jNGUxMTIxODVmZjY0ZWJiOWIxZTRkNGQ3OWZiZmY2OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250byAoTTdBKTogUXVlZW4mIzM5O3MgUGFyayAtIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMmExZmMxOWU4YzdlNDgzY2E4Y2YwODg3ZjI0YTk3NTkuc2V0Q29udGVudChodG1sX2M0ZTExMjE4NWZmNjRlYmI5YjFlNGQ0ZDc5ZmJmZjY4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2JkOTY5NGE3YzZmODQ1MzA5ODI0YzIyYjIwODI2ZDVkLmJpbmRQb3B1cChwb3B1cF8yYTFmYzE5ZThjN2U0ODNjYThjZjA4ODdmMjRhOTc1OSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84OGQzNjE0YjI4OTM0ZTkwOTcxY2VkYjU4NmM5N2YxNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzk2NywtNzkuMzY3Njc1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jMzc0NzI0ZGY4ZDA0YTgwOTQ0OWViMDgzYTE0Y2YxNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lYjk3NDRhNDM3Yjg0NjUwOWUxYWFiMDNlZGMyNjIxZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82YjgzZGRjYWE2NTk0ZGVjODY0YTkxMTY3Y2U4N2ZmZSA9ICQoJzxkaXYgaWQ9Imh0bWxfNmI4M2RkY2FhNjU5NGRlYzg2NGE5MTE2N2NlODdmZmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG8gKE00WCk6IENhYmJhZ2V0b3duLCBTdC4gSmFtZXMgVG93biAtIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZWI5NzQ0YTQzN2I4NDY1MDllMWFhYjAzZWRjMjYyMWYuc2V0Q29udGVudChodG1sXzZiODNkZGNhYTY1OTRkZWM4NjRhOTExNjdjZTg3ZmZlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg4ZDM2MTRiMjg5MzRlOTA5NzFjZWRiNTg2Yzk3ZjE3LmJpbmRQb3B1cChwb3B1cF9lYjk3NDRhNDM3Yjg0NjUwOWUxYWFiMDNlZGMyNjIxZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yOTQ1ZmM4Mzc4ZjY0ZWMwOTZmMzdjYTU3Yzg0NDE5MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1NDI1OTksLTc5LjM2MDYzNTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYzM3NDcyNGRmOGQwNGE4MDk0NDllYjA4M2ExNGNmMTcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTBjZjZiODM4MDFkNDljZTk3ZDliNzcwNzJmNDQzZWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODk3NjIwN2IyMmJmNDg5MjlhNWFiN2E1NTlkNTI0M2MgPSAkKCc8ZGl2IGlkPSJodG1sXzg5NzYyMDdiMjJiZjQ4OTI5YTVhYjdhNTU5ZDUyNDNjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvIChNNUEpOiBIYXJib3VyZnJvbnQgLSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzEwY2Y2YjgzODAxZDQ5Y2U5N2Q5Yjc3MDcyZjQ0M2VlLnNldENvbnRlbnQoaHRtbF84OTc2MjA3YjIyYmY0ODkyOWE1YWI3YTU1OWQ1MjQzYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yOTQ1ZmM4Mzc4ZjY0ZWMwOTZmMzdjYTU3Yzg0NDE5MC5iaW5kUG9wdXAocG9wdXBfMTBjZjZiODM4MDFkNDljZTk3ZDliNzcwNzJmNDQzZWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTFmNGQwNTRhZGIzNDdkNWIyYzhiNTk5MDI4M2Y2ODkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTcxNjE4LC03OS4zNzg5MzcwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jMzc0NzI0ZGY4ZDA0YTgwOTQ0OWViMDgzYTE0Y2YxNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84YzZmNjlhYjBmNzc0MDI3YjVlNzM4YmE1MzJkMjNjOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lNDg2NGI3MzZhMjI0NTZjOWM1NGJlNDJlYzYzMDBkZSA9ICQoJzxkaXYgaWQ9Imh0bWxfZTQ4NjRiNzM2YTIyNDU2YzljNTRiZTQyZWM2MzAwZGUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG8gKE01Qik6IFJ5ZXJzb24sIEdhcmRlbiBEaXN0cmljdCAtIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOGM2ZjY5YWIwZjc3NDAyN2I1ZTczOGJhNTMyZDIzYzguc2V0Q29udGVudChodG1sX2U0ODY0YjczNmEyMjQ1NmM5YzU0YmU0MmVjNjMwMGRlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzExZjRkMDU0YWRiMzQ3ZDViMmM4YjU5OTAyODNmNjg5LmJpbmRQb3B1cChwb3B1cF84YzZmNjlhYjBmNzc0MDI3YjVlNzM4YmE1MzJkMjNjOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iNmQxZGIyMzIxNTU0MzE4YmU3MmIxM2JlODY4ZjcxZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MTQ5MzksLTc5LjM3NTQxNzldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYzM3NDcyNGRmOGQwNGE4MDk0NDllYjA4M2ExNGNmMTcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTZjMTIzNGY4MmU4NDJiYjgzZmRjMzc5YWJiOTJkODggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmQyODUzYTk4OTM0NGVhZmI0NzlhMWQwNjk2ODk3ZjggPSAkKCc8ZGl2IGlkPSJodG1sX2JkMjg1M2E5ODkzNDRlYWZiNDc5YTFkMDY5Njg5N2Y4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3dudG93biBUb3JvbnRvIChNNUMpOiBTdC4gSmFtZXMgVG93biAtIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTZjMTIzNGY4MmU4NDJiYjgzZmRjMzc5YWJiOTJkODguc2V0Q29udGVudChodG1sX2JkMjg1M2E5ODkzNDRlYWZiNDc5YTFkMDY5Njg5N2Y4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I2ZDFkYjIzMjE1NTQzMThiZTcyYjEzYmU4NjhmNzFmLmJpbmRQb3B1cChwb3B1cF9lNmMxMjM0ZjgyZTg0MmJiODNmZGMzNzlhYmI5MmQ4OCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yZjBjYmQ3MWI4MWU0MWY3YWE1YmU3MjRmMTAwZTQ2ZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NDc3MDc5OTk5OTk5NiwtNzkuMzczMzA2NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jMzc0NzI0ZGY4ZDA0YTgwOTQ0OWViMDgzYTE0Y2YxNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mZDE1ODkyMjkyMTQ0YjcxOGVlODI5Yjk5ZWI2YTM0YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iYTgwZDhiYmIzOGU0YWE2YjYwYzc5MzdkNjI1ZmQ4ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfYmE4MGQ4YmJiMzhlNGFhNmI2MGM3OTM3ZDYyNWZkOGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG8gKE01RSk6IEJlcmN6eSBQYXJrIC0gQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mZDE1ODkyMjkyMTQ0YjcxOGVlODI5Yjk5ZWI2YTM0YS5zZXRDb250ZW50KGh0bWxfYmE4MGQ4YmJiMzhlNGFhNmI2MGM3OTM3ZDYyNWZkOGQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmYwY2JkNzFiODFlNDFmN2FhNWJlNzI0ZjEwMGU0NmYuYmluZFBvcHVwKHBvcHVwX2ZkMTU4OTIyOTIxNDRiNzE4ZWU4MjliOTllYjZhMzRhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzdhYzMyMjJjMWVhNzRmMTBhM2VlMzNkZGM3OTBiYzAyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3OTUyNCwtNzkuMzg3MzgyNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jMzc0NzI0ZGY4ZDA0YTgwOTQ0OWViMDgzYTE0Y2YxNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xYWM5ODhiZWIwNTA0MTZjYjJlZmU5M2Y3OTI3OGM2OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kZGQzZGI2MzE2OWI0NTBhOWMxZmNlMjgyMTZhNjZkZCA9ICQoJzxkaXYgaWQ9Imh0bWxfZGRkM2RiNjMxNjliNDUwYTljMWZjZTI4MjE2YTY2ZGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd250b3duIFRvcm9udG8gKE01Ryk6IENlbnRyYWwgQmF5IFN0cmVldCAtIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMWFjOTg4YmViMDUwNDE2Y2IyZWZlOTNmNzkyNzhjNjguc2V0Q29udGVudChodG1sX2RkZDNkYjYzMTY5YjQ1MGE5YzFmY2UyODIxNmE2NmRkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdhYzMyMjJjMWVhNzRmMTBhM2VlMzNkZGM3OTBiYzAyLmJpbmRQb3B1cChwb3B1cF8xYWM5ODhiZWIwNTA0MTZjYjJlZmU5M2Y3OTI3OGM2OCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jY2MzYWZlODA0ZDc0MTJjYWM2YzcyMGViMjkyZTZkMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2NTg1OTksLTc5LjM4MzE1OTkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2MzNzQ3MjRkZjhkMDRhODA5NDQ5ZWIwODNhMTRjZjE3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2EyNTUwNTU5YWQ3ZjQyYjc5YjU0ODllOGVlOWVkMjczID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJhOWZiNmJmODRmYjRiMjI4OGU5YzU5YmQyNzhiY2UzID0gJCgnPGRpdiBpZD0iaHRtbF8yYTlmYjZiZjg0ZmI0YjIyODhlOWM1OWJkMjc4YmNlMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnRvd24gVG9yb250byAoTTRZKTogQ2h1cmNoIGFuZCBXZWxsZXNsZXkgLSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2EyNTUwNTU5YWQ3ZjQyYjc5YjU0ODllOGVlOWVkMjczLnNldENvbnRlbnQoaHRtbF8yYTlmYjZiZjg0ZmI0YjIyODhlOWM1OWJkMjc4YmNlMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jY2MzYWZlODA0ZDc0MTJjYWM2YzcyMGViMjkyZTZkMS5iaW5kUG9wdXAocG9wdXBfYTI1NTA1NTlhZDdmNDJiNzliNTQ4OWU4ZWU5ZWQyNzMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjFlOTNhOThmNmNjNGFkODlhZGI0Y2JlOTE1ZGVlY2UgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTE2OTQ4LC03OS40MTY5MzU1OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjMDBiNWViIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzAwYjVlYiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jMzc0NzI0ZGY4ZDA0YTgwOTQ0OWViMDgzYTE0Y2YxNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kZDVhMDI5ZjQ2NTM0ODViOWQ1NThlOWM1YzI1MTczMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82OTc1ZTRlMTNhYjE0MTA2YjlkMzA4MmQxMWZhZWM1NyA9ICQoJzxkaXYgaWQ9Imh0bWxfNjk3NWU0ZTEzYWIxNDEwNmI5ZDMwODJkMTFmYWVjNTciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgVG9yb250byAoTTVOKTogUm9zZWxhd24gLSBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RkNWEwMjlmNDY1MzQ4NWI5ZDU1OGU5YzVjMjUxNzMzLnNldENvbnRlbnQoaHRtbF82OTc1ZTRlMTNhYjE0MTA2YjlkMzA4MmQxMWZhZWM1Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yMWU5M2E5OGY2Y2M0YWQ4OWFkYjRjYmU5MTVkZWVjZS5iaW5kUG9wdXAocG9wdXBfZGQ1YTAyOWY0NjUzNDg1YjlkNTU4ZTljNWMyNTE3MzMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTMwZTk1MDg3MTdlNGMzYTllMGQxNjMxMDJlOGY0M2EgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzYzNTczOTk5OTk5OSwtNzkuMjkzMDMxMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jMzc0NzI0ZGY4ZDA0YTgwOTQ0OWViMDgzYTE0Y2YxNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yZTUyYjE4N2FkM2M0NTBiYTU5YTFlNGU2NmZlNmU2ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81NDMxMjJjZDk4OTE0YTVlYmM3N2NlYjEwMzBhNDAxYiA9ICQoJzxkaXYgaWQ9Imh0bWxfNTQzMTIyY2Q5ODkxNGE1ZWJjNzdjZWIxMDMwYTQwMWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkVhc3QgVG9yb250byAoTTRFKTogVGhlIEJlYWNoZXMgLSBDbHVzdGVyIDM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzJlNTJiMTg3YWQzYzQ1MGJhNTlhMWU0ZTY2ZmU2ZTZmLnNldENvbnRlbnQoaHRtbF81NDMxMjJjZDk4OTE0YTVlYmM3N2NlYjEwMzBhNDAxYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMzBlOTUwODcxN2U0YzNhOWUwZDE2MzEwMmU4ZjQzYS5iaW5kUG9wdXAocG9wdXBfMmU1MmIxODdhZDNjNDUwYmE1OWExZTRlNjZmZTZlNmYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzU1ZjA2NzRkYmNjNGU4MWJmMjljMjIwMWM5NWU4MmUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTY5NDc2LC03OS40MTEzMDcyMDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9jMzc0NzI0ZGY4ZDA0YTgwOTQ0OWViMDgzYTE0Y2YxNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81MDg2ZmQ1MzdlNmY0NWM2YTFjYTI4YTc0ZTQ4M2U2NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yYjgzYTdiNjk4MWI0MTM0YmZjY2Y3OWViOGJhNGVmMyA9ICQoJzxkaXYgaWQ9Imh0bWxfMmI4M2E3YjY5ODFiNDEzNGJmY2NmNzllYjhiYTRlZjMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgVG9yb250byAoTTVQKTogRm9yZXN0IEhpbGwgTm9ydGgsIEZvcmVzdCBIaWxsIFdlc3QgLSBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzUwODZmZDUzN2U2ZjQ1YzZhMWNhMjhhNzRlNDgzZTY2LnNldENvbnRlbnQoaHRtbF8yYjgzYTdiNjk4MWI0MTM0YmZjY2Y3OWViOGJhNGVmMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jNTVmMDY3NGRiY2M0ZTgxYmYyOWMyMjAxYzk1ZTgyZS5iaW5kUG9wdXAocG9wdXBfNTA4NmZkNTM3ZTZmNDVjNmExY2EyOGE3NGU0ODNlNjYpOwoKICAgICAgICAgICAgCiAgICAgICAgCjwvc2NyaXB0Pg== onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### Examine Clusters

#### Cluster 1


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Central Toronto</td>
      <td>0</td>
      <td>Photography Studio</td>
      <td>Construction &amp; Landscaping</td>
      <td>Park</td>
      <td>Bus Line</td>
      <td>Swim School</td>
      <td>Diner</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Central Toronto</td>
      <td>0</td>
      <td>Tennis Court</td>
      <td>Playground</td>
      <td>Park</td>
      <td>Restaurant</td>
      <td>Doner Restaurant</td>
      <td>Dim Sum Restaurant</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Dog Run</td>
      <td>Dumpling Restaurant</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Downtown Toronto</td>
      <td>0</td>
      <td>Park</td>
      <td>Playground</td>
      <td>Trail</td>
      <td>Yoga Studio</td>
      <td>Dumpling Restaurant</td>
      <td>Diner</td>
      <td>Discount Store</td>
      <td>Dog Run</td>
      <td>Doner Restaurant</td>
      <td>Donut Shop</td>
    </tr>
  </tbody>
</table>
</div>



#### Cluster 2


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Aquarium</td>
      <td>Hotel</td>
      <td>Italian Restaurant</td>
      <td>Café</td>
      <td>Brewery</td>
      <td>Restaurant</td>
      <td>Scenic Lookout</td>
      <td>Fried Chicken Joint</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Central Toronto</td>
      <td>1</td>
      <td>Sandwich Place</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Liquor Store</td>
      <td>BBQ Joint</td>
      <td>Pharmacy</td>
      <td>Cosmetics Shop</td>
      <td>Pizza Place</td>
      <td>Pub</td>
      <td>Middle Eastern Restaurant</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Café</td>
      <td>Bakery</td>
      <td>Bookstore</td>
      <td>Japanese Restaurant</td>
      <td>Restaurant</td>
      <td>Bar</td>
      <td>Yoga Studio</td>
      <td>Chinese Restaurant</td>
      <td>Beer Store</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Bar</td>
      <td>Café</td>
      <td>Vietnamese Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Coffee Shop</td>
      <td>Dumpling Restaurant</td>
      <td>Bakery</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Mexican Restaurant</td>
      <td>Farmers Market</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Airport Service</td>
      <td>Airport Lounge</td>
      <td>Airport Terminal</td>
      <td>Plane</td>
      <td>Sculpture Garden</td>
      <td>Coffee Shop</td>
      <td>Boat or Ferry</td>
      <td>Harbor / Marina</td>
      <td>Boutique</td>
      <td>Airport Gate</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Seafood Restaurant</td>
      <td>Hotel</td>
      <td>Beer Bar</td>
      <td>Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Farmers Market</td>
      <td>Creperie</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Burger Joint</td>
      <td>Deli / Bodega</td>
      <td>Gym</td>
      <td>Steakhouse</td>
      <td>American Restaurant</td>
      <td>Bakery</td>
      <td>Seafood Restaurant</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Grocery Store</td>
      <td>Café</td>
      <td>Park</td>
      <td>Baby Store</td>
      <td>Gas Station</td>
      <td>Italian Restaurant</td>
      <td>Restaurant</td>
      <td>Candy Store</td>
      <td>Nightclub</td>
      <td>Athletics &amp; Sports</td>
    </tr>
    <tr>
      <th>31</th>
      <td>West Toronto</td>
      <td>1</td>
      <td>Bakery</td>
      <td>Pharmacy</td>
      <td>Brewery</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Bank</td>
      <td>Middle Eastern Restaurant</td>
      <td>Portuguese Restaurant</td>
      <td>Pool</td>
      <td>Music Venue</td>
    </tr>
    <tr>
      <th>32</th>
      <td>West Toronto</td>
      <td>1</td>
      <td>Bar</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Asian Restaurant</td>
      <td>Vietnamese Restaurant</td>
      <td>Pizza Place</td>
      <td>Café</td>
      <td>Men's Store</td>
      <td>Yoga Studio</td>
      <td>Record Shop</td>
    </tr>
    <tr>
      <th>33</th>
      <td>West Toronto</td>
      <td>1</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Breakfast Spot</td>
      <td>Grocery Store</td>
      <td>Intersection</td>
      <td>Performing Arts Venue</td>
      <td>Pet Store</td>
      <td>Gym</td>
      <td>Nightclub</td>
      <td>Climbing Gym</td>
    </tr>
    <tr>
      <th>34</th>
      <td>West Toronto</td>
      <td>1</td>
      <td>Café</td>
      <td>Mexican Restaurant</td>
      <td>Bar</td>
      <td>Thai Restaurant</td>
      <td>Grocery Store</td>
      <td>Furniture / Home Store</td>
      <td>Music Venue</td>
      <td>Diner</td>
      <td>Cajun / Creole Restaurant</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>35</th>
      <td>West Toronto</td>
      <td>1</td>
      <td>Gift Shop</td>
      <td>Italian Restaurant</td>
      <td>Dessert Shop</td>
      <td>Bookstore</td>
      <td>Eastern European Restaurant</td>
      <td>Movie Theater</td>
      <td>Bar</td>
      <td>Bank</td>
      <td>Dog Run</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>36</th>
      <td>West Toronto</td>
      <td>1</td>
      <td>Pizza Place</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Sushi Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Yoga Studio</td>
      <td>Gastropub</td>
      <td>Restaurant</td>
      <td>Pub</td>
      <td>Latin American Restaurant</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Restaurant</td>
      <td>Gym</td>
      <td>Deli / Bodega</td>
      <td>American Restaurant</td>
      <td>Bakery</td>
      <td>Steakhouse</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Hotel</td>
      <td>Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Bakery</td>
      <td>Steakhouse</td>
      <td>Seafood Restaurant</td>
      <td>American Restaurant</td>
      <td>Bar</td>
    </tr>
    <tr>
      <th>38</th>
      <td>East Toronto</td>
      <td>1</td>
      <td>Pizza Place</td>
      <td>Auto Workshop</td>
      <td>Moving Target</td>
      <td>Restaurant</td>
      <td>Butcher</td>
      <td>Burrito Place</td>
      <td>Skate Park</td>
      <td>Brewery</td>
      <td>Farmers Market</td>
      <td>Comic Shop</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Bar</td>
      <td>Thai Restaurant</td>
      <td>Cosmetics Shop</td>
      <td>Burger Joint</td>
      <td>Restaurant</td>
      <td>Steakhouse</td>
      <td>Bakery</td>
      <td>Sushi Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>East Toronto</td>
      <td>1</td>
      <td>Greek Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Bookstore</td>
      <td>Furniture / Home Store</td>
      <td>Ice Cream Shop</td>
      <td>Fruit &amp; Vegetable Store</td>
      <td>Brewery</td>
      <td>Bubble Tea Shop</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>2</th>
      <td>East Toronto</td>
      <td>1</td>
      <td>Sandwich Place</td>
      <td>Pizza Place</td>
      <td>Park</td>
      <td>Ice Cream Shop</td>
      <td>Burger Joint</td>
      <td>Burrito Place</td>
      <td>Liquor Store</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Steakhouse</td>
      <td>Food &amp; Drink Shop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>East Toronto</td>
      <td>1</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Brewery</td>
      <td>Gastropub</td>
      <td>American Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Middle Eastern Restaurant</td>
      <td>Bar</td>
      <td>Clothing Store</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Central Toronto</td>
      <td>1</td>
      <td>Breakfast Spot</td>
      <td>Park</td>
      <td>Department Store</td>
      <td>Food &amp; Drink Shop</td>
      <td>Hotel</td>
      <td>Sandwich Place</td>
      <td>Gym</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Central Toronto</td>
      <td>1</td>
      <td>Clothing Store</td>
      <td>Coffee Shop</td>
      <td>Yoga Studio</td>
      <td>Bagel Shop</td>
      <td>Pet Store</td>
      <td>Chinese Restaurant</td>
      <td>Dessert Shop</td>
      <td>Restaurant</td>
      <td>Diner</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Central Toronto</td>
      <td>1</td>
      <td>Sandwich Place</td>
      <td>Pizza Place</td>
      <td>Dessert Shop</td>
      <td>Gym</td>
      <td>Café</td>
      <td>Italian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Sushi Restaurant</td>
      <td>Pharmacy</td>
      <td>Deli / Bodega</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Central Toronto</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Pub</td>
      <td>Pizza Place</td>
      <td>Fried Chicken Joint</td>
      <td>Restaurant</td>
      <td>Liquor Store</td>
      <td>Sports Bar</td>
      <td>Supermarket</td>
      <td>Sushi Restaurant</td>
      <td>Light Rail Station</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Gym</td>
      <td>Park</td>
      <td>Yoga Studio</td>
      <td>Japanese Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Portuguese Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Bar</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Park</td>
      <td>Restaurant</td>
      <td>Pub</td>
      <td>Bakery</td>
      <td>Pizza Place</td>
      <td>Italian Restaurant</td>
      <td>Pharmacy</td>
      <td>Butcher</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Park</td>
      <td>Pub</td>
      <td>Mexican Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Theater</td>
      <td>Restaurant</td>
      <td>Café</td>
      <td>Chocolate Shop</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Clothing Store</td>
      <td>Japanese Restaurant</td>
      <td>Café</td>
      <td>Cosmetics Shop</td>
      <td>Middle Eastern Restaurant</td>
      <td>Electronics Store</td>
      <td>Bakery</td>
      <td>Bookstore</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Clothing Store</td>
      <td>Beer Bar</td>
      <td>Hotel</td>
      <td>American Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Diner</td>
      <td>Cosmetics Shop</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Farmers Market</td>
      <td>Beer Bar</td>
      <td>Steakhouse</td>
      <td>Bakery</td>
      <td>Cocktail Bar</td>
      <td>Cheese Shop</td>
      <td>Seafood Restaurant</td>
      <td>Café</td>
      <td>French Restaurant</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Juice Bar</td>
      <td>Sandwich Place</td>
      <td>Burger Joint</td>
      <td>Ice Cream Shop</td>
      <td>Bakery</td>
      <td>Chinese Restaurant</td>
      <td>Bar</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Downtown Toronto</td>
      <td>1</td>
      <td>Coffee Shop</td>
      <td>Japanese Restaurant</td>
      <td>Sushi Restaurant</td>
      <td>Gay Bar</td>
      <td>Restaurant</td>
      <td>Mediterranean Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Gym</td>
      <td>Hotel</td>
      <td>Dance Studio</td>
    </tr>
  </tbody>
</table>
</div>



#### Cluster 3


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>Central Toronto</td>
      <td>2</td>
      <td>Garden</td>
      <td>Pool</td>
      <td>Dessert Shop</td>
      <td>Farmers Market</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
      <td>Electronics Store</td>
      <td>Eastern European Restaurant</td>
      <td>Dumpling Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



#### Cluster 4


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>East Toronto</td>
      <td>3</td>
      <td>Health Food Store</td>
      <td>Other Great Outdoors</td>
      <td>Trail</td>
      <td>Neighborhood</td>
      <td>Pub</td>
      <td>Eastern European Restaurant</td>
      <td>Dumpling Restaurant</td>
      <td>Electronics Store</td>
      <td>Ethiopian Restaurant</td>
      <td>Event Space</td>
    </tr>
  </tbody>
</table>
</div>



#### Cluster 5


```python
toronto_merged.loc[toronto_merged['Cluster Labels'] == 4, toronto_merged.columns[[1] + list(range(5, toronto_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>Central Toronto</td>
      <td>4</td>
      <td>Mexican Restaurant</td>
      <td>Trail</td>
      <td>Sushi Restaurant</td>
      <td>Jewelry Store</td>
      <td>Yoga Studio</td>
      <td>Diner</td>
      <td>Falafel Restaurant</td>
      <td>Event Space</td>
      <td>Ethiopian Restaurant</td>
      <td>Electronics Store</td>
    </tr>
  </tbody>
</table>
</div>



#### Observations:
Most of the neighborhoods fall into Cluster 2 which are Coffee Shops, cafe, restaurants, bars etc.


```python

```
