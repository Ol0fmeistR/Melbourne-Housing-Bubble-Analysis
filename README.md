### Analyzing the Melbourne Housing Bubble of 2016-19 <br>
Data visualization notebook for the purpose of analyzing the Melbourne Housing Bubble of 2016-19. To view the Jupyter notebook, click <a href="https://melbournehousingbubble.netlify.app/">here</a> since github pages do not support git lfs. <br>


### Summary: <br>

**Variables involved in the data:** <br> 

**Suburb:** Denotes the suburb the property is located in <br>
**Address:** Address of the property <br>
**Rooms:** Number of rooms available in the property <br>
**Price:** Price in Aussie Dollars <br>

**Method:** <br>
S: Property Sold <br>
SP: Property Sold Prior <br>
PI*: Property Passed In <br>
PN: Sold Prior Not Disclosed <br>
SN: Sold Not Disclosed <br>
NB: No Bid <br>
VB: Vendor Bid <br>
W: Withdrawn Prior to Auction <br>
SA: Sold After Auction <br>
SS: Sold After Auction Price Not Disclosed <br>
N/A: Price or Highest Bid Not Available <br>

**Type:** <br>
h: House, Cottage, Villa, Semi, Terrace <br>
u: Unit, Duplex <br>
t: Townhouse <br>
dev site: Development Site <br>
o res: Other residential <br>

**SellerG:** Real Estate Agent <br>

**Date:** Date when the property was sold <br>

**Distance:** Distance from CBD in kilometers <br>

**Regionname:** General Region(North, West, South, East, North-East etc.) <br>

**Propertycount:** Number of properties that exist in the suburb <br>

**Bedroom2:** Number of Bedrooms in the property <br>

**Car:** Number of Car Spots <br>

**LandSize:** Size of the Land in squared meters <br>

**BuildingArea:** Size of the Building Area in squared meters <br>

**YearBuilt:** Year in which the property was built <br>

**CouncilArea:** Governing Council for the Area <br>

**PostCode:** Postal Code of the Area <br>

**Latitude:** Latitude of the property location <br>

**Longitude:** Longitude of the property location <br>


*The jupyter notebook has been hosted as an html page on <a href="https://melbournehousingbubble.netlify.app/">netlify</a>, but in case you wanna skip the razzle dazzle, here's a quick stats check with some fancy graphs.* <br>

**Latitude/Longitude geoplot using folium:** <br>

![](https://github.com/Ol0fmeistR/Melbourne-Housing-Bubble-Analysis/blob/main/Plots/Screenshot%20(94).png) 

<br>

**Average Price of Properties by Council Area:** <br>

![](https://github.com/Ol0fmeistR/Melbourne-Housing-Bubble-Analysis/blob/main/Plots/Council%20area%20vs%20Price.png)

<br>

**Average Price of Properties by Region:** <br>

![](https://github.com/Ol0fmeistR/Melbourne-Housing-Bubble-Analysis/blob/main/Plots/Avg%20Price%20vs%20Region.png)

<br>

**Relationship between Number of Rooms in a Property vs Price:** <br>

![](https://github.com/Ol0fmeistR/Melbourne-Housing-Bubble-Analysis/blob/main/Plots/Rooms%20vs%20Price.png)

<br>

**Distribution of Price with Season and Property Type:** <br>

![](https://github.com/Ol0fmeistR/Melbourne-Housing-Bubble-Analysis/blob/main/Plots/Price%20vs%20Season%20vs%20Prop%20type.png)

<br>

**Takeaways:** <br>
1. Training the data using time series cross validation (or adversarial if data is time series and train/test are not from the same distribution) will most likely yield better results compared to a simple k fold cross validation. <br>
2. Bayes optimization produces better hyperparameters compared to grid search CV provided we trained and validated our model based on a time series cross validation. <br>
3. Stacking / Ensembling different Machine Learning models instead of using just a singular one gives a significant boost to the accuracy. <br>
