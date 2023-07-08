import geopandas as gp
import osmnx as ox
from meteostat import Point as P
from meteostat import Daily
from datetime import datetime

import numpy as np
import pandas as pd
from numpy import random as rand
from shapely.geometry import Point
from math import sin, cos, acos, radians

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pulp import LpProblem, LpMinimize, LpVariable, LpBinary, lpSum, LpStatus, value

import matplotlib.pyplot as plt
plt.style.use('ggplot')


## OBJECTS ##
class City:
    def __init__(self, name, population, size, height, gdf):
        self.name = name
        self.population = population
        self.size = size
        self.height = height
        self.gdf = gdf

    def __str__(self) -> str:
        return str(f"Name: {self.name}\nPopulation: {self.population}\nSize: {self.size}\nHeight: {self.height}\nGDF: \n{self.gdf.head()}")

class Neighborhood:
    def __init__(self, id, x, y, population):
        self.id = id
        self.x = x
        self.y = y
        self.population = population
        
    def __str__(self) -> str:
        return str(f"ID: {self.id}\nCoordinates:\nX: {np.round(self.x[0], 3)}\nY: {np.round(self.y[0], 3)}\nPopulation: {self.population}\n")

class Allotment:
    def __init__(self, id, x, y, size):
        self.id = id
        self.x = x
        self.y = y
        self.size = size
    
    def __str__(self) -> str:
        return str(f"ID: {self.id}\nCoordinates: \nX: {np.round(self.x[0], 3)}\nY: {np.round(self.y[0], 3)}\nSize: {self.size}\n")



## PLOTS ##

def isInsideMap(point, multipolygon):
    """Returns True if the point is inside the multipolygon, False otherwise."""
    return any(multipolygon.geometry.contains(point))

def plotNeighboors(city):
    """Plots the neighborhoods inside the city and the plots. 
    Returns array of neighborhoods and plots"""
    gdf = city.gdf
    multiP = gp.GeoDataFrame(geometry=[gdf.head()['geometry'][0]])
    
    #center
    X = (gdf['bbox_east'] + gdf['bbox_west'])/ 2
    Y = (gdf['bbox_north'] + gdf['bbox_south'])/2

    #Max delta 
    deltaX = gdf['bbox_east'] - gdf['bbox_west']
    deltaY = gdf['bbox_north'] - gdf['bbox_south']

    ax = gdf.plot(color='white', edgecolor='black', figsize=(10, 10)).set_axis_off()
    
    #compute plots settings
    neigs, plots, allHoods, allPlots = [], [], [], []

    percSup = 0.00004
    size = 150
    plots.append(size)

    sum, i = plots[0], 1
    while sum < percSup * city.size:
        plots.append(size + rand.randint(-80, 80))
        sum += plots[i]
        i += 1

    #plot plots inside the gdf randomically
    i=1
    xtoplot = []
    ytoplot = []
    while i <= len(plots):
        x = X + (rand.uniform(-1*deltaX, deltaX))   
        y = Y + (rand.uniform(-1*deltaY, deltaY))
        if isInsideMap(Point(x, y), multiP):
            plot = Allotment(i, x, y, plots[i-1])
            allPlots.append(plot)
            xtoplot.append(x)
            ytoplot.append(y)
            i += 1

    plt.scatter(xtoplot, ytoplot, c='orange', s=30, label="Possible locations")

    #compute neighborood settings
    minSize, upperDelta = 10_000, 8_000
    # To make the hood point dynamic
    # minPointSize, maxPointSize = 40, 300
    # deltaPoint = maxPointSize - minPointSize
    # step = ((minSize+upperDelta) - minSize)/deltaPoint

    neigs.append(minSize)
    sum, i = neigs[0], 1
    while sum < city.population:    
        neigs.append(minSize + rand.randint(1_000, upperDelta))
        sum += neigs[i]
        i += 1

    plt.title(gdf['display_name'][0])
    
    #plot neighborhoods inside the gdf randomically
    i=1
    xtoplot = []
    ytoplot = []
    while i <= len(neigs):
        x = X + (rand.uniform(-1*deltaX, deltaX))    
        y = Y + (rand.uniform(-1*deltaY, deltaY))

        if isInsideMap(Point(x, y), multiP):
            neigh = Neighborhood(i, x, y, neigs[i-1])
            # pointSize = int(np.floor((neigh.population - minSize)/step) + minPointSize)
            xtoplot.append(x)
            ytoplot.append(y)
            #compute the pin size depending on the demand
            allHoods.append(neigh)
            i += 1
    
    plt.scatter(xtoplot, ytoplot, c='#0059b3', s=100, label='Neighborhoods')
    plt.legend(facecolor='white', edgecolor='black', loc=0)
    plt.plot(ax = ax)
    return allHoods, allPlots


def summary(y_j):    
    # To see which locations have been selected
    selected = [j for j in y_j if int(y_j[j].varValue) != 0]
    print(f"Total locations selected: {len(selected)}.")

    # To see how many locations have been discarded/selected
    recap = [[int(i), int([j.varValue for j in y_j.values()].count(i))] for i in set([j.varValue for j in y_j.values()])]
    print(f"Build site at: {list(selected)}")
    
    c = ['#990000', '#0059b3']
    vals = [recap[1][1], recap[0][1]]
    plt.figure(figsize=(8, 6))
    plt.barh(['Yes', 'No'], vals, color=c)
    plt.xlabel('Number of sites', fontweight='bold')
    plt.ylabel('Establish', fontweight='bold')
    plt.title('Plots sites to be established', fontweight='bold')
    for i, v in enumerate(vals):
        plt.text(v + 0.5, i, str(v), color=c[i], va='center', fontweight='bold')
    plt.show()
    return selected


def optimizedPlotSites(city, ids, plots): 
   # Plot the shape of the city
   ax = city.gdf.plot(color='white', edgecolor='black', figsize=(10, 10)).set_axis_off()

   # Plot sites to establish
   plt.scatter([j.x[0] for j in plots for i in ids if j.id == i],\
               [j.y[0] for j in plots for i in ids if j.id == i],\
               c='green', s=200, marker='*', label="Select")

   # Plot sites to discard
   plt.scatter([j.x[0] for j in plots if j.id not in ids],\
               [j.y[0] for j in plots if j.id not in ids],\
               c='#990000', s=60, marker='x', label="Discard")
   name = city.gdf['display_name'][0]
   plt.title(f'Optimized Plots Sites\n{name}', fontweight='bold')
   plt.legend(title='Plots Site', facecolor='white', edgecolor='black', loc=0)
   plt.plot(ax = ax)


def getLinkedHood(idPlot, x_ij):
    '''
    Find hood ids that are served by the input plot.
    
    Args:
        - idPlot: string (example: <plot 21>)
    Out:
        - List of hoods ids connected to the plot
    '''
    # Initialize empty list
    linkedHoods = []

    # Iterate through the xij decision variable
    for (k, v) in x_ij.items():
        
        # Filter the input plot and positive variable values
        if k[1] == idPlot and v.varValue > 0:

            # Customer is served by the input warehouse
            linkedHoods.append(k[0])
    return linkedHoods


def optimizedCity(city, hoods, plots, ids, x_ij):
    # Plot the shape of the city
    ax = city.gdf.plot(color='white', edgecolor='black', figsize=(10, 10)).set_axis_off()

    # Plot sites to establish
    plt.scatter([j.x[0] for j in plots for i in ids if j.id == i],\
                [j.y[0] for j in plots for i in ids if j.id == i],\
                c='#00960a', s=100, marker='o', label="Selected plots")

    # Plot hoods
    plt.scatter([i.x[0] for i in hoods], [i.y[0] for i in hoods], c='#0059b3', s=80, marker='o', label='Neighborhood')

    # For each plot to build
    for p in ids:

        # Extract list of hoods connected to the plot
        linkedHoods = getLinkedHood(p, x_ij)

        # for each linked hood
        for h in linkedHoods:

            # Plot connection between plot and the connected hood
            for i in hoods:
                if i.id == h:
                    xHood = i.x[0]
                    yHood = i.y[0]
            
            for i in plots:
                if i.id == p:
                    xPlot = i.x[0]
                    yPlot = i.y[0]

            plt.plot([xPlot, xHood], [yPlot, yHood], linewidth=0.8, linestyle='--', color='#01678f')
    name = city.gdf['display_name'][0]
    plt.title(f'Optimized Hood Connections\n{name}')
    plt.legend(facecolor='white', loc=0)
    plt.show()
    return


## COST FUNCTIONS ##
def dist(hood, plot):
    '''
    Calculate distance between two locations given hood and plot.
    '''
    lat1 = hood.x;  lon1 = hood.y
    lat2 = plot.x;  lon2 = plot.y

    return  np.round(6371.01 *\
            acos(sin(radians(lat1))*sin(radians(lat2)) +\
            cos(radians(lat1))*cos(radians(lat2))*cos(radians(lon1)-radians(lon2))), 3)


def computeCosts(hoods, plots):
    '''
    Computes the costs between each possible locations and each hood for each plot.
    Returns a dictionary with as a key has the id of the i-th hood and as a value a dictionary 
    where are stored the distance between the i-th hood and all the plots
    '''
    # Dict to store the distances between all hoods and plots
    costsDict= {}

    # for each possible locations 
    for _, hood in enumerate(hoods):

        # Dict to store the distances between the j-th plot and all hoods
        plotsCostDict = {} 

        # for each plot location
        for _, plot in enumerate(plots):

            # distance between hood i and location j
            d = dist(hood, plot)

            # Update the cost dict
            plotsCostDict.update({plot.id : d})
        
        # Update the general cost dict containing for each location all the dist wrt all hoods
        costsDict.update({hood.id : plotsCostDict})
    return costsDict



def solveProblem(city):
    hoods, plots = plotNeighboors(city)

    ## VARIABLES
    # d_j activation cost: defined by the size (mq) of the terrain
    d_j = {i.id : i.size for i in plots}

    # offer_j: how much the j-th plot offers
    q = 1.5     # production %
    offer_j = {i.id : i.size * q for i in plots}

    # r_i demand by the i-th hood
    k = 0.004   # demand %
    r_i ={i.id : i.population * k for i in hoods}

    # Connections cost
    c_ij = computeCosts(hoods, plots)
    # printDict(c_ij)


    ## LINEAR MODEL
    i_hood = list(c_ij.keys())          # IDs of each hood
    j_plot = [j.id for j in plots]      # IDs of each plot

    # CFLP: 'Capacitated Facility Location Problem'
    problem = LpProblem('CFLP', LpMinimize)

    # Var: y_j (costraint: it's binary)
    y_j = LpVariable.dicts('y', j_plot, 0, 1, LpBinary)

    # Var: x_ij (costraint: it's binary)
    x_ij = LpVariable.dicts('x', [(i, j) for i in i_hood for j in j_plot], 0, 1, LpBinary)

    # Objective function
    obj = lpSum(d_j[j] * y_j[j] for j in j_plot) +\
        lpSum(c_ij[i][j] * x_ij[(i, j)] for i in i_hood for j in j_plot)

    problem += obj

    ### DON'T TOUCH!! ###
    # Costraint: each hood hass exaclty 1 plot connected to 
    for i in i_hood:
        problem += lpSum(x_ij[(i, j)] for j in j_plot) == 1

    # Costraint: connection i-j ==> j is active
    for i in i_hood:
        for j in j_plot:
            problem += x_ij[(i, j)] <= y_j[j]

    # Costraint: the demand must be met
    for j in j_plot:
        problem += lpSum(r_i[i] * x_ij[(i, j)] for i in i_hood) <= offer_j[j]

    # print(problem)


    ## SOLUTION
    problem.solve()

    if LpStatus[problem.status] == 'Optimal':
        print("Best solution found!")
    else:
        print(f"Problem {LpStatus[problem.status]}")

    print(value(problem.objective))
    ids = summary(y_j)
    optimizedPlotSites(city, ids, plots)  
    optimizedCity(city, hoods, plots, ids, x_ij)



## MANAGE PLANTS ##

def locationDetails(lat, lon, height, ):

    #set the area of the lot of land
    #set the reference year 
    annoprec = 2022
    # Set time period
    start = datetime(annoprec, 1, 1)
    end = datetime(annoprec, 12, 31)

    # Create Point for location deired
    location = P(lat, lon, height) 

    emisfero=1
    if (location._lat<0):   
        emisfero=2
    # Get daily data for 2022
    data = Daily(location, start, end)
    data = data.fetch()

    # Plot line chart including average, minimum and maximum temperature
    data.plot(y=['tavg', 'tmin', 'tmax'])
    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")
    plt.title("Location Details")
    plt.show()

    M_DF=pd.DataFrame(data)

    med = [0,0,0,0,0,0,0,0,0,0,0,0]
    intmed = [0,0,0,0,0,0,0,0,0,0,0,0]
    i=0
    for x in M_DF.tavg:
        if(not pd.isnull(x)): # if value is != NaN
            cdate=M_DF.index[i].to_pydatetime()
            m=cdate.month-1
            med[m]=med[m]+x
            intmed[m]=intmed[m]+1
        i=i+1
    i=0
    for i in range(12):
        med[i]=med[i]/intmed[i]
    plt.plot(med)
    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")
    plt.title("AVG temperature on each month")
    plt.grid(axis='y')
    plt.show()
    return med, emisfero


# Define a function for converting the month's name in number
def mtn(x,emisfero):
    if(emisfero==1):
        months = {
            'January': 1, 'February': 2, 'March': 3, 'April':4, 'May':5, 'June':6, 
            'July':7, 'August':8, 'September':9, 'October':10,'November':11, 'December':12
            }
    else:
        months = {
            'January': 7, 'February': 8, 'March': 9, 'April':10, 'May':11, 'June':12,
            'July':1 ,'August':2, 'September':3, 'October':4, 'November':5, 'December':6}

    try:
        ez = months[x]
        return ez
    except:
        raise ValueError('Not a month')


# Tests if the plant can survive based on the temperature
def isOkToPlant(med, mese1, mese2,t_min, t_max, sett_prod):
    #change unit for sett_prod to months 
    mon_prod=sett_prod/4.333
    #i need to calculate a gap becouse i can pant in every month between these
    if(mese1<mese2):
        gap=mese2-mese1
    else:
        gap=12-(mese1-mese2)
    isok = [True] * (gap)
    #now lets check for every possible temporal space
    for i in range(gap):
        for j in range(int(mon_prod)):
            #looking for not going out of month array
            if(mese1+i+j>12):
                m=mese1+i+j-12
            else:
                m=mese1+i+j
            #checking if is ok
            if(t_min>med[m-1] or med[m-1]>t_max):
                isok[i]=False
    return any(isok) 


def plantSuggestions(Piante_DF, emisfero, med):
    pianteOk = pd.DataFrame()

    for x in Piante_DF['Plants']:
        pianta = Piante_DF.loc[Piante_DF['Plants'] == x].iloc[0]

        mesi = pianta['MONTHS landfill']                          #Piante_DF.loc[Piante_DF['Piante'] == x, 'MESI interramento'].iloc[0]
        t_min = pianta['MINIMUM TEMPERATURE']                        #Piante_DF.loc[Piante_DF['Piante'] == x, 'TEMPERATURA MINIMA'].iloc[0]
        t_max = pianta['MAXIMUM TEMPERATURE']                       #Piante_DF.loc[Piante_DF['Piante'] == x, 'TEMPERATURA MASSIMA'].iloc[0]
        sett_prod = pianta['PRODUCTION TIME (weeks)']          #Piante_DF.loc[Piante_DF['Piante'] == x, 'TEMPO PRODUZIONE (settimane)'].iloc[0]

        mesi = mesi.split("-")
        mese1s = mesi[0]
        mese2s = mesi[1]
        mese1=mtn(mese1s,emisfero)
        mese2=mtn(mese2s,emisfero)

        isok = isOkToPlant(med,mese1,mese2,t_min,t_max,sett_prod)
        mesi2 = pianta['MONTHS landfill 2']
        if(isok==True):
            pianteOk = pd.concat([pianteOk, pianta],axis=1, ignore_index=True)
        elif(not pd.isnull(mesi2)): # if value is != NaN we have another spot for planting this
            mesi2 = mesi2.split("-")
            mese1s = mesi2[0]
            mese2s = mesi2[1]
            mese1=mtn(mese1s,emisfero)
            mese2=mtn(mese2s,emisfero)
            isok = isOkToPlant(med,mese1,mese2,t_min,t_max,sett_prod)
            if(isok==True):
                pianteOk = pd.concat([pianteOk, pianta],axis=1, ignore_index=True)

    #since I have not assigned names to the columns if I had entered axis=0 it would have resulted in a DF with a single column,
    # instead using axis=1 the column names end up under the same column, so I do the transpose
    pianteOk = pianteOk.transpose()
    return pianteOk


def yeldAvg(pianteOk, plotSize):
    
    #calculating the average return
    media_ren=0
    for x in pianteOk['YIELD PER SQM']:
        media_ren=media_ren+x
    media_ren = media_ren/pianteOk.shape[0]

    #now we can calculate the yield of the land expressed in kg
    avg = plotSize * media_ren
    return avg



## To print ##

def printPlots(plots):
    for idx, plot in enumerate(plots):
        print(f"Plot {idx}:\n{plot}")

def printHoods(hoods):
    for idx, hood in enumerate(hoods):
        print(f"Hood {idx}:\n{hood}")
    
def printDict(dict):
    for _, key in enumerate(dict):
        print(f"{key} : {dict.get(key)}")
