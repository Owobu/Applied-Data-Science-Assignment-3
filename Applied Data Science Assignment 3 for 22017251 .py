# Importing necessary Python libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
from sklearn.cluster import KMeans
import cluster_tools as ct
import importlib

importlib.reload(ct)
import errors as err

# Reading in the dataset
df_co2 = pd.read_csv("API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5455005.csv", skiprows = 3 )
df_gdp = pd.read_csv('API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_5455061.csv', skiprows = 3)

def cluster(data):
    
    import sklearn.cluster as cluster
    import sklearn.cluster as cluster
    # reading data
    df_co2 = pd.read_csv('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5455005.csv', skiprows = 3)
    print(df_co2)
    df_gdp = pd.read_csv("API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_5455061.csv", skiprows=(4))
    print(df_gdp.describe())
    
    # cleaning the data and transposing
    df_co2 = df_co2.drop(['1960','1961','1962','1963','1964','1965','1966','1967','1968','1969','1970','1971','1972','1973','1974','1975','1976','1977','1978','1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989'], axis=1)
    
    # transposing
    df = df_co2.transpose()
    
    #using logical slicing
    df_co2 = df_co2[['Country Name','1991','2001','2009']]
    df_co2 = df_co2.dropna()
    print(df_co2)
    

    # transposing gdp data
    df_two = df_gdp.transpose()
    # working gdp data, cleaning and slicing
    df_gdp = df_gdp[['Country Name', '1991', '2001','2009']]
    df_gdp = df_gdp.dropna()
    print(df_gdp)
    
    #making a copy ,year 2010
    df_co2_2009 = df_co2[["Country Name", "2009"]].copy()
    df_gdp_2009 = df_gdp[["Country Name", "2009"]].copy()
    print(df_gdp_2009)
    print(df_co2_2009.describe())
    print(df_gdp_2009.describe())
    
    df_2009 = pd.merge(df_co2_2009, df_gdp_2009, on="Country Name", how="outer")
    print(df_2009.describe())
    df_2009 = df_2009.dropna()
    df_2009.describe()
    df_2009 = df_2009.rename(columns={"2009_x":"emission", "2009_y":"gdp"})
    print(df_2009)
    
    # using scatter matrix 
    pd.plotting.scatter_matrix(df_2009, figsize=(12, 12), s=5, alpha=0.8)
    
    # correlation of variables
    df2009 = df_2009.corr()
    print(df2009)
    df_egdp2009 = df_2009[["emission", "gdp"]].copy()
    
    df_norm, df_min, df_max = ct.scaler(df_egdp2009)
    
    
        
        # loop over number of clusters
    for ncluster in range(2, 10):
        kmeans = cluster.KMeans(n_clusters=ncluster)
        kmeans.fit(df_egdp2009)     
        labels = kmeans.labels_
        cen = kmeans.cluster_centers_
     # calculate the silhoutte score
    print(ncluster, skmet.silhouette_score(df_egdp2009, labels))
    
    n = 3
    kmeans = cluster.KMeans(n_clusters=ncluster)
    kmeans.fit(df_egdp2009)    
    labels = kmeans.labels_
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    xcen = cen[:, 0]
    ycen = cen[:, 1]
    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_egdp2009["emission"], df_egdp2009["gdp"], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel(" co2 emission")
    plt.ylabel("gdp")
    plt.title('world cluster 2009')
    plt.savefig('cluster42009.png', dpi = 300)
    plt.show()
    
    cen = ct.backscale(cen, df_min, df_max)
    xcen = cen[:, 0]
    ycen = cen[:, 1]

    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))

    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_2009["emission"], df_2009["gdp"], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel("emission")
    plt.ylabel("gdp")
    plt.title('world cluster fitted to original scale (2009) ')
    plt.savefig('cluster_orig2009.png', dpi = 300)
    plt.show()
    
    return df_co2
    
Finally = cluster(df_co2)

def fiting(df_gdp):
    df_gdp2 = pd.read_csv("API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_5455061.csv", skiprows=(4))
    df_germany = df_gdp2.iloc[55:56,:]
    df_germany = df_germany.drop(['1960','1961','1962','1963','1964','1965','1966','1967','1968','1969','1970'], axis=1)
    print(df_germany)
    christen = [[1971, 2.94365], [1972, 3.802476], [1973, 4.448017], [1974, 0.85076], [1975, -0.496434], [1976,5.400212], [1977,3.581437], [1978, 3.0981816], [1979, 4.1043313], [1980, 1.198694], [1981,0.3762425 ],[1982,-0.300057812 ],[1983,1.83903412], [1984,3.1789872],[1985,2.556884], [1986,2.240535], [1987, 3.302864],[1988,3.09618], [1989, 4.3516391], [1990,4.34522001],[1991,1.151045],[1992,1.151045],[1993,2.037354],[1994,1.2461419],[1995,0.5144373],[1996, 1.6433343],[1997,1.9984853],[1998, 1.821428031],[1999,2.7732213],[2000,1.5105583],[2001,-0.365628327],[2002,0.7550772],[2003,1.197055352],[2004,1.197055352],[2005,0.78892],[2006,3.933610331],[2007,3.114245883],[2008,1.152029742], [2009,-5.454577169],[2010,4.339606777],[2011,5.869636],[2012,0.230161],[2013,0.230161],[2014,1.784342],[2015,0.617105],[2016,1.408102],[2017,2.297206],[2018,0.678213],[2019,0.828958],[2020,-3.77543],[2021,2.583557]]
    df_germany = pd.DataFrame(data = christen, columns = ['year', 'GDP'])
    df_germany.plot('year','GDP')
    
    def poly(x, a, b, c, d, e):
        
        
        x = x - 2010
        f = a + b*x + c*x**2 + d*x**3 + e*x**4
        return f
    param, covar = opt.curve_fit(poly, df_germany["year"], df_germany["GDP"])
    print(param)
    df_germany["fit"] = poly(df_germany["year"], *param)
    df_germany.plot("year", ["GDP", "fit"])
    plt.xlabel('year')
    plt.ylabel('GDP')
    plt.title('Germany GDP against year')
    plt.savefig('fittingpoly.png', dpi = 300)
    plt.show()
    print(df_germany) 
    
    year = np.arange(1960, 2041)
    forecast = poly(year, *param)
    plt.figure()
    plt.plot(year, forecast, label="forecast")
    plt.plot(df_germany["year"], df_germany["GDP"], label="GDP")
    plt.xlabel("year")
    plt.ylabel("GDP")
    plt.title('Germany GDP for 2041 forecasted')
    plt.legend()
    plt.savefig('poly_forecast.png', dpi = 300)
    plt.show()
    
    sigma = np.sqrt(np.diag(covar))
    df_germany["fit"] = poly(df_germany["year"], *param)
    df_germany.plot("year", ["GDP", "fit"])
    plt.show()

    print("turning point", param[2], "+/-", sigma[2])
    print("GDP at turning point", param[0]/100, "+/-", sigma[0]/100)
    print("growth rate", param[1], "+/-", sigma[1])
    
    # error bandgap
    sigma = np.sqrt(np.diag(covar))
    low, up = err.err_ranges(year, poly, param, sigma)
    plt.figure()
    plt.plot(df_germany["year"], df_germany["GDP"], label="GDP")
    plt.plot(year, forecast, label="forecast")
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.xlabel("year")
    plt.ylabel("GDP")
    plt.legend()
    plt.savefig('errorband.png', dpi=300)
    plt.show()
    print(param)
    
    
    
    
    return df_germany
    

    
curve_fit = fiting(df_gdp)



def germany():
    import sklearn.cluster as cluster
    df_gdp2 = pd.read_csv("API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_5455061.csv", skiprows=(4))
    df_germany = df_gdp2.iloc[55:56,:]
    christen = [[1971, 2.94365], [1972, 3.802476],[1973, 4.448017],[1974, 0.85076],[1975, -0.496434],[1976,5.400212], [1977,3.581437], [1978, 3.0981816], [1979, 4.1043313], [1980, 1.198694], [1981,0.3762425 ],[1982,-0.300057812 ],[1983,1.83903412], [1984,3.1789872],[1985,2.556884], [1986,2.240535], [1987, 3.302864],[1988,3.09618], [1989, 4.3516391], [1990,4.34522001],[1991,1.151045],[1992,1.151045],[1993,2.037354],[1994,1.2461419],[1995,0.5144373],[1996, 1.6433343],[1997,1.9984853],[1998, 1.821428031],[1999,2.7732213],[2000,1.5105583],[2001,-0.365628327],[2002,0.7550772],[2003,1.197055352],[2004,1.197055352],[2005,0.78892],[2006,3.933610331],[2007,3.114245883],[2008,1.152029742], [2009,-5.454577169],[2010,4.339606777],[2011,5.869636],[2012,0.230161],[2013,0.230161],[2014,1.784342],[2015,0.617105],[2016,1.408102],[2017,2.297206],[2018,0.678213],[2019,0.828958],[2020,-3.77543],[2021,2.583557]]
    df_germany = pd.DataFrame(data = christen, columns = ['year', 'GDP'])
    
    df_germany = pd.DataFrame(data = christen, columns = ['year', 'GDP'])
    df_germany['emission'] = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0, 310589.9963, 303350.0061, 298299.9878, 285730.011, 289079.9866, 299799.9878, 312859.9854, 311910.0037, 295130.0049
    df_germany.describe()
    df_germany_cluster = df_germany[["GDP", "emission"]].copy()
    df_cluster, df_min, df_max = ct.scaler(df_germany_cluster)
    for ncluster in range(2, 8):
    
        kmeans = cluster.KMeans(n_clusters=ncluster)
        kmeans.fit(df_germany_cluster)     
        labels = kmeans.labels_
        cen = kmeans.cluster_centers_
        print(ncluster, skmet.silhouette_score(df_germany_cluster, labels))
        df_cluster, df_min, df_max = ct.scaler(df_germany_cluster)
    
    ncluster = 4
    kmeans = cluster.KMeans(n_clusters=ncluster)
    kmeans.fit(df_germany_cluster)     
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    xcen = cen[:, 0]
    ycen = cen[:, 1]

    plt.figure(figsize=(8.0, 8.0))

    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_germany_cluster["GDP"], df_germany_cluster["emission"], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel("GDP")
    plt.ylabel("emission")
    plt.title('Germany cluster')
    plt.savefig('germany_cluster.png', dpi=300)
    plt.show()


    cen = ct.backscale(cen, df_min, df_max)
    xcen = cen[:, 0]
    ycen = cen[:, 1]
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_germany["GDP"], df_germany["emission"], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel("GDP")
    plt.ylabel("emission")
    plt.title('Germany cluster centered to original scale')
    plt.savefig('Germany_cluster_orig.png', dpi=300)
    plt.show()
    
germany()
    
    

    
    
    
        
