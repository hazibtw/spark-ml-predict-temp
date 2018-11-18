# spark-ml-predict-temp

1)download  csv data which is avaliable in data folder.

2)load csv data 

3)structure of the data:

      year: 2016 for all data points

      month: number for month of the year(categorical)

      day: number for day of the year(categorical)

      week: day of the week as a character string(categorical)

      temp_2: max temperature 2 days prior

      temp_1: max temperature 1 day prior

      average: historical average max temperature

      actual: max temperature measurement

      friend: your friend’s prediction, a random number between 20 below the average and 20 above the average


3)  Random forest algorithm predicting temp

      why Random forest algorithm for this data points.

      Each individual brings their own background experience and information sources to the problem. Some people may swear by Accuweather, while others will only look at NOAA (National Oceanic and Atmospheric Administration) forecasts. 
      but by combining everyone’s predictions together, our net of information is much greater. 

      people might rely on different sources to make a prediction, each decision tree in the forest considers a random subset of features when forming questions and only has access to a random set of the training data points. 

  
