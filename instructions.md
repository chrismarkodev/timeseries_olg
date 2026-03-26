This project uses python and pytorch.
I have a timeseries where each data point or observation is a vector of 7 categorical values.
The categorical values are represented by numbers from 0 up to 49, so there is 50 categories.
The timeseries data is in a csv file "data/data_all_l649.csv".
The csv file has columns headings in the first row. 
Each subsequent row represents a single observations and it has 
an observation date in the column named 'date'. 
The subsequent columns are named 'd1', 'd2','d3','d4','d5','d6','bonus' 
and contain the observed categorical values. 
A single row may not have repeating categories.

Create python code to input the data file, prepare it for processing, 
then create a transformer model to train on the created sequences, 
and then predict next observation values in columns d1 - bonus. 

The length of the sequence should be a parameter that can be modified.
You may also use other parameter variable sinstead of hardcoding values for column names, file name, etc.

Clarification: The order of the 7 categorical values does not matter; however, they are stored in the file in ascending order except for the 'bonus' column.