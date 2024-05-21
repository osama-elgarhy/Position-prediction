# Position-prediction
The Datasets (Real and synthetic) used for position prediction. 

The U-blox log file (.ubx) contains a data file obtained from one of the RTK measurements campaign, which is used for the position prediction. 

data_RTK_Field_25.csv: Field trials' dataset. This is one of the Excel files that represent the real dataset (from the measurement campaign). The input features are 25 (The features are COG (course over ground), COGD (COG difference), longitude, latitude, and speed. Five previous measurements are stored and used for each of these features).

ANN_GridSearch.py: An ANN python script (with grid-search) that performs the position prediction.

The Excel files are some of the different synthetic datasets that are used for position prediction.

ErlangB.m: An example two Erlang-B calculation functions. That can be used to calculate the Blocking probability. 
