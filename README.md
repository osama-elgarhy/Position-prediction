# Position-prediction
The Datasets (Real and synthetic) used for position prediction. 

The repo includes some functions and datasets for position prediction that are used for service placement and migration task. 

The Datasets (Real and synthetic) used for position prediction.

  -  Data file obtained from one of the RTK measurements campaign, which is used for the position 
     prediction.
  -  data_col25.csv: Field trials' dataset. This is one of the Excel files that represent the real dataset       (from the measurement campaign). The input features are 25 (The features are COG (course over ground),      COGD (COG difference), longitude, latitude, and speed. Five previous measurements are stored and used 
     for each of these features).
  -  The Excel files are some of the different synthetic datasets that are used for position prediction.

    
ANNGridSearch.py: An ANN python script (with grid-search) that performs the position prediction.

ErlangB.m: An example of two Erlang-B calculation functions. That can be used to calculate the Blocking probability.

Other real and synthetic datasets, obtained using different devices and from various physical locations, are also available. For inquiries, please reach out to the contact person (Osama Elgarhy).
