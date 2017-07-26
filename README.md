# Delta-Quad-Sims
Calculations regarding kinematics of delta quad control. 

The major work in this project includes using ANFIS to evaulate the kinmetaics of the Delta-Quad legs. To accomplish this, forward kinemtaics are calculated (forwardKinematics.py) and are used to create a large lookup table of the kinematics throughout the workpsace. All of these lookup tables are the ".csv" files. 
To ensure the kinematics are being caluclated correctly, a 3D model is made to ensure there are no major issues with the kinematics.  
To train the ANFIS network, multiple versions of the "neuroFuzzyControl_#.py" were experimented with to achieve the best results. Since this would often take a long time to execute, ".log" files were made to keep status of the results over time. ANFIS results were saved to ".pkl" files.
The trajectory planning was designed using various functions found in the "piecewiseMotion.py" script.
A few scripts test the Blynk Python API in action. 
There are many other "random" Python scripts used to test key functions, calculations and test results. 

# Dependencies/Setup
* numpy
* scikit-fuzzy
* matplotlib
* anfis (https://github.com/twmeggs/anfis)
* Blynk (https://github.com/xandr2/blynkapi)


