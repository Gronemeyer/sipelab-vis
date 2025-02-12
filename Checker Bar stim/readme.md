# Moves an alternating checkerboard light-bar across the screen 



Each randomize trial is currently ~106.8 seconds

- Horizontal Down (23.4s)

- Horizontal Up (23.4s)

- Vertical Right (30.0s)

- Vertical Left (30.0s)

NOTE: Trials are defined in an accompanying `.csv` file


## Features:



`InputOutput` Routine available for running experiment as a script/subprocess. Accepts a serialized python dictionary object containing parameters for saving files to BIDS format; is extensible to receive extra parameters from any parent process capable of serializing a python dictionary object.





## How to prep for exporting to script from builder:


- Disable dialog info in builder settings

- Set Trials to nTrials

- Edit settings -> Data output to `sysarg_save_path`

- Enable InputOutput routine

