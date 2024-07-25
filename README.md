# Sipefield 
## Real-time, synchronized, sensorimotor data collection
 PsychoPy Visual stimulus gratings for head fixed behavioral experiments:
 - Uses NIDAQ to trigger and sychnronize presentation
 - Arduino Rotary encoder thread encodes locomotion at screen refresh rate
 - Presents 2 second sinusoidal grating and 3 second gray screen for user-define amount of trials

 ## Includes Arduino rotary encoder script
 - Arduino file sends 'clicks' for a python script to decode during the PsychoPy experiment
 - A Spyder python scrtipt is included to troubleshoot encoder parameters
    - For example, a wheel with a diameter of 151 cm should use an encoder CPR of 1000

- Arduino code in the PsychoPy Builder can be found at in the `code_encoder` custom code block in the Routine
