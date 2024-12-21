#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.1post4),
    on December 18, 2024, at 18:14
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from import_modules
import numpy as np
import time
from psychopy import visual

from stimupy.stimuli.checkerboards import checkerboard


# Run 'Before Experiment' code from get_input_arguments
#=============================== Custom Codeblock jgronemeyer =====================================#
#add system argument functionality
#Get command line arguments passed if present
sysarg_protocol_id = 'prot'
sysarg_subject_id = 'sub'
sysarg_session_id = 'ses'
sysarg_save_dir = None
nTrials = 1 # changed data.TrialHandler2 object value nReps value to call this variable
#if len(sys.argv) > 1:
#    sysarg_protocol_id = sys.argv[1]  # get the first argument from command line
#    sysarg_subject_id = sys.argv[2]  # get the second argument from command line
#    sysarg_session_id = sys.argv[3]  # get the third argument from command line
#    sysarg_save_dir = sys.argv[4]  # get the fourth argument from command line
#    nTrials = sys.argv[5] # get the fifth argument from the command line
    #encoder = sys.argv[6] # TODO: accept encoder parameters
#==================================================================================================#

# Run 'Before Experiment' code from save_bids_format
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.1post4'
expName = 'CheckerBar_vis_build-v0.8'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'Protocol ID': sysarg_protocol_id,
    'Subject ID': sysarg_subject_id,
    'Session ID': sysarg_session_id,
    'Save Directory': sysarg_save_dir,
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s/sub-%s/ses-%s/beh/sub-%s_ses-%s_%s' % (expInfo['Protocol ID'], expInfo['Subject ID'], expInfo['Session ID'], expInfo['Subject ID'], expInfo['Session ID'], timestamp)
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\sipefield\\sipefield-vis\\Checker Bar stim\\CheckerBar_vis_build-v0.8.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('warning')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=2,
            winType='pyglet', allowStencil=False,
            monitor='wfieldMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = True
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Custom_Imports" ---
    
    # --- Initialize components for Routine "Input_Output" ---
    
    # --- Initialize components for Routine "Custom_Trigger" ---
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "Checkerboard" ---
    # Run 'Begin Experiment' code from code
    #PARAMETERS
    ppd = 31
    frame_rate = 60
    color_alternating_freq = int(frame_rate/12)
    
    # Vertical parameters
    # vertical_size = (0.2, 1)
    vertical_stim = checkerboard(
        visual_size=(13, 3),
        ppd=ppd,
        shape=None,
        frequency=0.5,
        board_shape=(13, 3),
        check_visual_size=1,
        period='ignore',
        rotation=0,
        intensity_checks=(-1, 1),
        intensity_target=1
    )
    
    # Horizontal parameters
    # horizontal_size = (1.8, 0.2)
    horizontal_stim = checkerboard(
        visual_size=(3, 15),
        ppd=ppd,
        shape=None,
        frequency=0.5,
        board_shape=(3, 15),
        check_visual_size=1,
        period='ignore',
        rotation=0,
        intensity_checks=(-1, 1),
        intensity_target=1
    )
    
    #orientation_dict = {
    #    'vertical_left': vertical_stim,
    #    'horizontal_up': horizontal_stim,
    #    'vertical_right': vertical_stim,
    #    'horizontal_down': horizontal_stim
    #}
    
    # NOTE: Assign the 'img' key value 
    # `vertical_stim['img']` to the grating Texture
    
    checker_mask = visual.GratingStim(
        win=win, name='checker_mask',
        tex=None, mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0, sf=None, phase=1.0,
        color='white', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-1.0)
    gray_screen = visual.ImageStim(
        win=win,
        name='gray_screen', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Custom_Imports" ---
    # create an object to store info about Routine Custom_Imports
    Custom_Imports = data.Routine(
        name='Custom_Imports',
        components=[],
    )
    Custom_Imports.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Custom_Imports
    Custom_Imports.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Custom_Imports.tStart = globalClock.getTime(format='float')
    Custom_Imports.status = STARTED
    Custom_Imports.maxDuration = None
    # keep track of which components have finished
    Custom_ImportsComponents = Custom_Imports.components
    for thisComponent in Custom_Imports.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Custom_Imports" ---
    Custom_Imports.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Custom_Imports.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Custom_Imports.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Custom_Imports" ---
    for thisComponent in Custom_Imports.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Custom_Imports
    Custom_Imports.tStop = globalClock.getTime(format='float')
    Custom_Imports.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "Custom_Imports" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Input_Output" ---
    # create an object to store info about Routine Input_Output
    Input_Output = data.Routine(
        name='Input_Output',
        components=[],
    )
    Input_Output.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Input_Output
    Input_Output.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Input_Output.tStart = globalClock.getTime(format='float')
    Input_Output.status = STARTED
    Input_Output.maxDuration = None
    # keep track of which components have finished
    Input_OutputComponents = Input_Output.components
    for thisComponent in Input_Output.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Input_Output" ---
    Input_Output.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Input_Output.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Input_Output.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Input_Output" ---
    for thisComponent in Input_Output.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Input_Output
    Input_Output.tStop = globalClock.getTime(format='float')
    Input_Output.tStopRefresh = tThisFlipGlobal
    thisExp.nextEntry()
    # the Routine "Input_Output" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Custom_Trigger" ---
    # create an object to store info about Routine Custom_Trigger
    Custom_Trigger = data.Routine(
        name='Custom_Trigger',
        components=[key_resp],
    )
    Custom_Trigger.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for Custom_Trigger
    Custom_Trigger.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Custom_Trigger.tStart = globalClock.getTime(format='float')
    Custom_Trigger.status = STARTED
    thisExp.addData('Custom_Trigger.started', Custom_Trigger.tStart)
    Custom_Trigger.maxDuration = None
    # keep track of which components have finished
    Custom_TriggerComponents = Custom_Trigger.components
    for thisComponent in Custom_Trigger.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Custom_Trigger" ---
    Custom_Trigger.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Custom_Trigger.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Custom_Trigger.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Custom_Trigger" ---
    for thisComponent in Custom_Trigger.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Custom_Trigger
    Custom_Trigger.tStop = globalClock.getTime(format='float')
    Custom_Trigger.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Custom_Trigger.stopped', Custom_Trigger.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "Custom_Trigger" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=nTrials, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('checkerboardTrials.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "Checkerboard" ---
        # create an object to store info about Routine Checkerboard
        Checkerboard = data.Routine(
            name='Checkerboard',
            components=[checker_mask, gray_screen],
        )
        Checkerboard.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code
        
        # Fetching parameters from Conditions csv file
        if stim_label == 'vertical':
            orientation = vertical_stim['img']
        elif stim_label == 'horizontal':
            orientation = horizontal_stim['img']
            
        #Initialize frame counter
        frame_counter = 0
        
        #direction_dict = {
        #    'right': [1 + t * -0.1, 0],
        #    'left': [-1 + t * 0.1, 0],
        #    'down': [0, 0.5 + t * -0.07],
        #    'up': [0, -0.5 + t * 0.07]
        #}
        checker_mask.setSize(size)
        checker_mask.setTex(orientation)
        # store start times for Checkerboard
        Checkerboard.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Checkerboard.tStart = globalClock.getTime(format='float')
        Checkerboard.status = STARTED
        thisExp.addData('Checkerboard.started', Checkerboard.tStart)
        Checkerboard.maxDuration = None
        # keep track of which components have finished
        CheckerboardComponents = Checkerboard.components
        for thisComponent in Checkerboard.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Checkerboard" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        Checkerboard.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code
            #need to change foreground color value from black to 
            #white at 6Hz
            
            # Update frame counter
            frame_counter += 1
            
            #Swap foreground color at 6Hz frequency
            if frame_counter % (color_alternating_freq * 2) < color_alternating_freq:
                checker_mask.color = 'white'
            else:
                checker_mask.color = 'black'
            
            # Fetching parameters from Conditions csv file
            if direction == 'right':
                p_x=(-1 + t * 0.1)
                p_y=0
            elif direction == 'left':
                p_x=(1 + t * -0.1)
                p_y=0
            elif direction == 'down':
                p_x=0
                p_y=(0.5 + t * -0.07)
            elif direction == 'up':
                p_x=0
                p_y=(-0.5 + t * 0.07)
            
            
            # *checker_mask* updates
            
            # if checker_mask is starting this frame...
            if checker_mask.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                checker_mask.frameNStart = frameN  # exact frame index
                checker_mask.tStart = t  # local t and not account for scr refresh
                checker_mask.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(checker_mask, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'checker_mask.started')
                # update status
                checker_mask.status = STARTED
                checker_mask.setAutoDraw(True)
            
            # if checker_mask is active this frame...
            if checker_mask.status == STARTED:
                # update params
                checker_mask.setPos((1 * p_x, 1 * p_y), log=False)
                checker_mask.setPhase(0.0, log=False)
            
            # if checker_mask is stopping this frame...
            if checker_mask.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > checker_mask.tStartRefresh + duration-frameTolerance:
                    # keep track of stop time/frame for later
                    checker_mask.tStop = t  # not accounting for scr refresh
                    checker_mask.tStopRefresh = tThisFlipGlobal  # on global time
                    checker_mask.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'checker_mask.stopped')
                    # update status
                    checker_mask.status = FINISHED
                    checker_mask.setAutoDraw(False)
            
            # *gray_screen* updates
            
            # if gray_screen is starting this frame...
            if gray_screen.status == NOT_STARTED and tThisFlip >= duration-frameTolerance:
                # keep track of start time/frame for later
                gray_screen.frameNStart = frameN  # exact frame index
                gray_screen.tStart = t  # local t and not account for scr refresh
                gray_screen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gray_screen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gray_screen.started')
                # update status
                gray_screen.status = STARTED
                gray_screen.setAutoDraw(True)
            
            # if gray_screen is active this frame...
            if gray_screen.status == STARTED:
                # update params
                pass
            
            # if gray_screen is stopping this frame...
            if gray_screen.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > gray_screen.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    gray_screen.tStop = t  # not accounting for scr refresh
                    gray_screen.tStopRefresh = tThisFlipGlobal  # on global time
                    gray_screen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'gray_screen.stopped')
                    # update status
                    gray_screen.status = FINISHED
                    gray_screen.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Checkerboard.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Checkerboard.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Checkerboard" ---
        for thisComponent in Checkerboard.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Checkerboard
        Checkerboard.tStop = globalClock.getTime(format='float')
        Checkerboard.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Checkerboard.stopped', Checkerboard.tStop)
        # Run 'End Routine' code from code
        orientation = None
        # the Routine "Checkerboard" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed nTrials repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # Run 'End Experiment' code from save_bids_format
    #=============================== Custom Codeblock jgronemeyer =====================================#
    # Sipelab standard BIDS protocol file naming for 
    #   future batch analysis and data wrangling
    protocol_id = expInfo['Protocol ID']
    subject_id = expInfo['Subject ID']
    session_id = expInfo['Session ID']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') # get current timestamp (BIDS)
    
    #save_to = u'data/%s/sub-%s/ses-%s/beh' % (expInfo['Protocol ID'], expInfo['Subject ID'], expInfo['Session ID'])
    save_to = f'data/{protocol_id}/sub-{subject_id}/ses-{session_id}/beh'
    new_dir = os.path.join(_thisDir, save_to)
    
    # Create the path if it does not exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    
    
    filename = os.path.join(new_dir, f"sub-{subject_id}_ses-{session_id}_{timestamp}_wheeldf.csv")
    
    if encoder_data is not None:
        encoder_data.to_csv(filename, index=False)
    else:
        print("No encoder data in `encorder_data`")
    #==================================================================================================#
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
