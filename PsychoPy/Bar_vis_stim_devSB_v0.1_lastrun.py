#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.5),
    on November 13, 2024, at 17:42
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

# Run 'Before Experiment' code from checker_texture
import numpy as np
from stimupy.stimuli.checkerboards import checkerboard

stim = checkerboard(visual_size=(13,3), ppd=31, 
shape=None, frequency=0.5, board_shape=(13,3), 
check_visual_size=1, period='ignore', rotation=0, 
intensity_checks=(-1,1), intensity_target=1)


# Run 'Before Experiment' code from checker_texture_3
import numpy as np
from stimupy.stimuli.checkerboards import checkerboard

stim = checkerboard(visual_size=(13,3), ppd=31, 
shape=None, frequency=0.5, board_shape=(13,3), 
check_visual_size=1, period='ignore', rotation=0, 
intensity_checks=(-1,1), intensity_target=1)


# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.5'
expName = 'Bar_vis_stim_devSB_v0.1'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
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
_fullScr = False
_winSize = [1920, 1080]
_loggingLevel = logging.getLevel('warning')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

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
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\dev\\sipefield-gratings\\PsychoPy\\Bar_vis_stim_devSB_v0.1_lastrun.py',
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
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
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
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=1,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
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
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
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
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
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
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
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
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


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
    
    # --- Initialize components for Routine "checkerboard_mask" ---
    white_check = visual.GratingStim(
        win=win, name='white_check',
        tex=stim['img'], mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2,1), sf=None, phase=0.0,
        color='black', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-1.0)
    black_check_2 = visual.GratingStim(
        win=win, name='black_check_2',
        tex=stim['img'], mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2,1), sf=None, phase=0.0,
        color='white', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-2.0)
    white_check_2 = visual.GratingStim(
        win=win, name='white_check_2',
        tex=stim['img'], mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2,1), sf=None, phase=0.0,
        color='black', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-3.0)
    black_check_3 = visual.GratingStim(
        win=win, name='black_check_3',
        tex=stim['img'], mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2,1), sf=None, phase=0.0,
        color='white', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-4.0)
    white_check_3 = visual.GratingStim(
        win=win, name='white_check_3',
        tex=stim['img'], mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2,1), sf=None, phase=0.0,
        color='black', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-5.0)
    black_check_4 = visual.GratingStim(
        win=win, name='black_check_4',
        tex=stim['img'], mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2,1), sf=None, phase=0.0,
        color='white', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-6.0)
    white_check_4 = visual.GratingStim(
        win=win, name='white_check_4',
        tex=stim['img'], mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2,1), sf=None, phase=0.0,
        color='black', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-7.0)
    black_check_5 = visual.GratingStim(
        win=win, name='black_check_5',
        tex=stim['img'], mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2,1), sf=None, phase=0.0,
        color='white', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-8.0)
    white_check_5 = visual.GratingStim(
        win=win, name='white_check_5',
        tex=stim['img'], mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2,1), sf=None, phase=0.0,
        color='black', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-9.0)
    black_check_6 = visual.GratingStim(
        win=win, name='black_check_6',
        tex=stim['img'], mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2,1), sf=None, phase=0.0,
        color='white', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-10.0)
    
    # --- Initialize components for Routine "checkerboard_mask_2" ---
    black_check = visual.GratingStim(
        win=win, name='black_check',
        tex=stim['img'], mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2,1), sf=None, phase=0.0,
        color='white', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "horizontal_movement2" ---
    horlef = visual.Rect(
        win=win, name='horlef',
        width=(0.1, 1)[0], height=(0.1, 1)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=10.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=1.0, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "vertical_movement" ---
    vertical_down = visual.Rect(
        win=win, name='vertical_down',
        width=(2, 0.1)[0], height=(2, 0.1)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    gray_screen_2 = visual.ImageStim(
        win=win,
        name='gray_screen_2', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[0,0,0], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    vertical_up = visual.Rect(
        win=win, name='vertical_up',
        width=(2, 0.1)[0], height=(2, 0.1)[1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    
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
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=100.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "checkerboard_mask" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('checkerboard_mask.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        checkerboard_maskComponents = [white_check, black_check_2, white_check_2, black_check_3, white_check_3, black_check_4, white_check_4, black_check_5, white_check_5, black_check_6]
        for thisComponent in checkerboard_maskComponents:
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
        
        # --- Run Routine "checkerboard_mask" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 10.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from checker_texture
            #need to change foreground color value from black to 
            #white at 6Hz
            
            # *white_check* updates
            
            # if white_check is starting this frame...
            if white_check.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                white_check.frameNStart = frameN  # exact frame index
                white_check.tStart = t  # local t and not account for scr refresh
                white_check.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(white_check, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'white_check.started')
                # update status
                white_check.status = STARTED
                white_check.setAutoDraw(True)
            
            # if white_check is active this frame...
            if white_check.status == STARTED:
                # update params
                white_check.setPos((-1+t*0.1, 0), log=False)
            
            # if white_check is stopping this frame...
            if white_check.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > white_check.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    white_check.tStop = t  # not accounting for scr refresh
                    white_check.tStopRefresh = tThisFlipGlobal  # on global time
                    white_check.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'white_check.stopped')
                    # update status
                    white_check.status = FINISHED
                    white_check.setAutoDraw(False)
            
            # *black_check_2* updates
            
            # if black_check_2 is starting this frame...
            if black_check_2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
                # keep track of start time/frame for later
                black_check_2.frameNStart = frameN  # exact frame index
                black_check_2.tStart = t  # local t and not account for scr refresh
                black_check_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_check_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'black_check_2.started')
                # update status
                black_check_2.status = STARTED
                black_check_2.setAutoDraw(True)
            
            # if black_check_2 is active this frame...
            if black_check_2.status == STARTED:
                # update params
                black_check_2.setPos((-1+t*0.1, 0), log=False)
            
            # if black_check_2 is stopping this frame...
            if black_check_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black_check_2.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    black_check_2.tStop = t  # not accounting for scr refresh
                    black_check_2.tStopRefresh = tThisFlipGlobal  # on global time
                    black_check_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'black_check_2.stopped')
                    # update status
                    black_check_2.status = FINISHED
                    black_check_2.setAutoDraw(False)
            
            # *white_check_2* updates
            
            # if white_check_2 is starting this frame...
            if white_check_2.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                white_check_2.frameNStart = frameN  # exact frame index
                white_check_2.tStart = t  # local t and not account for scr refresh
                white_check_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(white_check_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'white_check_2.started')
                # update status
                white_check_2.status = STARTED
                white_check_2.setAutoDraw(True)
            
            # if white_check_2 is active this frame...
            if white_check_2.status == STARTED:
                # update params
                white_check_2.setPos((-1+t*0.1, 0), log=False)
            
            # if white_check_2 is stopping this frame...
            if white_check_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > white_check_2.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    white_check_2.tStop = t  # not accounting for scr refresh
                    white_check_2.tStopRefresh = tThisFlipGlobal  # on global time
                    white_check_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'white_check_2.stopped')
                    # update status
                    white_check_2.status = FINISHED
                    white_check_2.setAutoDraw(False)
            
            # *black_check_3* updates
            
            # if black_check_3 is starting this frame...
            if black_check_3.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                # keep track of start time/frame for later
                black_check_3.frameNStart = frameN  # exact frame index
                black_check_3.tStart = t  # local t and not account for scr refresh
                black_check_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_check_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'black_check_3.started')
                # update status
                black_check_3.status = STARTED
                black_check_3.setAutoDraw(True)
            
            # if black_check_3 is active this frame...
            if black_check_3.status == STARTED:
                # update params
                black_check_3.setPos((-1+t*0.1, 0), log=False)
            
            # if black_check_3 is stopping this frame...
            if black_check_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black_check_3.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    black_check_3.tStop = t  # not accounting for scr refresh
                    black_check_3.tStopRefresh = tThisFlipGlobal  # on global time
                    black_check_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'black_check_3.stopped')
                    # update status
                    black_check_3.status = FINISHED
                    black_check_3.setAutoDraw(False)
            
            # *white_check_3* updates
            
            # if white_check_3 is starting this frame...
            if white_check_3.status == NOT_STARTED and tThisFlip >= 4-frameTolerance:
                # keep track of start time/frame for later
                white_check_3.frameNStart = frameN  # exact frame index
                white_check_3.tStart = t  # local t and not account for scr refresh
                white_check_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(white_check_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'white_check_3.started')
                # update status
                white_check_3.status = STARTED
                white_check_3.setAutoDraw(True)
            
            # if white_check_3 is active this frame...
            if white_check_3.status == STARTED:
                # update params
                white_check_3.setPos((-1+t*0.1, 0), log=False)
            
            # if white_check_3 is stopping this frame...
            if white_check_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > white_check_3.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    white_check_3.tStop = t  # not accounting for scr refresh
                    white_check_3.tStopRefresh = tThisFlipGlobal  # on global time
                    white_check_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'white_check_3.stopped')
                    # update status
                    white_check_3.status = FINISHED
                    white_check_3.setAutoDraw(False)
            
            # *black_check_4* updates
            
            # if black_check_4 is starting this frame...
            if black_check_4.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
                # keep track of start time/frame for later
                black_check_4.frameNStart = frameN  # exact frame index
                black_check_4.tStart = t  # local t and not account for scr refresh
                black_check_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_check_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'black_check_4.started')
                # update status
                black_check_4.status = STARTED
                black_check_4.setAutoDraw(True)
            
            # if black_check_4 is active this frame...
            if black_check_4.status == STARTED:
                # update params
                black_check_4.setPos((-1+t*0.1, 0), log=False)
            
            # if black_check_4 is stopping this frame...
            if black_check_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black_check_4.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    black_check_4.tStop = t  # not accounting for scr refresh
                    black_check_4.tStopRefresh = tThisFlipGlobal  # on global time
                    black_check_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'black_check_4.stopped')
                    # update status
                    black_check_4.status = FINISHED
                    black_check_4.setAutoDraw(False)
            
            # *white_check_4* updates
            
            # if white_check_4 is starting this frame...
            if white_check_4.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
                # keep track of start time/frame for later
                white_check_4.frameNStart = frameN  # exact frame index
                white_check_4.tStart = t  # local t and not account for scr refresh
                white_check_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(white_check_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'white_check_4.started')
                # update status
                white_check_4.status = STARTED
                white_check_4.setAutoDraw(True)
            
            # if white_check_4 is active this frame...
            if white_check_4.status == STARTED:
                # update params
                white_check_4.setPos((-1+t*0.1, 0), log=False)
            
            # if white_check_4 is stopping this frame...
            if white_check_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > white_check_4.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    white_check_4.tStop = t  # not accounting for scr refresh
                    white_check_4.tStopRefresh = tThisFlipGlobal  # on global time
                    white_check_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'white_check_4.stopped')
                    # update status
                    white_check_4.status = FINISHED
                    white_check_4.setAutoDraw(False)
            
            # *black_check_5* updates
            
            # if black_check_5 is starting this frame...
            if black_check_5.status == NOT_STARTED and tThisFlip >= 7-frameTolerance:
                # keep track of start time/frame for later
                black_check_5.frameNStart = frameN  # exact frame index
                black_check_5.tStart = t  # local t and not account for scr refresh
                black_check_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_check_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'black_check_5.started')
                # update status
                black_check_5.status = STARTED
                black_check_5.setAutoDraw(True)
            
            # if black_check_5 is active this frame...
            if black_check_5.status == STARTED:
                # update params
                black_check_5.setPos((-1+t*0.1, 0), log=False)
            
            # if black_check_5 is stopping this frame...
            if black_check_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black_check_5.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    black_check_5.tStop = t  # not accounting for scr refresh
                    black_check_5.tStopRefresh = tThisFlipGlobal  # on global time
                    black_check_5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'black_check_5.stopped')
                    # update status
                    black_check_5.status = FINISHED
                    black_check_5.setAutoDraw(False)
            
            # *white_check_5* updates
            
            # if white_check_5 is starting this frame...
            if white_check_5.status == NOT_STARTED and tThisFlip >= 8-frameTolerance:
                # keep track of start time/frame for later
                white_check_5.frameNStart = frameN  # exact frame index
                white_check_5.tStart = t  # local t and not account for scr refresh
                white_check_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(white_check_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'white_check_5.started')
                # update status
                white_check_5.status = STARTED
                white_check_5.setAutoDraw(True)
            
            # if white_check_5 is active this frame...
            if white_check_5.status == STARTED:
                # update params
                white_check_5.setPos((-1+t*0.1, 0), log=False)
            
            # if white_check_5 is stopping this frame...
            if white_check_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > white_check_5.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    white_check_5.tStop = t  # not accounting for scr refresh
                    white_check_5.tStopRefresh = tThisFlipGlobal  # on global time
                    white_check_5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'white_check_5.stopped')
                    # update status
                    white_check_5.status = FINISHED
                    white_check_5.setAutoDraw(False)
            
            # *black_check_6* updates
            
            # if black_check_6 is starting this frame...
            if black_check_6.status == NOT_STARTED and tThisFlip >= 9-frameTolerance:
                # keep track of start time/frame for later
                black_check_6.frameNStart = frameN  # exact frame index
                black_check_6.tStart = t  # local t and not account for scr refresh
                black_check_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_check_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'black_check_6.started')
                # update status
                black_check_6.status = STARTED
                black_check_6.setAutoDraw(True)
            
            # if black_check_6 is active this frame...
            if black_check_6.status == STARTED:
                # update params
                black_check_6.setPos((-1+t*0.1, 0), log=False)
            
            # if black_check_6 is stopping this frame...
            if black_check_6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black_check_6.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    black_check_6.tStop = t  # not accounting for scr refresh
                    black_check_6.tStopRefresh = tThisFlipGlobal  # on global time
                    black_check_6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'black_check_6.stopped')
                    # update status
                    black_check_6.status = FINISHED
                    black_check_6.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in checkerboard_maskComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "checkerboard_mask" ---
        for thisComponent in checkerboard_maskComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('checkerboard_mask.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.000000)
        
        # --- Prepare to start Routine "checkerboard_mask_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('checkerboard_mask_2.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        checkerboard_mask_2Components = [black_check]
        for thisComponent in checkerboard_mask_2Components:
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
        
        # --- Run Routine "checkerboard_mask_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from checker_texture_3
            #need to change foreground color value from black to 
            #white at 6Hz
            
            # *black_check* updates
            
            # if black_check is starting this frame...
            if black_check.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                black_check.frameNStart = frameN  # exact frame index
                black_check.tStart = t  # local t and not account for scr refresh
                black_check.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_check, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'black_check.started')
                # update status
                black_check.status = STARTED
                black_check.setAutoDraw(True)
            
            # if black_check is active this frame...
            if black_check.status == STARTED:
                # update params
                black_check.setPos((-1+t*0.1, 0), log=False)
            
            # if black_check is stopping this frame...
            if black_check.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black_check.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    black_check.tStop = t  # not accounting for scr refresh
                    black_check.tStopRefresh = tThisFlipGlobal  # on global time
                    black_check.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'black_check.stopped')
                    # update status
                    black_check.status = FINISHED
                    black_check.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in checkerboard_mask_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "checkerboard_mask_2" ---
        for thisComponent in checkerboard_mask_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('checkerboard_mask_2.stopped', globalClock.getTime(format='float'))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
    # completed 100.0 repeats of 'trials'
    
    
    # --- Prepare to start Routine "horizontal_movement2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('horizontal_movement2.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    horizontal_movement2Components = [horlef]
    for thisComponent in horizontal_movement2Components:
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
    
    # --- Run Routine "horizontal_movement2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 20.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *horlef* updates
        
        # if horlef is starting this frame...
        if horlef.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            horlef.frameNStart = frameN  # exact frame index
            horlef.tStart = t  # local t and not account for scr refresh
            horlef.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(horlef, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'horlef.started')
            # update status
            horlef.status = STARTED
            horlef.setAutoDraw(True)
        
        # if horlef is active this frame...
        if horlef.status == STARTED:
            # update params
            horlef.setPos((1+t*-0.1, 0), log=False)
        
        # if horlef is stopping this frame...
        if horlef.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > horlef.tStartRefresh + 20-frameTolerance:
                # keep track of stop time/frame for later
                horlef.tStop = t  # not accounting for scr refresh
                horlef.tStopRefresh = tThisFlipGlobal  # on global time
                horlef.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'horlef.stopped')
                # update status
                horlef.status = FINISHED
                horlef.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in horizontal_movement2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "horizontal_movement2" ---
    for thisComponent in horizontal_movement2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('horizontal_movement2.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-20.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "vertical_movement" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('vertical_movement.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    vertical_movementComponents = [vertical_down, gray_screen_2, vertical_up]
    for thisComponent in vertical_movementComponents:
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
    
    # --- Run Routine "vertical_movement" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 29.799999999999997:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *vertical_down* updates
        
        # if vertical_down is starting this frame...
        if vertical_down.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            vertical_down.frameNStart = frameN  # exact frame index
            vertical_down.tStart = t  # local t and not account for scr refresh
            vertical_down.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(vertical_down, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'vertical_down.started')
            # update status
            vertical_down.status = STARTED
            vertical_down.setAutoDraw(True)
        
        # if vertical_down is active this frame...
        if vertical_down.status == STARTED:
            # update params
            vertical_down.setPos((0, 0.5+t*-0.07), log=False)
        
        # if vertical_down is stopping this frame...
        if vertical_down.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > vertical_down.tStartRefresh + 13.4-frameTolerance:
                # keep track of stop time/frame for later
                vertical_down.tStop = t  # not accounting for scr refresh
                vertical_down.tStopRefresh = tThisFlipGlobal  # on global time
                vertical_down.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'vertical_down.stopped')
                # update status
                vertical_down.status = FINISHED
                vertical_down.setAutoDraw(False)
        
        # *gray_screen_2* updates
        
        # if gray_screen_2 is starting this frame...
        if gray_screen_2.status == NOT_STARTED and tThisFlip >= 13.4-frameTolerance:
            # keep track of start time/frame for later
            gray_screen_2.frameNStart = frameN  # exact frame index
            gray_screen_2.tStart = t  # local t and not account for scr refresh
            gray_screen_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(gray_screen_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'gray_screen_2.started')
            # update status
            gray_screen_2.status = STARTED
            gray_screen_2.setAutoDraw(True)
        
        # if gray_screen_2 is active this frame...
        if gray_screen_2.status == STARTED:
            # update params
            pass
        
        # if gray_screen_2 is stopping this frame...
        if gray_screen_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > gray_screen_2.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                gray_screen_2.tStop = t  # not accounting for scr refresh
                gray_screen_2.tStopRefresh = tThisFlipGlobal  # on global time
                gray_screen_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gray_screen_2.stopped')
                # update status
                gray_screen_2.status = FINISHED
                gray_screen_2.setAutoDraw(False)
        
        # *vertical_up* updates
        
        # if vertical_up is starting this frame...
        if vertical_up.status == NOT_STARTED and tThisFlip >= 16.4-frameTolerance:
            # keep track of start time/frame for later
            vertical_up.frameNStart = frameN  # exact frame index
            vertical_up.tStart = t  # local t and not account for scr refresh
            vertical_up.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(vertical_up, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'vertical_up.started')
            # update status
            vertical_up.status = STARTED
            vertical_up.setAutoDraw(True)
        
        # if vertical_up is active this frame...
        if vertical_up.status == STARTED:
            # update params
            vertical_up.setPos((0, -2+t*0.07), log=False)
        
        # if vertical_up is stopping this frame...
        if vertical_up.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > vertical_up.tStartRefresh + 13.4-frameTolerance:
                # keep track of stop time/frame for later
                vertical_up.tStop = t  # not accounting for scr refresh
                vertical_up.tStopRefresh = tThisFlipGlobal  # on global time
                vertical_up.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'vertical_up.stopped')
                # update status
                vertical_up.status = FINISHED
                vertical_up.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in vertical_movementComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "vertical_movement" ---
    for thisComponent in vertical_movementComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('vertical_movement.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-29.800000)
    thisExp.nextEntry()
    
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
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
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
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
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
