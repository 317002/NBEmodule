#!/usr/bin/python3
'''
My personal script for processing intf data to locate nbes

This is not part of the NBE package.

NBE(Narrow Bipolar Event) locating and clasification script

1 loads in raw files from local storage
2 attempts to locate an nbe from the sferic signal(channel D from the array)
3 Processes the data to aproximate where the sources of the VHF signals are
4 From the avg change in elevation over time of the sources and the initial
  deviation from the base line the sferic signal: determin the breakdown
  polarity of the NBE
'''


import re
import os
import gc
import sys
import datetime
import resource
import shutil

import numpy as np
import os.path as pat
import intfDataStore as store
import matplotlib.pyplot as plt

from collections import namedtuple
from shutil import copyfile
from matplotlib.offsetbox import AnchoredText



#file structure is allready sorted by event.
eventLocation = '/media/nathan/External/events'    #where the event folders are
perFileLoc = './perameters.txt'
figSaveBase = pat.join(os.getcwd(),'figs')      #where thg graphs will be saved

def isolateFilePaths(files):
    '''Orginizes the file paths via a dictionary object, where the key is that
    of each file's extension(all lower case though)

    usefull for manipulating the events


    :param files:
        list of unintelligently organized file directories
    :type files:
        :list:[strings]
    :return:
        dict object containing files paths keyed via their file exstention
    :rtype:
        :dict:{strings:strings}
    '''
    keys = [pat.splitext(c)[1][1:].lower() for c in files]     #file exstentions
    if len(set(keys)) != len(keys):
        raise Exception('More the one file with the same file exstention')

    return dict(zip(keys,files))

def plotHorzBar(ax,interval,level,color = 'r'):
    level = np.ones(100)*level
    start,end = interval if interval[0] <= interval[1] else (interval[1],interval[0])
    bar = np.linspace(start,end,level.size)
    ax.plot(bar,level,c = color)

def plotVertBar(ax,interval,level,color = 'r'):
    level = np.ones(100)*level
    start,end = interval if interval[0] <= interval[1] else (interval[1],interval[0])
    bar = np.linspace(start,end,level.size)
    ax.plot(level,bar,c = color)

def passCriteria(nbeProp,perameters,eventName = None):
    '''Takes in the generated properties of the detected nbe and returns
    boolean values coresponding to whether the script should look for another
    nbe canidate further into the event(the 100ms interval the intf array
    recored), or whether the event should be ignored all together.

    Essentially this contains the logic used to filter out events bassed on the
    sferic signal and the properties of detected nbe canidates

    :param nbeProp:
        Dict object contain various properties of the nbe canidate. Generated
        by intfDataStore.getNBE_TimeInterval()
    :type nbeProp:
        dict
    :param perameters:
        Dict object containt various quanties from processing perametes config
        file
    :type perameters:
        dict
    :param eventName:
        wether or not to log the skipped files, and it gives the name of the
        event to log the skip under
    :type eventName:
        string
    :return:
        boolean values in a nammed tupple, coresponding to whether the event
        should be skipped entirely or another canidate should be searched for
        further into the event
    :rtype:
        namedtuple(bool,bool)
    '''
    skip,lookFurther,logDetails = False,False,''

    #ifelse for looking further into the event or not
    if nbeProp['eventWidth'] < perameters['minEventWidth']:
        #duration of the event is too short
        logDetails = 'duration of the event is too short'
        lookFurther = True

    #ifelse for skipping the event or not

    if (eventName != None and (skip or lookFurther)):
        skipLogger(eventName,logDetails,skip)

    outputFormater = namedtuple('passCriteria',['skip','lookFurther'])

    return outputFormater(skip,lookFurther)

def skipLogger(eventName,reason,skip = False,success = False,clear = False):
    '''Used to log skipped events, and instances where the first detection was
    ignored
    '''
    logSaveLocation = pat.join(os.getcwd(),'skipLog.txt')
    if not success:
        if skip:
            logTxt = 'skipped: {0},reason: {1}\n'
            with open(logSaveLocation,'a+') as doc:
                doc.write(logTxt.format(eventName,reason))
        else:
            logTxt = '\tfurther: {0},reason: {1}\n'
            with open(logSaveLocation,'a+') as doc:
                doc.write(logTxt.format(eventName,reason))
    else:
        logTxt = 'NBE found: {0}\n'
        with open(logSaveLocation,'a+') as doc:
            doc.write(logTxt.format(eventName))
    if clear:
        '''clears out the skip log file'''
        with open(logSaveLocation,'w+') as doc:
            pass

def main():
    while True:
        ans = str(input('Clear the skip log?[y/n]: '))
        if ans == 'n':#if not
            print('Exit')
            sys.exit()#close out the script
        if ans == 'y':#if yes continue as normal
            #Clearing out the skip log
            skipLogger('','',clear = True)
            break

    #reading in processing parameters
    per = store.loadPerameters(perFileLoc)

    events = (pat.join(eventLocation,c) for c in os.listdir(eventLocation))

    for i,event in enumerate(events):
        n = 6
        if i != n:
            continue
        if i > n:
            break



        #used to name the event folder
        eventName = pat.split(event)[1]
        print(eventName)

        #dict object containing the files in the event folder, bassed on file extention
        files = isolateFilePaths([pat.join(event,c) for c in os.listdir(event)])

        #checking that the file size not over the max allowed value
        if pat.getsize(files["chd"])*1e-6 > per["maxFileSize"]:
            skipLogger(eventName,'file size too large',True)
            continue

        sferic = store.loadFA(files['chd'])
        #amplitude of the signal during the begining of the time interval
        initialAmplitude = sferic.max_of_a_section(0,\
                                           int(1/per['startingRegionFraction']))


        lookFurther,skip,noiseLevels,disp = False,False,False,0 #init
        #helps min the noise for finding the detection points
        windowAvgSferic = sferic.genWindowAvg(per['windowAvgSamples'])
        while True:
            #used to select the region to search for an NBE
            filterArray = np.arange(disp,sferic.t_vhf.size)
            nbeContents = store.getNBE_TimeInterval(sferic.filt(filterArray),\
                                                    windowAvgSferic.filt(filterArray),\
                                                    per['nbeRange'],\
                                                    per['numSections'],\
                                                    noiseLevels)

            #if no NBE could be found:skip event
            if nbeContents == 'No Event Was located':
                skipLogger(eventName,nbeContents,True)
                skip = True
                break
            elif nbeContents == 'No noise level found':
                skipLogger(eventName,nbeContents,True)
                skip = True
                break
            else:#  otherwise unpack like normal
                nbeInterval,nbeProp = nbeContents

            noiseLevels = nbeProp['noiseLevels']
            #Shift the search past the point in time the second detection is
            disp += nbeProp['second_detection_index']

            if initialAmplitude >= nbeProp['upperThreshold']:
                #activity in the begining -> want to skip this event
                    #could be missing info not captured in this recording
                skip = True
                skipLogger(eventName,'initial amplitude hit',True)

            '''##################Checking for pass criteria##################'''
            if not skip:
                skip,lookFurther = passCriteria(nbeProp,per,eventName)


            if skip:
                '''Means that for some reason a canidate, that meets the desired
                criteria, couldnt be found anywhere in the event. This results
                in the script moving on the the next event.
                '''
                break

            if lookFurther:
                '''This means the identified event, failed to meet the set
                criteria, but we want to look further into the event
                for other possible canidates
                '''
                continue


            # if the script made it this far: it passed
            #create a new sferic store obj that only covers the temporal
            #region coresponding to the potential NBE
            nbeSferic = sferic.timeSelect(nbeInterval)
            break
        else:
            skipLogger(eventName,'Failed to find NBE, that' + \
                                        'match the chosen perameters',True)
            skip = True



        if skip:
            '''Means that for some reason an event that meets the desired
            criteria couldnt be found anywhere in the event. This results in the
            script moving on the the next event.
            '''
            continue

        ######################Generating Sorce Locations########################

        #remove any source location files from other runs
        if 'gz' in files:
            os.remove(files['gz'])

        #generating the source locations data
        sourceInterval = store.sfericToSourcesInterval(nbeInterval,files['chd'])
        sources,triggerTime = \
                    store.generateSources(files['chd'],sourceInterval,perFileLoc)

        if not sources: #if there are no source location points
            skipLogger(eventName,'No source location data could be generated',True)
            continue


        ######################cleaing source location data######################

        #removing points below the horizon line
        sources.filt(np.where(sources.elev > 0))
        sources.filt(np.where(sources.ecls < per['max_ecls']))

        #matches the def of time with that of the sferic data
        sources = sources.convertToUTC(triggerTime)

        #making best fit of time V.S elev in between second and first dection
        lSquarInterval = (nbeProp['detectionTime'],nbeProp['secondDetectionTime'])
        #this is just the detection interval start and end point in ms
        lSquarInterval = [c*1e-3 for c in lSquarInterval]

        #The returned values will come back as None if there are no source data
            #Points in the interval(the two red vertical bars)
        m,b,fitTime,fitElev,avgDeviation,npoints = \
                            sources.leastSqauresInterval(lSquarInterval,1e3)

        if m == None:
            skipLogger(eventName,'No sources in desired time interval',True)
            continue

        #We ignore events that have too few points
        if npoints < 2:
            skipLogger(eventName,'Too few source location data points in' + \
                                            ' detection time interval',True)
            continue


        if avgDeviation > per['maxDeviation']:
            skipLogger(eventName,'over max source deviation',True)
            continue

        #storing the slope of the line in the properties dict --> file
        nbeProp['sourcesFitSlope'] = m
        #storing how many source location data points are in the interval(vertival bars)
        nbeProp['numSourcesInInterval'] = npoints
        #storing the net change in elevation of the source location best fit line
        nbeProp['sourceElevChange'] = fitElev[-1] - fitElev[0]

        #Determining the polarity of the best fit line slope and then the
            #break down polarity
        intfDisplacment = 'Negative' if m < 0 else 'Positive'
        breakDownPol    = 'Negative' if intfDisplacment == nbeProp['polarity']\
                                     else 'Positive'

        ################################Plotting################################

        sfericColor    = 'g'   #color of the sferic points in the graph
        sourceColor    = 'r'   #color of the source location points in the graph
        bestFitColor   = 'k'   #color of the best fit line
        windowAvgColor = 'b'   #color of the window avg array plot


        # +- 500 microseconds around the detection point
        zoomedSferic = sferic.timeSelect((nbeProp['detectionTime'] - 500,\
                                          nbeProp['detectionTime'] + 500))
        intervalSferic = sferic.timeSelect((nbeProp['detectionTime'],\
                                           nbeProp['secondDetectionTime']))

        #initalizing the figure
        fig =      plt.figure(figsize = (12,9))

        nbeAx =         fig.add_subplot(413)
        zoomedAx =      fig.add_subplot(412)
        eventAx =       fig.add_subplot(411)
        nbeIntervalAx = fig.add_subplot(414)



        #for the windows where we plot the source locations
        intervalSourceAx = plt.twinx(nbeIntervalAx)
        sourceAx =      plt.twinx(nbeAx)


        def check(a,b,small = True):
            '''
            reaturns the smaller value when small == True
            otherwise it returns the larger value
            '''
            if small:
                if a <= b:
                    return a
                else:
                    return b
            else:
                if a >= b:
                    return a
                else:
                    return b

        ##################Setting X and Y limits for the plots##################
        def offset(min_max,offsetFactor):
            invert = False if min_max[0] <= min_max[1] else True
            a,b = min_max if not invert else (min_max[1],min_max[0])

            offset = abs(b - a)*offsetFactor
            a -= offset
            b += offset
            if not invert:
                return (a,b)
            else:
                return(b,a)

        #Setting the bounds of the time axis for the sub plots
        nbe_xMinMax =    (np.min(nbeSferic.t_vhf),np.max(nbeSferic.t_vhf))
        zoomed_xMinMax = (zoomedSferic.t_vhf[0],np.max(zoomedSferic.t_vhf))
        sferic_xMinMax = (np.min(sferic.t_vhf),np.max(sferic.t_vhf))
        interval_xMinMax = (intervalSferic.t_vhf[0],intervalSferic.t_vhf[-1])

        offsetFactor = .1

        nbe_Ymin      = check(nbeProp['lowerThreshold'],np.min(nbeSferic.fa))
        zoomed_Ymin   = check(nbeProp['lowerThreshold'],np.min(zoomedSferic.fa))
        sferic_Ymin   = check(nbeProp['lowerThreshold'],np.min(sferic.fa[::12]))
        interval_Ymin = check(nbeProp['lowerThreshold'],np.min(intervalSferic.fa))

        nbe_Ymax =    check(nbeProp['upperThreshold'],np.max(nbeSferic.fa),False)
        zoomed_Ymax = check(nbeProp['upperThreshold'],np.max(zoomedSferic.fa),False)
        sferic_Ymax = check(nbeProp['upperThreshold'],np.max(sferic.fa[::12]),False)
        interval_Ymax = check(nbeProp['upperThreshold'],np.max(intervalSferic.fa),False)

        nbe_yMinMax =    offset((nbe_Ymin,nbe_Ymax),offsetFactor)
        zoomed_yMinMax = offset((zoomed_Ymin,zoomed_Ymax),offsetFactor)
        sferic_yMinMax = offset((sferic_Ymin,sferic_Ymax),offsetFactor)
        interval_yMinMax = offset((interval_Ymin,interval_Ymax),offsetFactor)
        intervalSource_yMinMax = sources.intervalYlim(lSquarInterval)

        #########################Creating the sub plots#########################

        nbeSferic.genSubPlot(nbeAx,nbe_xMinMax,nbe_yMinMax,\
                                        '',\
                                        r'$\Delta E\left[ \frac{V}{m}\right]$',\
                                        '')
        sferic.genSubPlot(eventAx,sferic_xMinMax,sferic_yMinMax,\
                                        '',
                                        r'$\Delta E\left[ \frac{V}{m}\right]$',\
                                        '{1} NBE[{0}]'.format(breakDownPol,eventName))
        zoomedSferic.genSubPlot(zoomedAx,zoomed_xMinMax,zoomed_yMinMax,\
                                        '',\
                                        r'$\Delta E\left[ \frac{V}{m}\right]$',\
                                        '')
        sources.genSubPlot(sourceAx,sources.time,sources.elev,\
                                        '',\
                                        'Elevation Angle(deg)',\
                                        ''\
                                        ,twinx = True)

        nbeSferic.genSubPlot(nbeIntervalAx,interval_xMinMax,interval_yMinMax,\
                                        r'Time($\mu s$)',\
                                        r'$\Delta E\left[ \frac{V}{m}\right]$',\
                                        '')

        sources.genSubPlot(intervalSourceAx,sources.time,sources.elev,\
                                        '',\
                                        'Elevation Angle(deg)',\
                                        ''\
                                        ,twinx = True)

        ############################Plotting the data###########################
        eventAx.scatter(sferic.t_vhf[::12],\
                            sferic.fa[::12],\
                            s = .4,\
                            c = sfericColor)

        nbeAx.scatter(nbeSferic.t_vhf,\
                       nbeSferic.fa,\
                       s = .4,\
                       c = sfericColor)

        windowAvgNbe = windowAvgSferic.timeSelect(nbeInterval)

        nbeAx.scatter(windowAvgNbe.t_vhf,\
                       windowAvgNbe.fa,\
                       s = .4,\
                       c = windowAvgColor)

        windowAvgZoomed = windowAvgSferic.timeSelect(zoomed_xMinMax)

        zoomedAx.scatter(windowAvgZoomed.t_vhf,\
                             windowAvgZoomed.fa,\
                             s = .4,\
                             c = windowAvgColor,
                             zorder = 100,\
                             alpha = .5)

        zoomedAx.scatter(zoomedSferic.t_vhf,\
                             zoomedSferic.fa,\
                             s = .4,\
                             c = sfericColor,
                             zorder = 99)


        sourceAx.scatter((sources.time)*1e3,\
                          sources.elev,\
                          s = 4,\
                          c = sourceColor)


        #plotting the best fit line
        sourceAx.plot(fitTime,fitElev,c = bestFitColor)
        intervalSourceAx.plot(fitTime,fitElev,c = bestFitColor)


        #Plotting the data in the nbe interval
        nbeIntervalAx.scatter(nbeSferic.t_vhf,\
                       nbeSferic.fa,\
                       s = .4,\
                       c = sfericColor)

        nbeIntervalAx.scatter(windowAvgNbe.t_vhf,\
                       windowAvgNbe.fa,\
                       s = .4,\
                       c = windowAvgColor)

        intervalSourceAx.scatter((sources.time)*1e3,\
                          sources.elev,\
                          s = 10,\
                          c = sourceColor)

        #setting the sources ylim
        intervalSourceAx.set_ylim(*intervalSource_yMinMax,.05)
        #setting the sources y-tick-marks
        intervalSourceYticks = np.linspace(*intervalSource_yMinMax,7).round(2)
        intervalSourceAx.set_yticks(intervalSourceYticks)
        # intervalSourceAx.grid()







        ############################Plotting the Bars###########################

        ####nbe section####
        plotHorzBar(nbeAx,(nbeSferic.t_vhf[0],nbeSferic.t_vhf[-1]),\
                    nbeProp['upperThreshold'],'b')
        plotHorzBar(nbeAx,(nbeSferic.t_vhf[0],nbeSferic.t_vhf[-1]),\
                    nbeProp['lowerThreshold'],'b')
        plotVertBar(nbeAx,nbe_yMinMax,
                    nbeProp['detectionTime'])
        plotVertBar(nbeAx,nbe_yMinMax,
                    nbeProp['secondDetectionTime'])

        nbeAx.vlines(nbe_xMinMax[0]+.3,*nbe_yMinMax,color = 'fuchsia')
        nbeAx.vlines(nbe_xMinMax[1]-.1,*nbe_yMinMax,color = 'fuchsia')
        ####zoomed in section####
        plotHorzBar(zoomedAx,(zoomedSferic.t_vhf[0],zoomedSferic.t_vhf[-1]),\
                    nbeProp['upperThreshold'],'b')
        plotHorzBar(zoomedAx,(zoomedSferic.t_vhf[0],zoomedSferic.t_vhf[-1]),\
                    nbeProp['lowerThreshold'],'b')
        plotVertBar(zoomedAx,zoomed_yMinMax,
                    nbeProp['detectionTime'])
        plotVertBar(zoomedAx,zoomed_yMinMax,
                    nbeProp['secondDetectionTime'])
        zoomedAx.vlines(zoomed_xMinMax[0]+2,*zoomed_yMinMax,color = 'k')
        zoomedAx.vlines(zoomed_xMinMax[1]-1,*zoomed_yMinMax,color = 'k')

        zoomedAx.vlines(nbe_xMinMax[0],*zoomed_yMinMax,color = 'fuchsia')
        zoomedAx.vlines(nbe_xMinMax[1],*zoomed_yMinMax,color = 'fuchsia')
        ####Main Section####
        plotVertBar(eventAx,sferic_yMinMax,\
                    nbeProp['detectionTime'])
        plotVertBar(eventAx,sferic_yMinMax,\
                    nbeProp['secondDetectionTime'])
        plotHorzBar(eventAx,(sferic.t_vhf[::12][0],sferic.t_vhf[::12][-1]),\
                    nbeProp['lowerThreshold'],'b')
        plotHorzBar(eventAx,(sferic.t_vhf[::12][0],sferic.t_vhf[::12][-1]),\
                    nbeProp['upperThreshold'],'b')
        #ilustrating the zoomed in section's placment in the full interval
        eventAx.vlines(zoomed_xMinMax[0],*sferic_yMinMax,color = 'k')
        eventAx.vlines(zoomed_xMinMax[1],*sferic_yMinMax,color = 'k')

        ####Interval Section####
        nbeIntervalAx.vlines(nbeProp['detectionTime'],\
                                     *interval_yMinMax,color = 'r',linewidths = 4)
        nbeIntervalAx.vlines(nbeProp['secondDetectionTime'],\
                                     *interval_yMinMax,color = 'r',linewidths = 3)



        ##########################Formating the layout##########################
        fig.tight_layout(h_pad = .2)
        #adding perameter text box
        perBoxStr = f"""Best-fit slope = ${{ {round(m,3)} \\frac{{\Delta deg}}{{\mu s}} }}$
        Best-fit Change in Elevation = ${{{round(nbeProp['sourceElevChange'],3)}^\circ}}$
        Number of Sources = {npoints}
        Average Source Deviation = {round(avgDeviation,3)}
        """
        perBoxStr = re.sub(f"  ","",perBoxStr)
        perBoxStr = perBoxStr[:-1]

        per_box = AnchoredText(perBoxStr,loc = 'lower right')
        nbeIntervalAx.add_artist(per_box)
        per_box.set_zorder(0)


        base = pat.join(figSaveBase,eventName)
        if not pat.isdir(base):
            os.makedirs(base)
        figSaveLocation = pat.join(base,\
             '{event}_[sfPol:{faPol}][soPol:{sPol}][{pol}].png'.format(\
                                                event = eventName,\
                                                faPol = nbeProp['polarity'],\
                                                sPol = intfDisplacment,\
                                                pol = breakDownPol))

        #saving the figure
        fig.savefig(figSaveLocation)
        gc.collect() #collecting mem used by matplotlib
        plt.show()
        plt.close('all')

        skipLogger(eventName,'',success= True)

        #saving the nbe properties dict object to a text file
        propSaveLocation = pat.join(base,'nbeProperties.txt')
        with open(propSaveLocation,'w') as doc:
            pass
        with open(propSaveLocation,'a+') as doc:
            txtF = '{0} = {1}\n'
            for key,value in zip(nbeProp,nbeProp.values()):
                doc.write(txtF.format(key,value))
            doc.write(txtF.format('avgDeviation',avgDeviation))

        #making a copy of the perameters file used in generating the file and
            #saving it to the event directory
        copyfile(perFileLoc,pat.join(base,pat.split(perFileLoc)[1]))






if __name__ == '__main__':
    main()
