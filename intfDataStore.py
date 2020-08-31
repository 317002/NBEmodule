'''
Module that provides class objects for the affective storage of the
sferic(change in E-field) signal and the generated sources' location data.

Also provides tools for basic manipulation of data,generation of source locations
data, methodes for autonomously identifying NBE from the sferic wave form and a
few other ease of life functions
'''

'''
I cant stress this enough. The unit time for the sources data is in MILISECONDS,
while the unit time for the sferic signal is in MICROSECONDS

I've been burned by this an embarrassing number of times
'''

import numpy as np
import pandas as pd
import os.path as pat

import os
import gzip
import datetime
import subprocess #needed for integrating python-2 code.(The generation of
                  #source location data)

from collections import namedtuple

####Custom Imports####
import mas_raw




class sfericStore:
    '''Object that stores and can take measurments of the contents of the sferic
    channel from the INTF array
    '''

    def __init__(self,t_vhf,fa):
        '''init

        :param t_vhf:
            time array of the sferic data
        :type t_vhf:
            :class:'numpy.ndarray'
        :param fa:
            change in electric field data from the sferic channel
        :type fa:
            :class:'numpy.ndarray'
        '''

        self.t_vhf = t_vhf
        self.fa = fa


        self.lower_noise_level = None
        self.upper_noise_level = None
        self.hit_time = None
        self.safety_factor = None
        self.recur = 0

    def filt(self,points):

        #returning a sfericStore object now filtered
        return sfericStore(self.t_vhf[points],self.fa[points])

    def timeSelect(self,rang):
        '''Returns the same sfericStore object except the value of t_vhf and fa
        are filtered such that they are within the time interval defined by
        rang

        :param rang:
            The time interval used to filter the arrays
        :type rang:
            tuple
        :return:
            sfericStore object filtered out by the time interval
        :rtype:
            :class:'sfericStore'
        '''
        start,end = (rang[0],rang[1]) if rang[0] < rang[1] else (rang[1],rang[0])
        #selecting the indices that are withing the interval
        indicies = np.where(((self.t_vhf >= start) & (self.t_vhf <= end)))

        return sfericStore(self.t_vhf[indicies],self.fa[indicies])

    '''################NEEDS WORK##########################'''
    def genSubPlot(self,ax,\
                    xMinMax,\
                    yMinMax,\
                    xlabel = None,\
                    ylabel = None,\
                    title = None,\
                    nXticks = 10,\
                    nYticks = 6):
        '''Formats and then plots the sferic onto the matplotlib subplot object

        :param ax:
            subplot object where the data will be plotted and where the
            the formating will be done
        :type ax:
            :class:'matplotlib.axes._subplots.AxesSubplot'
        :param xlabel:
            label for the x axis
        :type xlabel:
            string
        :param ylabel:
            label for the y axis
        :type ylabel:
            string
        :param title:
            Tile for the subplot
        :type title:
            string
        :param nXticks:
            The number of tick marks that should be on the x axis
        :type nXticks:
            int
        :param nYticks:
            The number of tick marks that should be on the Y axis
        :type nYticks:
            int

        :return:
            The reformated axis object
        :rtype:
            :class:'matplotlib.axes._subplots.AxesSubplot'
        '''
        def check(a,b):
            if a<b:
                return (a,b)
            else:
                return (b,a)
        xmin,xmax = check(*xMinMax)
        ymin,ymax = check(*yMinMax)
        #label and values for the x axix/time axis
        xTickValues = np.linspace(xmin,xmax,nXticks)
        xTickValues = np.round(xTickValues,5)

        yTickValues = np.linspace(ymin,ymax,nYticks)
        yTickValues = np.round(yTickValues,2)

        #labeling the graph
        if title != None:
            ax.set_title(title,fontsize = 18)
        if xlabel != None:
            ax.set_xlabel(xlabel,fontsize = 14)
        if ylabel != None:
            ax.set_ylabel(ylabel,fontsize = 14)

        #formating the axis
        ax.set_xticks(xTickValues)
        ax.set_yticks(yTickValues)

        ax.set_xlim(xTickValues[0],xTickValues[-1])
        ax.set_ylim(yTickValues[0],yTickValues[-1])


    def max_of_a_section(self,n,numSections):
        '''Gives the max of the n-th section of a 1-d arrays

        array: class:'numpy.ndarray'
            The array where the max of section will be found
        n : int
            The index of the section where the max will be calculated
        numSections : int
            The total number of sections the array should be split into
        return : float
            The max amplitude of the array in the nth section
        '''

        ####quick retrofit####
        array = self.fa

        if n > numSections:
            raise Exception('n > total number of sections: n > numSections')

        #defing num of indicies in 1 section
        secSize = int(array.size/numSections)

        if n == numSections:
            #handles that the array cant be evenly split
            sMax = np.max(array[(n - 1)*secSize:])
        else:
            n+=1
            sMax = np.max(array[(n - 1)*secSize:n*secSize])

        return sMax

    def min_of_a_section(self,n,numSections):
        '''Gives the min of the n-th section of a 1-d arrays

        array: class:'numpy.ndarray'
            The array where the min of section will be found
        n : int
            The index of the section where the min will be calculated
        numSections : int
            The total number of sections the array should be split into
        return : float
            The min amplitude of the array in the nth section
        '''
        array = self.fa

        if n > numSections:
            raise Exception('n > total number of sections: n > numSections')

        #defing num of indicies in 1 section
        secSize = int(array.size/numSections)

        if n == numSections:
            #handles that the array cant be evenly split
            sMin = np.min(array[(n - 1)*secSize:])
        else:
            n+=1
            sMin = np.min(array[(n - 1)*secSize:n*secSize])

        return sMin

    def getNoiseLevels(self,numSections):
        '''Aproximates the upper and lower level of the noise baseline

        :param array:
            Input array noise calculation is done on
        :type array:
            :class:'numpy.ndarray'
        :param numSections:
            Number of sections to split the time interval into when estimating
            the noise level
                *The smaller number of sections: the more confident the estimantion
                *The smaller number of sections: the more likely a noise level
                 cant be predicted with that level of confidence.
        :type numSections:
            int
        :param tolerence:
            The max allowed percentage difference between the two sections used
            to predict the noise level
        :type tolerence:
            float
        :param firstFraction:
            When a given, defines the size of the region when Calculating
            the amplitude of the signal during the begining of the signal
        :type firstFraction:
            float
        '''

        array = self.fa


        ####init####
        tolerence = .01
        lr_max,rl_max = None,None

        def diff_measurment(a,b):
            '''Calculates the relative difference between values a and b
            returns the fractional difference from a to b

            a:float
            b:float
            '''
            #handling discontinuity at b = 0
            if b != 0:
                return np.abs((a-b)/b)
            else:
                return a



        while tolerence  <= .99:
            #left to right
            check = False
            for n in range(numSections):
                for m in range(numSections):

                    if m != n:#dont compare the same section
                        #the max of two sections
                        n_max = self.max_of_a_section(n,numSections)
                        m_max = self.max_of_a_section(m,numSections)
                        n_min = self.min_of_a_section(n,numSections)
                        m_min = self.min_of_a_section(m,numSections)

                        #checking if both sets of values are within a tolerence
                        cond = (diff_measurment(n_max,m_max) < tolerence) and \
                                    (diff_measurment(n_min,m_min) < tolerence)
                        if cond:
                            #when the sections are within some tolerence, this
                            #common value is highly unlikley to be the product of
                            #signal activity

                            lr_max = n_max if n_max > m_max else m_max
                            lr_min = n_min if n_min < m_min else m_min
                            check = True
                            break
                if check:
                    break
            if lr_max == None:
                tolerence += .01
            else:
                break
        else:
            #means no noise level could be found
            return None,None

        #re-init
        tolerence = .01
        while tolerence <= .99:
            #right to left
            check = False
            for n in range(-numSections,0):
                n = -n
                for m in range(-numSections,0):
                    m = -m

                    if m != n:#dont compare the same section
                        #max of the two sections
                        n_max = self.max_of_a_section(n,numSections)
                        m_max = self.max_of_a_section(m,numSections)
                        n_min = self.min_of_a_section(n,numSections)
                        m_min = self.min_of_a_section(m,numSections)

                        #checking if both sets of values are within a tolerence
                        cond = (diff_measurment(n_max,m_max) < tolerence) and \
                                    (diff_measurment(n_min,m_min) < tolerence)
                        if cond:
                            #when the sections are within some tolerence, this
                            #common value is highly unlikley to be the product of
                            #signal activity
                            rl_max = n_max if n_max > m_max else m_max
                            rl_min = n_min if n_min < m_min else m_min
                            check = True
                            break
                if check:
                    break
            if rl_max == None:
                tolerence += .01
            else:
                break
        else:
            #means a noise level couldnt be found.
            return None,None


        upperNoiseLevel = lr_max if lr_max <= rl_max else rl_max
        lowerNoiseLevel = lr_min if lr_min >= rl_min else rl_min
        outputFormater = namedtuple('noiseLevels',\
                                        ['lowerNoiseLevel','upperNoiseLevel'])

        return outputFormater(lowerNoiseLevel,upperNoiseLevel)

class sourcesStore:
    '''Object for storing and processing generated vhf/hf sources data
    '''


    def __init__(self,time,azi,elev,cosa,cosb,pk2pk,rms,expk,ecls\
                ,estd,emlt,red,green,blue,startsample,imax):
        '''
        :param *data:
            The output from the loadSources function
        :type data:
            many numpy arrays



        '''
        self.time = time#mili Seconds
        self.azi = azi#degrees
        self.elev = elev#degrees
        self.cosa = cosa
        self.cosb = cosb
        self.pk2pk = pk2pk
        self.rms = rms
        self.expk = expk
        self.ecls = ecls#The Confidence value?
        self.estf = estd
        self.emlt = emlt
        self.red = red
        self.green = green
        self.blue = blue
        self.startsample = startsample
        self.imax = imax

        #used for returning an altered state
        self.pack = (c for c in [self.time,self.azi,self.elev,self.cosa,\
                                self.cosb,self.pk2pk,self.rms,self.expk,\
                                self.ecls,self.estf,self.emlt,self.red,\
                                self.green,self.blue,self.startsample,\
                                self.imax])

    def convertToUTC(self,triggerTime):
        '''Simple conversion to utc time by adding the trigger time to the time
        array

        :param triggerTime:
            The trigger time row entry from the sources file header
        :type triggerTime:
            string
        '''
        # "#TriggerTime        :'2016/08/24 00:03:30.822972'\n"
        #isolating the time from the string
        triggerTime = triggerTime.split(':')[1:]
        triggerTime[0] = triggerTime[0][-2:]
        triggerTime[-1] = triggerTime[2][:-4]
        triggerTime = [float(c) for c in triggerTime]

        #getting the trigger time in seconds
        triggerTime = triggerTime[0]*3600 + triggerTime[1]*60 + triggerTime[2]
        triggerTime = triggerTime*1e3

        #returning modified sourcesStore object.
        # print([c[0] + triggerTime if i == 0 else c[0] for i,c in enumerate(self.pack)])
        return sourcesStore(*[c + triggerTime if i == 0 else c for i,c in enumerate(self.pack)])

    def leastSqauresInInterval(self,timeInterval,cFactor):
        '''Gives linear least squares solution for the time verses elevation
        angle plot within the given time interval

        :param timeInterval:
            The time interval where the linear least squares solution will be
            found

            When set to None, the entire available set of data is used
        :type timeInterval:
            Tuple
                (float,float)
        '''

        if timeInterval == None:
            time,elev = self.time,self.elev
        else:
            start,end = timeInterval if timeInterval[0] < timeInterval[1]\
                                    else (timeInterval[1],timeInterval[0])
            #the indicies coresponding to the set of data that is within the
            #time interval
            indicies = np.where((self.time >= start) & (self.time <= end))[0]
            if indicies.size == 0:
                #means there were no source locations in the interval we care
                #about
                return (None,None,None,None,None)
            #selecting the data points within the the interval
            time = self.time[indicies]*cFactor
            offset = time[0]
            time -= offset
            elev = self.elev[indicies]


        A = np.vstack([time,np.ones(len(time))]).T
        m, c = np.linalg.lstsq(A, elev, rcond=None)[0]

        fitFunction = lambda x:x*m + c
        avgDeviation = np.sqrt((elev - fitFunction(time))**2)
        avgDeviation = np.average(avgDeviation)

        outputFormater = namedtuple('leastSqaures',\
        ['slope','y_intercept','fitTime','fitElev','avgDeviation'])
        t = np.linspace(time[0],time[-1],20)
        # time*m + c
        return outputFormater(m,c,t + offset,fitFunction(t),avgDeviation)

    def genSubPlot(self,ax,x,y,xlabel = None,ylabel = None,title = None\
                    ,nXticks = 10\
                    ,nYticks = 6\
                    ,yMinMax = None,
                    twinx = False):
        '''Formats and then plots the sferic onto the matplotlib subplot object

        :param ax:
            subplot object where the data will be plotted and where the
            the formating will be done
        :type ax:
            :class:'matplotlib.axes._subplots.AxesSubplot'
        :param xlabel:
            label for the x axis
        :type xlabel:
            string
        :param ylabel:
            label for the y axis
        :type ylabel:
            string
        :param title:
            Tile for the subplot
        :type title:
            string
        :param nXticks:
            The number of tick marks that should be on the x axis
        :type nXticks:
            int
        :param nYticks:
            The number of tick marks that should be on the Y axis
        :type nYticks:
            int
        :param yMinMax:
            The range of values the y axis should cover. Defaults to the min and
            max values of the fa array
        :type yMinMax:
            :tuple:(float,float)
        :return:
            The reformated axis object
        :rtype:
            :class:'matplotlib.axes._subplots.AxesSubplot'
        '''
        #label and values for the x axix/time axis
        xTickValues = np.linspace(0,x[-1] - x[0],nXticks) + x[0]
        xTickValues = np.round(xTickValues,5)
        #Determining the range of the yaxis
        if yMinMax == None:
            yMin,yMax = np.min(y),np.max(y)
        else:
            yMin,yMax = yMinMax if yMinMax[0] < yMinMax[1] \
                                                    else yMinMax[1],yMinMax[0]
        yTickValues = np.linspace(yMin,yMax,nYticks)
        yTickValues = np.round(yTickValues,2)

        #labeling the graph
        if title != None:
            ax.set_title(title,fontsize = 18)
        if xlabel != None:
            ax.set_xlabel(xlabel,fontsize = 14)
        if ylabel != None:
            ax.set_ylabel(ylabel,fontsize = 14)

        #formating the axis
        if twinx == False:
            ax.set_xticks(xTickValues)
            ax.set_xlim(xTickValues[0],xTickValues[-1])

        ax.set_yticks(yTickValues)
        ax.set_ylim(yTickValues[0],yTickValues[-1])

    def filt(self,points):
        self.time = self.time[points]
        self.azi = self.azi[points]
        self.elev = self.elev[points]
        self.cosa = self.cosa[points]
        self.cosb = self.cosb[points]
        self.pk2pk = self.pk2pk[points]
        self.rms = self.rms[points]
        self.expk = self.expk[points]
        self.ecls = self.ecls[points]
        self.estf = self.estf[points]
        self.emlt = self.emlt[points]
        self.red = self.red[points]
        self.green = self.green[points]
        self.blue = self.blue[points]
        self.startsample = self.startsample[points]
        self.imax = self.imax[points]

    def clean(self,ecls = 5):
        '''Cleans up the sorces generation by removing points below the horizon
        line(elev < 0 deg) a it also removes and data points that have a
        coresponding ecls value less then the ecls perameter

        !!!!the ecls value goes from 0 - 5!!!!

        :param ecls:
            The max allowed ecls value for the data set
        :type ecls:
            float
        '''
        #removing data with an ecls less then the perameter
        self.filt(np.where(self.ecls < ecls))
        #removing data coresponding to an elevation angle less the 0 deg
        self.filt(np.where(self.elev < 0))
#code by Mark Stanley
def get_date_time_from_filename(filename, maxExt=4 ):
    """atvt.get_date_time_from_filename( filename, maxExt=4 ):

    Returns a (date,time) tuple where date is a datetime.date object and time is a 64-bit
    float corresponding to the UNCORRECTED trigger time in seconds since midnight. The time
    may be significantly offset from UT (>1 sec) when this code was written (Nov 17, 2015).
    If the date can not be determined, (None, None) is returned.  If only the time can not
    be determined, then (date, None) is returned.

    The file name is assumed to be of the form:  PRE_YYYY.MM.DD_HH-MM-SS_uuuuuu.EXT

    OPTIONS:
        maxExt          The maximum length of the extension.  If this code detects a longer
                         extension, it will assume that the extension is actually missing
    """

    # Get file name and split into components
    filePath, fileNameExt = os.path.split( filename )
    fileName, fileExt     = os.path.splitext( fileNameExt )
    if len( fileExt ) > maxExt:
        # Extension is longer than expected.  Assume extension is missing
        fileName = fileNameExt
    fileList = fileName.split('_')

    # If there are not a sufficient number of components, exit
    if len(fileList) < 4:
        return  (None, None)

    # Extract date and time components
    dateTxt = fileList[1]
    timeTxt = fileList[2]
    usecTxt = fileList[3]

    # Split date string into components.  Exit if not valid
    dateList = dateTxt.split('.')
    if len(dateList) < 3:
        return  (None, None)
    try:
        year  = int( dateList[0] )
        month = int( dateList[1] )
        day   = int( dateList[2] )
    except:
        return  (None, None)
    else:
        date = datetime.date( year, month, day )

    # Split time string into components.  Exit if not valid
    timeList = timeTxt.split('-')
    if len(timeList) < 3:
        return  (date, None)
    try:
        hour   = int( timeList[0] )
        minute = int( timeList[1] )
        second = int( timeList[2] )
    except:
        return  (date, None)
    else:
        daySec = hour*3600. + minute*60. + second

    # Determine microsecond component of time
    try:
        subSec = float( usecTxt ) / 1e6
    except:
        return  (date, None)
    else:
        daySec += subSec

    return  (date, daySec)

def loadFA(fileLocation):
    '''Reads data from the VHF and HF sferic channel of the INTF array and
    returns it in a fa data object

    :param fileLocation:
        The file path to the sferic channelS of the intf array
    :type fileLocation:
        string
    :return:
        sfericStore object
    :rtype:
        :class:'sfericStore'


    !!!!!!!!!!!!!!!!!!!Time data is in units of microseconds!!!!!!!!!!!!!!!!!!!!
    '''

    #loading in the sferic data from disk
    sferic_file_handler = mas_raw.atvt()
    sferic_file_handler.load_file(fileLocation)

    ####isolating specific arrays from the sferc handler####
    t_vhf = np.array(sferic_file_handler.t_i)
    t_vhf = t_vhf*1e6               #converting to microseconds from midnight

    fa = np.array(sferic_file_handler.raw_i)
    fa = fa - 2.0**15               # calibration (linear)
    fa_calibConst = -160.0/(2**15)	#160 V/m per 2^15 digital units (bits).

    fa = fa_calibConst*fa

    return sfericStore(t_vhf,fa)

def loadSources(fileLocation,uncompressed = False):
    '''Loads in radio sources data from disk

    :param fileLocation:
        The file path to the sources data on disk
    :type fileLocation:
        string
    :param uncompressed:
        Whether or not the file is uncompressed from the original .gz state.
    :type uncompressed:
        bool
    :return:
        sourcesStore object for the data
    :rtype:
        :class:'sourcesStore'

    !!!!!!!!!!!!!!!!!!!!Time data is in units of miliseconds!!!!!!!!!!!!!!!!!!!!
    '''


    '''
    There hast to be a better way to do this
    '''
    if uncompressed == True:
        with open(fileLocation) as doc:
            for i,line in enumerate(doc):
                if i == 1:
                    triggerTime = str(line)
                    break


        data = np.loadtxt(fileLocation).transpose()
    else:
        with gzip.open(fileLocation) as doc:
            for i,line in enumerate(doc):
                if i == 1:
                    triggerTime = str(line)
                    break

        data = np.array(pd.read_csv(fileLocation\
                        ,skiprows= range(0,44)\
                        ,delimiter = ' '\
                        ,skipinitialspace = True\
                        ,skip_blank_lines = True)).transpose()

    outputFormater = namedtuple('sourcesData',['dataObject','triggerTime'])
    return outputFormater(sourcesStore(*data),triggerTime)

def getNBE_TimeInterval(sferic,safetyFactor,nbe_range,numSections,noiseLevels = False):
    '''attempts to locate an NBE waveform in the sferic data set

    safety_factor : float
        The factor to multiply Pthe estimated noise levels by to generate
        values used as the detection threshold.
    nbe_range : tuple (start,end)
        -Defines the time interval the nbe will ocupi. Centered around
        the detection point, the nbe is defined as ocupying the range:
        [-nbe_range[0],nbe_range[1]]
        -usally a [-30,100] microsecond range suffices
    numSections : int
        Number of sections to split the time interval into when estimating
        the noise level
            *The smaller number of sections: the more confident the estimantion
            *The smaller number of sections: the more likely a noise level
             cant be predicted with that level of confidence.\
    :return:
        The start and end time stamps of a predicted NBE, along with some
        properties in a dict format

        ((nbe start,nbe end),properties)
    :rtype:
        namedtuple(tuple(float,float),dict)
    '''
    if not noiseLevels:     #when a noise level is not provided
        #measuring the noise level of the sferic array
        lowerNoiseLevel,upperNoiseLevel = sferic.getNoiseLevels(numSections)
    else:       #For injecting your own noise to remove the need for recomputation
        lowerNoiseLevel,upperNoiseLevel = noiseLevels


    #means a noise level couldnt be found and therefor neither can an nbe
    if lowerNoiseLevel == None:
        #stops to func before more is done
        return 'No noise level found'

    #How much to deviate the noise levels as to not acidently 'scrap' the
    #noise floor when searching for a signal
    safetyFactor = abs(upperNoiseLevel - lowerNoiseLevel)*safetyFactor

    upperThreshold = upperNoiseLevel + safetyFactor
    lowerThreshold = lowerNoiseLevel - safetyFactor

    ####Locating Detection Index####
    try:
        #Index of the first detection using the upper threshold
        upper_index = np.where(sferic.fa >= upperThreshold)[0][0]
    except IndexError:#means no detection
        upper_index = None

    try:
        #Index of the first detection using the lower threshold
        lower_index = np.where(sferic.fa <= lowerThreshold)[0][0]
    except IndexError:#means no detection
        lower_index = None

    #checking that an event was found
    if upper_index == None and lower_index == None:
        #stops to func before more is done
        return 'No Event Was located'


    if (upper_index != None) and (lower_index != None):    #two detections

        if lower_index <= upper_index:  #select the smaller index value
            detection_index,sfericPolarity = (lower_index,'Negative')
        else:
            detection_index,sfericPolarity = (upper_index,'Positive')

    elif upper_index != None:   #pos detection
        detection_index,sfericPolarity = (upper_index,'Positive')
    elif lower_index != None:   #upper index must have a value
        detection_index,sfericPolarity = (lower_index,'Negative')


    #getting the time stamp for the nbe detection for the detected nbe
    detection_time = sferic.t_vhf[detection_index]

    #defining the time range of the NBE
    nbeStartTime = detection_time + nbe_range[0]
    nbeEndTime = detection_time + nbe_range[1]

    ####finding the second time the signal crosses the threashold####

    #finding the index value in the time array, that is atleast 1 microsecond
    #after the detection time
    offset = np.where(sferic.t_vhf >= detection_time + 1)[0][0]



    if sfericPolarity == 'Positive':
        #finding when the signal crosses below the upper threshold,
        #1 microsecond affter the detection_time
        second_detection_index = np.where(sferic.fa[offset:] \
                                            <= upperThreshold)[0][0]
        #re-centering the index value to that of the whole fa array
        second_detection_index += offset
    elif sfericPolarity == 'Negative':
        #finding when the signal crosses below the upper threshold,
        #1 microsecond affter the detection_time
        second_detection_index = np.where(sferic.fa[offset:] \
                                            >= lowerThreshold)[0][0]
        #re-centering the index value to that of the whole fa array
        second_detection_index += offset
    else:
        raise Exception('How?')

    second_detection_time = sferic.t_vhf[second_detection_index]

    #Building Properties Dictionary
    noiseFormater = namedtuple('noiseLevels',['lowerNoiseLevel','upperNoiseLevel'])

    nbeProp = {}
    nbeProp['lowerThreshold'] = lowerThreshold
    nbeProp['upperThreshold'] = upperThreshold
    nbeProp['noiseLevels'] = noiseFormater(lowerNoiseLevel,upperNoiseLevel)
    nbeProp['detectionTime'] = detection_time
    nbeProp['secondDetectionTime'] = second_detection_time
    nbeProp['second_detection_index'] = second_detection_index
    nbeProp['eventWidth'] = second_detection_time - detection_time
    nbeProp['polarity'] = sfericPolarity


    outputFormater = namedtuple('NBE',\
                                ['NBE_TimeInterval',\
                                'NBE_Properties'])

    return outputFormater((nbeStartTime,nbeEndTime),nbeProp)

def loadPerameters(procPerameters):
    ####Loading in processing perameters####
    def conversionFunc(a):
        try:
            return float(a)
        except ValueError:
            #hit tupple perameter
            if ('(' in a and ')' in a):
                a = a.strip().split(',')
                return (float(a[0][1:]),float(a[1][:-1]))
            else:#string perameter
                return a


    with open(procPerameters,'r') as doc:
        for i,line in enumerate(doc):
            #iterating over the file untill we are past the header
            if str(line).rstrip() == '##########':
                break
        per = pd.read_csv(doc,\
                                delimiter = '=',\
                                header = None,\
                                converters={1:conversionFunc})
    output = dict(zip(per[0],per[1]))
    output['numSections'] = int(output['numSections'])
    return output

def generateSources(chdPath,timeInterval,procPerameters):
    '''Script for implementing marks code for generating VHF source locations
    from information obtained from the INTF array

    :param chdPath:
        File pointing to the channel-D file from INTF array

        ****Note the other channel files must also be in the same parrent folder
        for the channel-D file****
            #code will compile, but the result will be a file with just
            #header data
    :type chdPath:
        string
    :param timeInterval:
        The temporal range where sources should be generated from the
        INTF array channel files

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !!!!!!!Must have units of miliseconds!!!!!!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    :type timeInterval:
        :tuple:(float,float)
    :param procPerameters:
        File path pointing to configuration file that dictates the conditions
        the source data points are processed from
    :type procPerameters:
        strings

    '''
    #the start and end times that define the processing interval
    start,end = timeInterval if timeInterval[0] < timeInterval[1] else\
                                            (timeInterval[1],timeInterval[0])

    per = loadPerameters(procPerameters)
    cwd = os.getcwd()
    os.chdir(per['sourceProcLocation'])



    #loading processing perameters used in generating the source data
    T = per['T']
    S = per['S']
    I = per['I']
    P = per['P']
    W = per['W']
    Y = per['Y']
    #generating the string used to make a subprocess call
        #done to get around python2 and python3 miss-matching
    cmd = 'python intf_process2.py ' + \
                '-S %i -P %i -I %i -W %i -T %i -Y %i -s %f -p %f --pix 200'\
                                %(S,P,I,W,T,Y,start,end)
    cmd = cmd.split(' ')
    cmd.append(chdPath)
    # cmd.append('/dev/null')
    sourcePath = chdPath[:-4] + '.dat.gz'

    n = 1
    while n <= per['numProcessingAttempts']:
        subprocess.run(cmd)     #generate the sources data
        #try to read the data
        try:
            intf,triggerTime = loadSources(sourcePath)
            os.chdir(cwd)
            return (intf,triggerTime)
            break
        except pd.errors.EmptyDataError:    #if the data file is empty
            os.remove(sourcePath)
        finally:
            n += 1
    else:   #means that no source data points were located
        os.chdir(cwd)
        try:
            os.remove(sourcePath)
        except:
            pass
        os.chdir(cwd)
        return False,False

    os.chdir(cwd)

def sfericToSourcesInterval(sfericTimeInterval,chdPath):
    '''Under the assumption that the units for the sfericTimeInterval interval
    are in microseconds: this will generate a time interval that can be used to
    generate the sources data for a single event.

    :param sfericTimeInterval:
        The time interval that needs to be converted
        !!!!units of microseconds!!!!
    :type sfericTimeInterval:
        :tuple:(float,float)
    :param chdPath:
        The file name of file path of the channel D file that coresponds to the
        same time interval for the sources. This would be the chD file that was
        used in the generation of the sources
    :type chdPath:
        string
    '''

    offset = get_date_time_from_filename(chdPath)[1]*10**3
    sourceInterval = [c*1e-3 - offset for c in sfericTimeInterval]
    return sourceInterval

def main():
    pass

if __name__ == '__main__':
    main()
