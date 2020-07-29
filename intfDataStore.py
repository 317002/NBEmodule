'''Module for reading and manipulating data from the INTF array
'''


# %%
import numpy as np
import matplotlib.pyplot as plt #temp

from collections import namedtuple

####Custom Imports####
import mas_raw

# %%


def loadFA(fileLocation):
    '''Reads data from the VHF and HF sferic channel of the INTF array and
    returns it in a fa data object

    :param fileLocation:
        The file path to the sferic channelS of the intf array
    :type fileLocation:
        string
    :return:
        fa_store object
    :rtype:
        :class:'fa_store'
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

    return fa_store(t_vhf,fa)

# %% sferic class
class sferic:
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

        self.t_vhf = self.t_vhf[points]
        self.fa = self.fa[points]

    def timeSelect(self,rang):
        '''Returns the values of t_vhf and fa that are within the the time
        interval defined by rang

        :param rang:
            The time interval used to filter the arrays
        :type rang:
            tuple
        :return:
            the time and sferic arrays within the interval
        :rtype:
            numpy arrays in named tuple
        '''
        start,end = (rang[0],rang[1]) if rang[0] < rang[1] else (rang[1],rang[0])
        #selecting the indices that are withing the interval
        indicies = np.where(((self.t_vhf >= start) & (self.t_vhf <= end)))

        outputFormater = namedtuple('filtOutput',['time','fa'])
        return outputFormater(self.t_vhf[indicies],self.fa[indicies])

    def genSubPlot(self,ax,xlabel,ylabel,title,nXticks = 10):
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
        :return:
            The reformated axis object
        :rtype:
            :class:'matplotlib.axes._subplots.AxesSubplot'
        '''
        #label and values for the x axis
        xTicks_values = np.round(np.linspace(self.t_vhf[0],self.t_vhf[0],nXticks),2)

        pass
# %%

# %% nbe class
class NBE_locator(sferic):

    def max_of_a_section(self,array,n,numSections):
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

    def min_of_a_section(self,array,n,numSections):
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

    def getAmpFirstSection(self,array,fraction):
        '''Gets the amplitude of an during the initial section of the time
        interval. The lenght of the initial section is defind as the product
        between <fraction> and the total lenght of the interval

        array: class:'numpy.ndarray'
            The array where the max of section will be found
        fraction : float
            The fraction of the total interval that will define the size of the
            intial section
        return : float
            The amplitude of the initial section
        '''
        numSections = int(1/fraction)
        return self.max_of_a_section(array,0,numSections)

    def getNoiseLevels(self,array,numSections):
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
                        n_max = self.max_of_a_section(array,n,numSections)
                        m_max = self.max_of_a_section(array,m,numSections)
                        n_min = self.min_of_a_section(array,n,numSections)
                        m_min = self.min_of_a_section(array,m,numSections)

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
                        n_max = self.max_of_a_section(array,n,numSections)
                        m_max = self.max_of_a_section(array,m,numSections)
                        n_min = self.min_of_a_section(array,n,numSections)
                        m_min = self.min_of_a_section(array,m,numSections)

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

    def getNBE(self,\
                safetyFactor,\
                nbe_range,\
                numSections,\
                firstFraction):
        '''attempts to locate an NBE waveform in the sferic data set

        This scripts takes advantage of known properties of an NBE sferic
        waveform:
            primary property:
                The fact that an NBE tend to be the first signal -present in as
                sferic waveform
            secondart properties:
                -The avg duration of the waveform
                -Sources tend to show up at the start of the waveform after
                 being absent for some time
                -The rise time of the wave form tend to be consistant


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
                 cant be predicted with that level of confidence.
        firstFraction:float range: (0,1)
            -Defines the fractional percentage
            of the region used to determin
            if there is activity in the begining of the event.

            .1 tends to be a good value

        :return:
            The start and end time stamps of a predicted NBE, along with some
            properties in a dict format

            ((nbe start,nbe end),properties)
        :rtype:
            namedtuple(tuple(float,float),dict)
        '''

        #measuring the noise level of the sferic array
        lowerNoiseLevel,upperNoiseLevel = self.getNoiseLevels(self.fa,numSections)

        #means a noise level couldnt be found and therefor neither can an nbe
        if lowerNoiseLevel == None:
            return None,None

        #How much to deviate the noise levels as to not acidently 'scrap' the
        #noise floor when searching for a signal
        safetyFactor = abs(upperNoiseLevel - lowerNoiseLevel)*safetyFactor

        upperThreshold = upperNoiseLevel + safetyFactor
        lowerThreshold = lowerNoiseLevel - safetyFactor

        ####Locating Detection Index####
        try:
            #Index of the first detection using the upper threshold
            upper_index = np.where(self.fa >= upperThreshold)[0][0]
        except IndexError:
            upper_index = None

        try:
            #Index of the first detection using the lower threshold
            lower_index = np.where(self.fa <= lowerThreshold)[0][0]
        except IndexError:
            lower_index = None

        #checking that an event was found
        if upper_index == None and lower_index == None:
            '''No Event Was Located'''
            return (None,None)


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
        detection_time = self.t_vhf[detection_index]

        #defining the time range of the NBE
        nbeStartTime = detection_time + nbe_range[0]
        nbeEndTime = detection_time + nbe_range[1]

        ####finding the second time the signal crosses the threashold####

        #finding the value in the time array, that is atleast 1 microsecond
        #after the detection time
        offset = np.where(self.t_vhf >= detection_time + 1)[0][0]



        if sfericPolarity == 'Positive':
            #finding when the signal crosses below the upper threshold,
            #1 microsecond affter the detection_time
            second_detection_index = np.where(self.fa[offset:] \
                                                <= upperThreshold)[0][0]
            #re-centering the index value to that of the whole fa array
            second_detection_index += offset
        elif sfericPolarity == 'Negative':
            #finding when the signal crosses below the upper threshold,
            #1 microsecond affter the detection_time
            second_detection_index = np.where(self.fa[offset:] \
                                                >= lowerThreshold)[0][0]
            #re-centering the index value to that of the whole fa array
            second_detection_index += offset
        else:
            raise Exception('How?')

        second_detection_time = self.t_vhf[second_detection_index]

        #intializing properties dict
        nbeProp = {}
        nbeProp['lowerNoiseLevel'] = lowerNoiseLevel
        nbeProp['upperNoiseLevel'] = upperNoiseLevel
        nbeProp['detectionTime'] = detection_time
        nbeProp['secondDetectionTime'] = second_detection_time


        outputFormater = namedtuple('NBE',\
                                    ['NBE_TimeInterval',\
                                    'NBE_Properties'])

        return outputFormater((nbeStartTime,nbeEndTime),nbeProp)
# %%

def main():

    # %% init
    import os
    import os.path as pat

    home = os.getcwd()


    # chd_path = pat.join(home,'test.chd')
    # sferic = loadFA(chd_path)
    a,c = 1,10
    testTime = np.linspace(-200,200,2**15)
    testFa = a*np.exp((-testTime**2)/c**2) + \
                                    np.random.random(testTime.shape)*.05 - a/2


    # %% locate NBE

    locator = NBE_locator(testTime,testFa)
    nbe,properties = locator.getNBE(.1,(-30,100),20,.1)

    t_nbe,fa_nbe = locator.timeSelect(nbe)

    vert = np.linspace(np.min(testFa),np.max(testFa),100)
    blank = np.ones(100)



    # %% plot
    plt.plot(testTime,testFa)
    plt.plot(t_nbe,fa_nbe)
    plt.plot(blank*properties['detectionTime'],vert)
    plt.show()
    # %%


if __name__ == '__main__':
    main ()
