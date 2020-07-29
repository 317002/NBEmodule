# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:07:04 2019

@author: Nathan Richard
"""

import numpy as np
import pandas as pd
import mas_raw
import gzip
import os
import convient as con
from matplotlib.colors import Normalize as norm
from collections import namedtuple

from scipy.signal import find_peaks
import copy
import time
import matplotlib.pyplot as plt


cwd = lambda :print(os.getcwd())

class intf_store:
    def __init__(self,data_path):
        #checking if the file is in a compressed format or not
        if '.gz' in data_path:
            #if compressed read data using pandas
            data = np.array(pd.read_csv(data_path\
                            ,comment = '#'\
                            ,delimiter = ' '\
                            ,skipinitialspace = True\
                            ,skip_blank_lines = True)).transpose()
        else:
            #if uncompressed read using basic numpy load txt
            data = np.loadtxt(data_path).transpose()


        time,azi,elev,cosa,cosb,pk2pk,rms,expk,ecls,estd,emlt,red,green,\
        blue,startsample,imax = data
        self.time = time#in seconds
        self.azi = azi
        self.elev = elev
        self.cosa = cosa
        self.cosb = cosb
        self.pk2pk = pk2pk
        self.rms = rms
        self.expk = expk
        self.ecls = ecls
        self.estf = estd
        self.emlt = emlt
        self.red = red
        self.green = green
        self.blue = blue
        self.startsample = startsample
        self.imax = imax
        self.check_normal = False
        self.time_check = False


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

    def time_filt(self,time_range):
        self.filt(np.where((self.time >= time_range[0])&\
                           (self.time <= time_range[1])))
        self.time_check = True

    def get_polarity_pos(self,order = 9):
        '''
        Its assumed that the nbe has been found and the time interval of that
        event has been used to filter out the intf data that dosent corespond
        to the nbe

        returns True when pol is positive
        '''
        ####Gathering the time interval of the event####
        time_range = [np.min(self.time),np.max(self.time)]
        #the time interval the intf data coresponds to

        time_range_zero = [c - min(time_range) for c in time_range]
        #starting the time range at zero to work around issues with the
        #np.polyfit func

        ####Ploynomial Regrasion####
        t = np.linspace(*time_range_zero,self.elev.shape[0])
        f = np.poly1d(np.polyfit(t,self.elev,order))
        t = np.linspace(*time_range_zero,1000)
        elev_fit = f(t)#the polynomial fit of the data set
        elev_fit_grad = np.gradient(elev_fit,t[1] - t[0])

        self.time_offset = t + min(time_range)
        self.elev_fit = elev_fit


        ####finding first zero derivative point####
        first_cross = np.where(np.diff\
            (np.sign(elev_fit_grad[np.where(elev_fit_grad != 0)])))[0][0]

        return (True if elev_fit[first_cross] - elev_fit[0] > 0 else False)

class fa_store:
    def __init__(self,data_path):
        #### Reding in the data####
        chd_file_handler = mas_raw.atvt()#defing the file handling class
        chd_file_handler.load_file(data_path)#loading the
        #chd file into the hanlder
        t_vhf = np.array(chd_file_handler.t_i)#pulling out the time data
        #(seconds after midnight)
        t_vhf = t_vhf*1e6#converting to microseconds from midnight
        fa = np.array(chd_file_handler.raw_i)#intesity?

        fa = fa - 2.0**15
        fa_calibConst = -160.0/(2**15)	# calibration (linear)
                                        #160 V/m per 2^15 digital units (bits).
        fa = fa_calibConst*fa

        ####variable decleration####
        self.t_vhf = t_vhf
        self.fa = fa
        self.c_time = False
        #a check to see if the time vector starts at zero

        ####None initializing####

        self.lower_noise_level = None
        self.upper_noise_level = None
        self.hit_time = None
        self.safety_factor = None
        self.recur = 0

    def filt(self,points):
        #lest you gather the data that corespond to the index vector: points
        self.t_vhf = self.t_vhf[points]
        self.fa = self.fa[points]

    def time_filt(self,time_range):
        self.filt(np.where((self.t_vhf >= time_range[0])&\
                           (self.t_vhf <= time_range[1])))

    def get_upper_noise_level(self,array,fraction,threshold,firstFraction = None):
        '''
            aproximates the highest noise level of the change in electric field
        signal from the channel D attenna. Used in the identification of NBEs

        array:numpy array
                The 1-d array the porcessing is done on
        fraction:float
            fraction of the time interval used to estimate the noise
                *The wider the interval the more confident the estimantion
                *The wider the interval the more likely a noise level cant be
                predicted with that level of confidence.
        threshold:float
            The max allowed percentage difference between the two sections used
            to predict the noise level
        firstFraction:float
                When a given, defines the size of the region when Calculating
            the amplitude of the signal during the begining of the signal
        '''

        if threshold > 1:
            raise Exception('threashold must be less then 1')

        def diff_measurment(a,b):
            '''
            Calculates the relative difference between values a and b
                returns the fractional difference from a to b

            a:float
            b:float
            '''
            #handling discontinuity at b = 0
            if b != 0:
                return np.abs((a-b)/b)
            else:
                return a

        #defining how many many different sections of the data can be defined
        #with the given fractional interval of the data set
        incrament = int(1/fraction)

        #cheking that the given value for fraction can be used with confidence
        if incrament - 1/fraction != 0:
            raise Exception('Must use a faction that is rational in decimal form:'+\
                            'you selected:' + str(fraction))

        #if a value is given: find the initial amplitude
        if firstFraction != None:
            self.amp_firstSection = \
                        con.max_of_a_section(array,1,firstFraction)
            self.avg_firstSection = \
                con.max_of_a_section(array,1,firstFraction,avg = True)

        #left to right
        check = False
        for n in range(1,incrament + 1):
            for m in range(1,incrament + 1):
                n_adv = con.max_of_a_section(array,n,fraction)
                m_adv = con.max_of_a_section(array,m,fraction)
                #we look at the max value of the section in the array
                if m != n:
                    #making sure we are not comparing the same section
                    if diff_measurment(n_adv,m_adv) < threshold:
                        #checking to see when two sections are within a threshold of
                        #each other. This should happen when the data as a whole is
                        #relativly flat
                        #gives us a ball park estimate of the max noise level.

                        lr_max = np.max([n_adv,m_adv])
                        check = True
                        break
            if check == True:
                break


        #right to left
        check = False
        for n in range(-incrament,0):
            n = -n
            for m in range(-incrament,0):
                m = -m
                n_adv = con.max_of_a_section(array,n,fraction)
                m_adv = con.max_of_a_section(array,m,fraction)
                if m != n:
                    #making sure we are not comparing the same section
                    if diff_measurment(n_adv,m_adv) < threshold:
                        #checking to see when two sections are within a threshold of
                        #each other. This should happen when the data as a whole is
                        #relativly flat
                        #gives us a ball park estimate of the max noise level.

                        rl_max = np.max([n_adv,m_adv])
                        check = True
                        break
            if check == True:
                break

        self.upper_noise_level = lr_max if lr_max >= rl_max else rl_max

    def get_lower_noise_level(self,array,fraction,threshold):
        '''
            aproximates the highest noise level of the change in electric field
        signal from the channel D attenna. Used in the identification of NBEs

        array:numpy array
                The 1-d array the porcessing is done on
        fraction:float
            fraction of the time interval used to estimate the noise
                *The wider the interval the more confident the estimantion
                *The wider the interval the more likely a noise level cant be
                predicted with that level of confidence.
        threshold:float
            The max allowed percentage difference between the two sections used
            to predict the noise level
        '''

        if threshold > 1:
            raise Exception('threashold must be less then 1')

        def diff_measurment(a,b):
            '''
            Calculates the relative difference between values a and b
                returns the fractional difference from a to b

            a:float
            b:float
            '''
            #handling discontinuity at b = 0
            if b != 0:
                return np.abs((a-b)/b)
            else:
                return a

        #defining how many many different sections of the data can be defined
        #with the given fractional interval of the data set
        incrament = int(1/fraction)

        #cheking that the given value for fraction can be used with confidence
        if incrament - 1/fraction != 0:
            raise Exception('Must use a faction that is rational in decimal form:'+\
                            'you selected:' + str(fraction))

        #left to right
        check = False
        for n in range(1,incrament + 1):
            for m in range(1,incrament + 1):
                n_adv = con.min_of_a_section(array,n,fraction)
                m_adv = con.min_of_a_section(array,m,fraction)
                #we look at the min value of the section in the array
                if m != n:
                    #making sure we are not comparing the same section

                    if diff_measurment(n_adv,m_adv) < threshold:
                        #checking to see when two sections are within a threshold of
                        #each other. This should happen when the data as a whole is
                        #relativly flat
                        #gives us a ball park estimate of the max noise level.

                        lr_max = np.min([n_adv,m_adv])
                        check = True
                        break
            if check == True:
                break


        #right to left
        check = False
        for n in range(-incrament,0):
            n = -n
            for m in range(-incrament,0):
                m = -m
                n_adv = con.min_of_a_section(array,n,fraction)
                m_adv = con.min_of_a_section(array,m,fraction)
                if m != n:
                    #making sure we are not comparing the same section
                    if diff_measurment(n_adv,m_adv) < threshold:
                        #checking to see when two sections are within a threshold of
                        #each other. This should happen when the data as a whole is
                        #relativly flat
                        #gives us a ball park estimate of the max noise level.

                        rl_max = np.min([n_adv,m_adv])
                        check = True
                        break
            if check == True:
                break

        self.lower_noise_level = lr_max if lr_max <= rl_max else rl_max

    def get_nbe(self,safety_factor,nbe_range,fraction,firstFraction,disp = 0):
        '''
        attempts to locate an NBE in a general sferic measurment for the
        channel D data file


        safety_factor:float
                The factor to multiply the estimated noise levels by to generate
            values used as the detection threshold.
        nbe_range:tuple (start,end)
                Defines the time interval the nbe will ocupi. Centered around
            the detection point, the nbe is defined as ocupying the range:
            [-nbe_range[0],nbe_range[1]]

            usally a [-30,100] microsecond range suffices
        fraction:float range: (0,1)
                Used in the detection of nbes. Defines how many parts the
            the fa signal is split into when estimating the noise floor levels

            .05 tends to be a good value
        firstFraction:float range: (0,1)
                Defines the fractional percentage of the region used to determin
            if there is activity in the begining of the event.

            .1 tends to be a good value
        disp:int
                The index value to start from when searching for and NBE. The
            default selects the entire event
        '''

        #generating a sub array bassed on the disp value
        fa = self.fa[disp:]
        t_vhf = self.t_vhf[disp:]

        ####Aproximating the upper and lower exstent of the noise floor####

        #init allowed tolerence between sections
        threshold = .01
        maxTolerence = .5 #max allowed tolerence set to 50%

        #Done so that if the func is called again,
            #the noise level isnt recaculated
        if self.lower_noise_level == None:
            while threshold < maxTolerence:
                try:
                    self.get_lower_noise_level(fa,fraction,threshold)
                    break#leaves the loop when noise level is found
                except UnboundLocalError:
                    #when the tolerence is too tight, try again with a greater
                    #allowed tolerence
                    threshold += .01
            else:
                raise Exception('Failed to find lower noise level')



        threshold = .01

        #getting the upper noise level of the floor using the same method
            #we also gather the amplitude of the siganl in the first section
            #of the fa signal
        if self.upper_noise_level == None:
            while threshold < maxTolerence:
                try:
                    self.get_upper_noise_level(fa,fraction,threshold\
                                                ,firstFraction=firstFraction)
                    break#leaves the loop when noise level is found
                except UnboundLocalError:
                    #when the tolerence is too tight, try again with a greater
                    #allowed tolerence
                    threshold += .01
            else:
                raise Exception('Failed to find upper noise level')

        if self.safety_factor == None:
            self.safety_factor = np.abs(self.upper_noise_level - \
                                        self.lower_noise_level)*safety_factor
            safety_factor = self.safety_factor


        #deffining the levels used in the detection of the first instance of
        #activity

        self.upper_threshold = self.upper_noise_level + safety_factor
        self.lower_threshold = self.lower_noise_level - safety_factor

        #Index of the first detection using the lower threshold
        lower_index = np.where(fa <= self.lower_threshold)[0]

        try:
            #Index of the first detection using the upper threshold
            upper_index = np.where(fa >= self.upper_threshold)[0][0]
        except IndexError:
            upper_index = None

        try:
            #Index of the first detection using the lower threshold
            lower_index = np.where(fa <= self.lower_threshold)[0][0]
        except IndexError:
            lower_index = None


        #checking that an event was found
        if upper_index == None and lower_index == None:
            raise Exception('No event was located')

        if upper_index == None:
            #lower index must have a value
            detection_index = lower_index
            #means we have a negative vent
            self.fa_pol = 'Negative'
        elif lower_index == None:
            #upper index must have a value
            detection_index = upper_index
            #means we have a positive event
            self.fa_pol = 'Positive'
        else:
            #if there is a value for both, select the one that takes place sooner
            detection_index,self.fa_pol = (lower_index,'Negative') \
                                    if lower_index <= upper_index \
                                        else (upper_index,'Positive')

        #getting the time stamp for the nbe detection for the detected nbe
        self.detection_time = t_vhf[detection_index]

        #defining the time range of the NBE
        self.nbeStartTime = self.detection_time - nbe_range[0]
        self.nbeEndTime = self.detection_time + nbe_range[1]
        #gathering the indicies that range
        nbe_indicies = np.where(((self.t_vhf >= self.nbeStartTime)\
                            &(self.t_vhf <= self.nbeEndTime)))[0]

        #gathering the array data coresponding to the nbe
            #gathered from the global set not the one offset by disp
        self.t_nbe = self.t_vhf[nbe_indicies]
        self.fa_nbe = self.fa[nbe_indicies]

        ####finding the second time the signal crosses the threashold####

        #finding the value in the time array, that is atleast 1 microsecond
        #after the detection time. Used in finding the second time the fa
        #signal crosses the threshold.
        offset = np.where(t_vhf >= self.detection_time + 1)[0][0]
        self.offset = offset


        if self.fa_pol == 'Positive':
            #finding when the signal crosses below the upper threshold,
            #1 microsecond affter the detection_time
            second_detection_index = np.where(fa[offset:] \
                                                <= self.upper_threshold)[0][0]
            #re-centering the index value to that of the whole fa array
            second_detection_index += offset
        elif self.fa_pol == 'Negative':
            #finding when the signal crosses below the upper threshold,
            #1 microsecond affter the detection_time
            second_detection_index = np.where(fa[offset:] \
                                                >= self.lower_threshold)[0][0]
            #re-centering the index value to that of the whole fa array
            second_detection_index += offset
        else:
            raise Exception('How?')

        self.second_detection_time = t_vhf[second_detection_index]


        ####Returning Various Properties of the detected event####

        '''
        eventWidth
                The difference in time between the moment the event is detected
            and the moment the signal crosses back over the used threshold.
            ie:
                    if the event was detected using the upper threshold, we find
                the next instance it crosses back under that threshold
                    if the event was detected using the lower threshold, we find
                the next instance it crosses back back over that threshold
        riseTime
                The slope of the of a line defined by the point defined by the
            detection index and the point of max amplitude, if the event has a
            positive polarity, the point of min amplitude otherwise.

            ##########This needs to be revisited##########
        '''


        ####defining those quantities####

                ####Event Width####
        eventWidth = self.second_detection_time - self.detection_time
        '''
                ####Rise Time####
        if self.fa_pol == 'Negative':
            #index corespoding to min of the event
            index = np.where(self.fa_nbe == np.min(self.fa_nbe))[0][0]
            try:
                riseTime = (self.fa_nbe[index] - fa[detection_index])\
                                    /(self.t_nbe[index] - self.detection_time)
            except ZeroDivisionError:
                raise Exception('The time that the event starts is also'+\
                                    ' the time where the fa signal is at a min')
        if self.fa_pol == 'Positive':
            #idex coresponding to the max of the event
            index = np.where(self.fa_nbe == np.max(self.fa_nbe))[0][0]
            try:
                riseTime = (self.fa_nbe[index] - fa[detection_index])\
                                    /(self.t_nbe[index] - self.detection_time)
            except ZeroDivisionError:
                raise Exception('The time that the event starts is also'+\
                                    ' the time where the fa signal is at a max')
        '''
        # properties = {'eventWidth':eventWidth,'riseTime':riseTime}
        properties = {'eventWidth':eventWidth}
        output_formater = namedtuple('Output_Properties',['disp','properties'])
        return output_formater(second_detection_index + disp,properties)















    def get_nbe_polarity_pos(self):
        #returns true when the polarity is positive, False when negative
        max_peak = abs(np.max(self.fa_nbe))
        min_peak = abs(np.min(self.fa_nbe))
        if max_peak > min_peak:
            return True
        elif max_peak < min_peak:
            return False
        else:
            return None

    def get_rise_time(self,exstent = 60,safety_factor = 2,ax = None):
        '''
        The exstent is the amount of points grabbed to isolate the rise time
        of the vhf signal. to many points will result in a false measurment
        but at the same time we want as many as possible to get an accurent
        measurment
        '''
        if self.noise_level == None:
            raise Exception('No noise level present. run get_noise_level')


        ####Isolating the rise time section####
        base_line = safety_factor*self.noise_level
        p = np.where(self.fa != 0)
        data = np.log(self.fa[p]**2)
        time = self.t_vhf[p]
        p = np.where(data >= base_line)
        data = data[p]
        time = time[p]
        data,time = data[0:exstent],time[0:exstent]

        ####Calculating the rise time
        constants = np.polyfit(time,data,1)
        rise_time = 1/constants[0]



        return data,time,rise_time

def extract_path(files,string):
    points = np.asarray([string in c for c in files])
    #list of true and false values. the true value tells me which entry in the list
    #is the one i want
    if True in points:
        path = files[np.where(points == True)[0][0]]
        #extracting the channel c file path
        return path
    else:
        raise Exception('File not found')

def color_map(array):
    #generates a color map array for some data
    maper = norm(np.min(array),np.max(array))
    return maper(array)
