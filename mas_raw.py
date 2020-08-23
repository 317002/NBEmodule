"""
MODULE: mas_raw

PURPOSE: A collection of classes for handling raw data

CLASSES:
    atvt        Mark Stanley's ATVT data.  This is commonly used for broadband
                 interferometers, but has been used for other applications.
    atvt_header Header of an ATVT data file
    sa24        Bill Rison's 24-bit A/D data written to SD card.  This is commonly used
                 for sampling slow antenna data, though it has also been used for acoustic
                 measurements.

MODIFICATION HISTORY:
'15Sep26  v0.1    Created by Mark Stanley
   Dec09  v0.11   1) Fixed bug in 'atvt.load()' which prevented it from working properly
                  2) Fixed bug in 'atvt.load_file()' which did not properly calculate
                     times when start sample was greater than zero
   Jun04  v0.2    1) Modified get_date_time_from_filename() to handle cases where the
                     extension is missing (by checking the length of the presumed ext)
                  2) Added 'load_offset_table()' and 'get_file_start_time()' to atvt class
                  3) Added 'bOffsetLoaded' and 'dfOffset' to atvt class
                  4) Added correct time/freq option to 'load_file()' in atvt class
   Jun14          1) Renamed 'get_file_start_time()' as 'get_file_timing()' and
                     modified output to be a class with attributes
                  2) Added correct time/freq option to 'load()' in atvt class
"""

__version__ = "0.2"


####  IMPORTS  ####

import datetime
import glob
import numpy as np
import os
import pandas as pd


# Generic result class (user assigns attributes)
class result_container:
    pass


# Generic raw data class
class generic( object ):
    """Generic class which is inherited by all other raw data classes
    """

    def __init__( self ):

        self.baseDir  = ''      # Base directory of data files
        self.fileList = []      # List of currently loaded raw data files
        self.fileExt  = ''      # Data file extension (example: '.dat')

        self.bDiscTrig = False  # Are there discrete triggers of uniformly spaced data?
        self.bUniform  = True   # Uniformly spaced data in time (as opposed to async)?
        self.iDiscList = []     # Start indices for discrete triggers of uniformly...
        self.raw_i = None       # Array of raw data in signed or unsigned integers
        self.t_i   = None       # Associated times in seconds-since-midnight UT
        self.y_i   = None       # Data, converted to final units and possibly filtered too
        self.init_data()        # Initialize above list and arrays to empty

        self.bytesPerSample = 1 # Bytes per sample in file
        self.dataType =np.uint8 # Type of data to read from file

        self.freq       = 0.    # Frequency of sampling in Hertz (if uniform)
        self.rawToUnits = 1.    # Conversion factor for raw to final units
        self.rawToV     = 1.    # ...to volts

        self.idTxt = ''         # Station ID
        self.lat = 0.           # Mean latitude of sensor [deg N]
        self.lon = 0.           # Mean longitude [deg E]
        self.alt = 0.           # Mean altitude [m]

        self.gpsLat_i = None    # Array of GPS latitudes [deg]
        self.gpsLon_i = None    # ...longitudes east [deg]
        self.gpsAlt_i = None    # ...altitudes MSL [m]
        self.gpsSatVis_i = None # ...number of satellites visible
        self.gpsSatTrack_i = None   # ...satellites being tracked
        self.init_gps()         # Initializes above GPS arrays

        self.date = datetime.date( 2015, 9, 27 )


    def get_file_size_bytes( self, filename ):
        """generic.get_file_size_bytes( filename ):

        Returns the size of the file in bytes.  If the file does not exist, then -1 is
        returned.
        """

        if os.path.exists( filename ):
            fStat = os.stat( filename )
            return  fStat.st_size
        else:
            return  -1


    def get_search_path( self, date=None, daySec=None ):
        """generic.get_search_path( date=None, daySec=None ):

        Returns a string corresponding to the complete search path when implemented in
        a child function
        """
        return  None


    def filter_data_time_range( self, tStart, tEnd ):
        """generic.filter_data_time_range( )

        Removes data elements which have time values outside of tStart and tEnd.

        RETURN:  TRUE if there were points within the requested interval, FALSE for all
                 other conditions
        """

        # Locate indices within bounds
        if len(self.t_i) > 0:
            ii = np.where( (self.t_i >= tStart) & (self.t_i <= tEnd) )[0]

            # Exit if no indices found
            if len(ii) == 0:
                return  False
        else:
            return  False

        # Make a copy of filtered time values
        tTemp_i = self.t_i[ii].copy()

        # Make a copy of data values (which exist)
        rawTemp_i = np.array( [], dtype=np.uint32 )
        yTemp_i   = np.array( [], dtype=np.float64 )
        if len(self.raw_i) > 0:
            rawTemp_i = self.raw_i[ii].copy()
        if len(self.y_i) > 0:
            yTemp_i = self.y_i[ii].copy()

        # Delete larger arrays (by initializing to empty)
        self.init_data()

        # Copy temporary arrays to internal copies
        self.t_i   = tTemp_i
        self.raw_i = rawTemp_i
        self.y_i   = yTemp_i

        return  True


    def init_data( self ):
        """generic.init_data( )

        Initializes data arrays to empty and discrete trigger index list to empty
        """
        self.raw_i = np.array( [], dtype=np.uint32 )
        self.t_i   = np.array( [], dtype=np.float64 )
        self.y_i   = np.array( [], dtype=np.float64 )

        self.iDiscList = []


    def init_gps( self ):
        """generic.init_gps( )

        Initializes GPS data arrays to empty
        """
        self.gpsLat_i = np.array( [], dtype=np.float64 )
        self.gpsLon_i = np.array( [], dtype=np.float64 )
        self.gpsAlt_i = np.array( [], dtype=np.float64 )

        self.gpsSatVis_i   = np.array( [], dtype=np.uint8 )
        self.gpsSatTrack_i = np.array( [], dtype=np.uint8 )


    def load( self, tStart, tEnd, date=None ):
        """generic.load( tStart, tEnd, date=None)

        Loads all data on the target date within the specified time interval
        (seconds of day) when implemented in child function.
        """
        pass


    def load_file( self, filename ):
        """generic.load_file( filename ):

        Loads data from the specified file when implemented in child function.
        """
        pass


    def set_base_directory( self, dirPath ):
        """generic.set_base_directory( dirPath ):

        Checks that the intended base directory exists and then sets the corresponding
        internal variable if it does.

        RETURNS:  TRUE if base directory set, FALSE otherwise
        """

        if os.path.exists( dirPath ):
            self.baseDir = dirPath
            return  True
        else:
            return  False


    def set_date( self, year, month, day ):
        """generic.set_date( year, month, day ):

        Sets the date on which the raw data was collected
        """

        self.date = self.date.replace( year, month, day )


    def set_location( self, lat, lon, alt ):
        """generic.set_location( lat, lon, alt ):

        Sets the location of the sensor where 'lat' is in degrees north, 'lon' is in
        degrees east and 'alt' is in meters above mean sea level.
        """

        self.lat = lat
        self.lon = lon
        self.alt = alt


# Class for ATVT data files:
class atvt( generic ):
    """Class for handling data files output from the AlazarTech Versatile Triggering
    (ATVT) software program written by Mark Stanley.
    """

    def __init__( self, offsetTable='' ):
        generic.__init__( self )

        # Raw data is always composed of 16-bit unsigned integers
        self.dataType = np.uint16
        self.bytesPerSample = 2

        # Initialize an instance of a file header class
        self.cHeader = atvt_header( )

        # Default to 1st channel
        self.fileExt = '.chA'

        # Attempt to load offset table, if specified
        self.dfOffset = pd.DataFrame()
        if offsetTable:
            self.bOffsetLoaded = self.load_offset_table( offsetTable )
        else:
            self.bOffsetLoaded = False


    def channel_int_to_str( self, channel ):
        """atvt.channel_int_to_str( channel ):

        Returns the string representation of an integer channel number.  Channel
        numbers start at 0 and map as expected to the alphabet (0='A', 1='B', etc).
        """

        return  chr( ord('A') + channel )


    def get_date_time_from_filename( self, filename, maxExt=4 ):
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


    def get_file_timing( self, filename, maxExt=4 ):
        """atvt.get_file_timing( filename, maxExt=4 ):

        USAGE:  cCal = atvt.get_file_timing( filename )

        Returns a class with the following attributes:
            t0          Start time [sec]
            freq        Sample rate [Hz]
            nAnt        Number of antennas averaged together
            nSig        Number of points in 1-sigma
            preSec      Pretrigger [sec]
            postSec     Posttrigger [sec]
            sigma       Sigma [nsec]
        ...using the internal time offset lookup table.  If the calibration
        information is not available for the specified file, all attributes
        will be None.
        """

        # Determine base file name (which is used as an index)
        filePath, fileNameExt = os.path.split( filename )
        fileName, fileExt     = os.path.splitext( fileNameExt )
        if len( fileExt ) > maxExt:
            # Extension is longer than expected.  Assume extension is missing
            fileName = fileNameExt

        # Create an instance of the result container
        cResult = result_container()

        # Locate the file in the lookup table, if it exists -and- set attributes
        dfFile = self.dfOffset.loc[self.dfOffset['File base'] == fileName]
        if dfFile.empty:
            cResult.t0      = None
            cResult.freq    = None
            cResult.nAnt    = None
            cResult.nSig    = None
            cResult.preSec  = None
            cResult.postSec = None
            cResult.sigma   = None
        else:
            cResult.t0      = dfFile['T0 [sec]'].values
            cResult.freq    = dfFile['Freq [Hz]'].values
            cResult.nAnt    = dfFile['N ant'].values
            cResult.nSig    = dfFile['N sig'].values
            cResult.preSec  = dfFile['Pre [sec]'].values
            cResult.postSec = dfFile['Post [sec]'].values
            cResult.sigma   = dfFile['Sigma [ns]'].values

        return  cResult


    def get_search_path( self, date=None ):
        """atvt.get_search_path( date=None ):

        Returns a string corresponding to the expected search path based on the
        current base directory as well as the target date.  If date is not specified,
        then the date last set is used.  The search path is (normally) assumed to be of
        the form: [baseDir]/YYYY.MM.DD/
        """

        # Append an OS path separater to base directory if it's not already there
        try:
            if self.baseDir[-1] != os.path.sep:
                basePath = self.baseDir + os.path.sep
            else:
                basePath = self.baseDir
        except:
            basePath = ''

        # Set date component of path
        if date is None:
            datePath = self.date.strftime('%Y.%m.%d') + os.path.sep
        elif type(date) == datetime.date:
            datePath = date.strftime('%Y.%m.%d') + os.path.sep
        else:
            datePath = ''

        return  basePath + datePath


    def load( self, tStart, tEnd, bCorrectTime=False, channel='A', date=None,
              filePrefix=None, subSample=1 ):
        """atvt.load( tStart, tEnd, bCorrectTime=False, channel='A', date=None,
        filePrefix=None, subSample=1):

        Loads data within the specified interval (in seconds since midnight) on the
        given date (datetime.date object).

        OPTIONS:
            bCorrectTime  Use corrected times if TRUE, raw (default) otherwise
            channel     The data channel to load.  If this is not set, then 'A' (0) is
                         assumed.  Note that channel can be either an integer or a string.
            date        The target date (datetime.date object).  If date is not specified,
                         the internal date last set is used.
            filePrefix  The file prefix to scan for.  If this is not specified, then the
                         internal station ID text string is used.
            subSample   Setting this to a value larger than one will subsample the data,
                         which is useful if the data is large and the full bandwidth is
                         not needed.

        TODO: Correct for time offsets from UT (using a lookup table)
        """

        # If date specified and is valid, set internal copy
        if date is not None:
            if type(date) == datetime.date:
                self.date = date

        # Set file extension
        if type(channel) == str:
            self.fileExt = '.ch' + channel
        elif type(channel) == int:
            self.fileExt = '.ch' + self.channel_int_to_str( channel )
        else:
            self.fileExt = '.chA'

        # Reset data arrays to empty
        self.init_data( )

        # Set file prefix used for search and search path
        if filePrefix is None:
            searchFile = self.idTxt + '*' + self.fileExt
        else:
            searchFile = filePrefix + '*' + self.fileExt

        # Determine trigger times of all files
        allFileList = glob.glob( self.get_search_path() + searchFile )
        allFileList.sort()
        numFiles = len( allFileList )
        if numFiles == 0:
            return
        tTrig_i = np.zeros( numFiles, dtype=np.float64 )
        iTrig = 0
        cTimingList = []
        for thisFile in allFileList:
            t0 = None
            if bCorrectTime:
                cTiming = self.get_file_timing( thisFile )
                cTimingList.append( cTiming )
                t0 = cTiming.t0
                if t0 is not None:
                    thisTime = t0 + cTiming.preSec
            else:
                cTimingList.append( None )
            if t0 is None:
                thisDate, thisTime = self.get_date_time_from_filename( thisFile )
            if thisTime is not None:
                tTrig_i[iTrig] = thisTime
                iTrig += 1
            else:
                discard = allFileList.pop( iTrig )

        # Shorten array if needed
        if iTrig < numFiles:
            tTrig_i = tTrig_i[:iTrig]

        # Retrieve indices of times which are less than tStart and greater than tEnd
        iiLT = np.where( tTrig_i < tStart )[0]
        iiGT = np.where( tTrig_i > tEnd )[0]

        # The files with the biggest trigger time Less-Than tStart and smallest trigger
        #  time Greater-Than tEnd will bound our interval
        if len(iiLT) > 0:
            i0 = iiLT[-1]
        else:
            i0 = 0
        if len(iiGT) > 0:
            i1 = iiGT[0]
        else:
            i1 = len( tTrig_i ) - 1

        # Loop through files, appending data if any is within interval
        iFile = i0
        thisHeader = atvt_header()
        while iFile <= i1:
            thisFile = allFileList[iFile]
            cTiming = cTimingList[iFile]

            # If no absolute timing information, load header.  Set t0/t1 times
            if cTiming is None:
                thisHeader.load_header( thisFile )
                thisT0 = tTrig_i[iFile] - thisHeader.preSec
                thisT1 = tTrig_i[iFile] + thisHeader.postSec
                thisFreq = thisHeader.sampleRate
            else:
                thisT0 = cTiming.t0
                thisT1 = cTiming.t0 + cTiming.preSec + cTiming.postSec
                thisFreq = cTiming.freq

            # Ignore file if data is entirely outside of desired interval
            if (thisT0 >= tEnd) or (thisT1 <= tStart):
                pass
            else:
                # Start sample index:
                if (tStart <= thisT0):
                    iSample0 = 0
                else:
                    iSample0 = int( np.round(( tStart - thisT0 ) * thisFreq ))

                # Count:
                if (tStart < thisT0):
                    t0 = thisT0
                else:
                    t0 = tStart
                if (tEnd > thisT1):
                    t1 = thisT1
                else:
                    t1 = tEnd
                count = int( np.round((t1 - t0) * thisFreq ) )

                # Load data, appending to existing if any
                discard = self.load_file( thisFile, bAppendData=True,
                                          bCorrectTime=bCorrectTime,
                                          count=count, startSample=iSample0,
                                          subSample=subSample )

            iFile += 1


    def load_file( self, filename, bAppendData=False, bCorrectTime=False, count=-1,
                         startSample=0, subSample=1 ):
        """atvt.load_file( filename, bAppendData=False, bCorrectTime=False, count=-1,
        startSample=0, subSample=1 ):

        Loads an ATVT data file.  Either the entire file can be loaded (default), or
        the sample start and count can be specified if only a portion of the file is
        desired.  In addition, the data can be subsampled by a specified amount,
        which is useful if the data is large and the full bandwidth is not needed.
        The data can either be appended to existing data or that data can be
        overwritten (default).

        OPTIONS:
            bAppendData     Appends the data array instead of overwriting if true
            bCorrectTime    Use corrected times (if available) for the file
            count           Number of raw samples to load (default: ALL).  Note: before
                             subsampling, if any
            startSample     Start sample number (default: 0)
            subSample       The factor by which to subsample the data (default: 1 = none)

        RETURNS:  True if successful, false otherwise
        """

        # Attempt to use search path if can't locate filename.  Exit if still can't find
        if not( os.path.exists( filename ) ):
            filename = self.get_search_path( ) + filename
            if not( os.path.exists( filename ) ):
                return  False

        # Attempt to retrieve header.  Exit if failed.
        if not( self.cHeader.load_header( filename ) ):
            return  False

        # Retrieve actual start time -and- sample rate information, if requested
        if bCorrectTime:
            cTiming = self.get_file_timing( filename )
            freq = cTiming.freq
            if freq is None:
                freq = self.cHeader.sampleRate
                bCorrectTime = False    # Discontinue future attempts to use corrected time
        else:
            freq = self.cHeader.sampleRate

        # Set the internal record of the sample rate
        self.freq = freq

        # Open file (note: this should succeed if we got this far)
        try:
            fp = open( filename, 'rb' )
        except:
            return  False

        # Fast forward to intended start, load data -and- then close file
        try:
            fp.seek( self.cHeader.size + startSample*self.bytesPerSample )
        except:
            return  False
        fileRaw_i = np.fromfile( fp, dtype=self.dataType, count=count )
        fp.close( )

        # Return false if no data was loaded
        if fileRaw_i.size == 0:
            return  False

        # Subsample the data, if requested
        if subSample > 1:
            iSub_i  = np.arange( 0, fileRaw_i.size, subSample ).astype( np.int64 )
            fileRaw_i = fileRaw_i[iSub_i]
            self.freq /= float( subSample )

        # Use raw time if unable to use corrected start time
        if not(bCorrectTime):
            timeTrig = self.cHeader.hour*3600. + self.cHeader.minute*60. + \
                       self.cHeader.second + ( self.cHeader.uSecond / 1e6 )
            t0 = timeTrig - self.cHeader.preSec + startSample/float(self.cHeader.sampleRate)
        else:
            t0 = cTiming.t0

        # Form seconds-of-day times
        dT = 1. / self.freq
        fileT_i = t0 + (dT * np.arange( fileRaw_i.size, dtype=np.float64 ) )

        # Append or overwrite raw/time data arrays
        if bAppendData:
            self.bDiscTrig = True
            self.iDiscList.append( self.raw_i.size )
            self.raw_i = np.append( self.raw_i, fileRaw_i )
            self.t_i   = np.append( self.t_i,   fileT_i )
            self.fileList.append( filename )
        else:
            self.bDiscTrig = False
            self.raw_i = fileRaw_i
            self.t_i   = fileT_i
            self.fileList = [ filename ]

        return  True


    def load_offset_table( self, offsetTable ):
        """ atvt.load_offset_table( offsetTable):

        Load the CSV file of time offset corrections.  This lookup table maps the trigger file
        names to the correct data start times [UT] as well as actual sample rates.  Please note
        that the corrections could not be determined for all triggers.

        SEE ALSO: merge_time_offsets.py
        """

        try:
            dfTemp = pd.read_csv( offsetTable)
        except:
            return  False
        else:
            self.dfOffset = dfTemp
            self.bOffsetLoaded = True
            return  True


# Class for ATVT file header:
class atvt_header( object ):
    """Class for handling the ATVT data file header
    """

    def __init__( self ):
        # Initial values will be bogus

        # Values at the start of the header are most important
        self.version = 0        # ATVT version (16-bit integer)
        self.size    = 0        # Header size in bytes

        # Trigger date, time & type (note: time is typically approximate)
        self.year    = 0
        self.month   = 0
        self.day     = 0
        self.hour    = 0
        self.minute  = 0
        self.second  = 0
        self.uSecond = 0.
        self.trigType= 0

        # Clock rate info
        self.sampleRate = 0
        self.decimation = 0

        # DMA block (record) size
        self.samplesPerRec = 0

        # Channel configuration info
        self.chRange     = 0
        self.couplingId  = 0
        self.impedanceId = 0
        self.bwLimit     = 0

        # Pre- and Post-trigger lengths in terms of records
        self.preRec  = 0
        self.postRec = 0

        # ...and in samples & seconds
        self.preSamples   = 0
        self.postSamples  = 0
        self.totalSamples = 0
        self.preSec   = 0.
        self.postSec  = 0.
        self.totalSec = 0.


    def load_header( self, filename ):
        """Loads the header for the provided filename.  Returns True if successful,
        False otherwise.
        """

        try:
            fp = open( filename, 'rb' )
        except:
            return  False

        # Read version & size.  Note: will fail if zero file size
        try:
            version = np.fromfile( fp, dtype=np.int16, count=1 )[0]
            size    = np.fromfile( fp, dtype=np.int16, count=1 )[0]
        except:
            return  False
        else:
            self.version = version
            self.size    = size

        # Read trigger date/time/type
        self.year     = np.fromfile( fp, dtype=np.uint16, count=1 )[0]
        self.month    = np.fromfile( fp, dtype=np.uint16, count=1 )[0]
        self.day      = np.fromfile( fp, dtype=np.uint16, count=1 )[0]
        self.hour     = np.fromfile( fp, dtype=np.uint16, count=1 )[0]
        self.minute   = np.fromfile( fp, dtype=np.uint16, count=1 )[0]
        self.second   = np.fromfile( fp, dtype=np.uint16, count=1 )[0]
        self.uSecond  = np.fromfile( fp, dtype=np.float64,count=1 )[0]
        self.trigType = np.fromfile( fp, dtype=np.uint32, count=1 )[0]

        # Read (target) clock rate & DMA info
        self.sampleRate    = np.fromfile( fp, dtype=np.uint32, count=1 )[0]
        self.decimation    = np.fromfile( fp, dtype=np.uint32, count=1 )[0]
        self.samplesPerRec = np.fromfile( fp, dtype=np.uint32, count=1 )[0]

        # Read channel info
        self.chRange     = np.fromfile( fp, dtype=np.uint32, count=1 )[0]
        self.couplingId  = np.fromfile( fp, dtype=np.uint32, count=1 )[0]
        self.impedanceId = np.fromfile( fp, dtype=np.uint32, count=1 )[0]
        self.bwLimit     = np.fromfile( fp, dtype=np.uint32, count=1 )[0]

        # Pre/post trigger lengths (in records) are only present for >=14 version
        if self.version >= 14:
            self.preRec  = np.fromfile( fp, dtype=np.uint32, count=1 )[0]
            self.postRec = np.fromfile( fp, dtype=np.uint32, count=1 )[0]
        else:
            # For earlier versions, determine pre/post as follows...
            # Need to determine total number of records in file using file size:
            bytesPerSample = 2
            fileStat  = os.stat( filename )
            dataBytes = fileStat.st_size - self.size    # File minus header size
            numRec    = dataBytes / bytesPerSample / self.samplesPerRec

            # Set pre/post record counts according to trigger type:
            #   AUX - Always had same size pre-/post-triggers (1.0/1.0 sec)
            #   Manual (1 or 2): Always entirely pretrigger in operations (not true
            #     of tests, though that data was not saved)
            if self.trigType == 0:
                self.preRec  = numRec / 2
                self.postRec = numRec / 2
            else:
                self.preRec  = numRec
                self.postRec = 0

        self.preSamples   = self.preRec  * self.samplesPerRec
        self.postSamples  = self.postRec * self.samplesPerRec
        self.totalSamples = self.preSamples + self.postSamples

        self.preSec   = self.preSamples  / float( self.sampleRate )
        self.postSec  = self.postSamples / float( self.sampleRate )
        self.totalSec = self.preSec + self.postSec

        # Close file
        fp.close()

        return  True


# 24-bit SA (or acoustic, etc) data class:
class sa24( generic ):
    """Class for handling Bill Rison's 24-bit continuously sampled data which is
    commonly used in association with slow antennas.
    """

    def __init__( self ):
        generic.__init__( self )
        self.bytesPerSample = 3     # Will be set later (could be 4)
        self.dataType = np.uint8    # Read bytes, since time is intermingled with data
        self.fileExt  = '.dat'      # File name extension


    def get_search_path( self, date=None, daySec=None ):
        """sa24.get_search_path( date=None, daySec=None ):

        Returns a string corresponding to the expected search path based on the
        current base directory as well as the target date and time.  If date is not
        specified, then the date last set is used.  If no time is specified, then no
        attempt is made to include it in the search path.  The search path is (normally)
        assumed to be of the form: [baseDir]/YYMMDD/HH/MM/
        """

        # Append an OS path separater to base directory if not already there
        try:
            if self.baseDir[-1] != os.path.sep:
                basePath = self.baseDir + os.path.sep
            else:
                basePath = self.baseDir
        except:
            basePath = ''

        # Set date component of path
        if date is None:
            datePath = self.date.strftime('%y%m%d') + os.path.sep
        elif type(date) == datetime.date:
            datePath = date.strftime('%y%m%d') + os.path.sep
        else:
            datePath = ''

        # Set time component of path
        if daySec is None:
            timePath = ''
        else:
            hours   = np.floor(daySec / 3600)
            minutes = np.floor((daySec - hours*3600) / 60)
            timePath = '%2.2i%s%2.2i%s' % (hours, os.path.sep, minutes, os.path.sep)

        return  basePath + datePath + timePath


    def load( self, tStart, tEnd, date=None, filePrefix=None ):
        """sa24.load( tStart, tEnd, date=None, filePrefix=None ):

        Loads data within the specified interval (in seconds since midnight) on the
        given date (datetime.date object).  If date is not specified, the internal date
        last set is used.  If filePrefix is not specified, then the internal station ID
        text string is used.  A '.dat' extension is assumed for the files.
        """

        # If date specified and is valid, set internal copy
        if date is not None:
            if type(date) == datetime.date:
                self.date = date

        # Reset data arrays to empty
        self.init_data( )

        # Determine search paths - will be one for each whole 60 minute boundary
        pathList = []
        tSearch = tStart - (tStart % 60)
        while (tSearch < tEnd):
            pathList.append( self.get_search_path( daySec=tSearch ) )
            tSearch += 60

        # Set file prefix used for search
        if filePrefix is None:
            searchFile = self.idTxt + '*' + self.fileExt
        else:
            searchFile = filePrefix + '*' + self.fileExt

        # Load files
        self.fileList = []
        for pathTxt in pathList:
            fileSearch = pathTxt + searchFile
            file_i = glob.glob( fileSearch )
            if len(file_i) == 1:
                filePath = file_i[0]
                discard = self.load_file( filePath, bAppendData=True )

        # Strip out data which is outside of requested time range
        self.filter_data_time_range( tStart, tEnd )


    def load_file( self, filename, bAppendData=False, bAppendGPS=False, \
                   bOmitLastGPS=True ):
        """sa24.load_file( filename, bAppendData=False, bAppendGPS=False,
                         bOmitLastGPS=True):

        Loads data from the 24-bit slow antenna (or similar) file.  Code is based on
        Rison's "SaFrame.py"

        OPTIONS:
            bAppendData     Appends the data array instead of overwriting if true
            bAppendGPS      Appends GPS arrays...
            bOmitLastGPS    If appending GPS arrays, omits the last GPS array element
                             (which appears to be always duplicated in the next file)

        RETURNS:  True if successful, False otherwise
        """

        # Exit if file does not exist
        if not( os.path.exists( filename ) ):
            return  False

        U8_i = np.fromfile( filename, dtype=self.dataType )
        ii = np.arange( 0, len(U8_i) - 8 )
        jj = np.where( (U8_i[ii  ] == ord('@')) & (U8_i[ii+1] == ord('@')) & \
                       (U8_i[ii+2] == ord('H')) & (U8_i[ii+3] == ord('b')) )[0]

        # If no sync words were found, exit now (invalid file?)
        if len(jj) == 0:
            return  False
        iSec_i = ii[jj]


        self.bytesPerSample = 3
        if (np.sum( U8_i[(iSec_i[0]+36):(iSec_i[1]-1):4] ) == 0):
            self.bytesPerSample = 4

        tPPS_i = np.float32( U8_i[iSec_i+29] )*256 + np.float32( U8_i[iSec_i+30] )
        tAD_i  = np.float32( U8_i[iSec_i+31] )*256 + np.float32( U8_i[iSec_i+32] )

        fileMonth_i = U8_i[iSec_i+4]
        fileDay_i   = U8_i[iSec_i+5]
        fileYear_i  = np.uint16( U8_i[iSec_i+6] )*256 + np.uint16( U8_i[iSec_i+7] )
        fileHour_i  = U8_i[iSec_i+8]

        # Take care of day rollover
        if ((fileHour_i[-1] == 0) & (fileHour_i[-2] == 23)):
            fileHour_i[-1] = 24

        fileMinute_i = U8_i[iSec_i+9]
        fileSecond_i = U8_i[iSec_i+10]

        fileLat_i = np.float64( U8_i[iSec_i+11] )*256*256*256 + \
                    np.float64( U8_i[iSec_i+12] )*256*256 + \
                    np.float64( U8_i[iSec_i+13] )*256 + np.float64( U8_i[iSec_i+14] )
        ii = np.where( fileLat_i > 324000000.0 )[0]
        fileLat_i[ii] = fileLat_i[ii] - float(2**32)
        fileLat_i = fileLat_i * 90.0 / 324000000.0

        fileLon_i = np.float64( U8_i[iSec_i+15] )*256*256*256 + \
                    np.float64( U8_i[iSec_i+16] )*256*256 + \
                    np.float64( U8_i[iSec_i+17] )*256 + np.float64( U8_i[iSec_i+18] )
        ii = np.where( fileLon_i > 648000000.0 )[0]
        fileLon_i[ii] = fileLon_i[ii] - float(2**32)
        fileLon_i = fileLon_i * 180.0 / 648000000.0

        fileAlt_i = np.float64( U8_i[iSec_i+19] )*256*256*256 + \
                    np.float64( U8_i[iSec_i+20] )*256*256 + \
                    np.float64( U8_i[iSec_i+21] )*256 + np.float64( U8_i[iSec_i+22] )
        ii = np.where( fileAlt_i > 1800000.0 )[0]
        fileAlt_i[ii] = fileAlt_i[ii] - float(2**32)
        fileAlt_i = fileAlt_i / 100

        fileSatVis_i   = U8_i[iSec_i+25]
        fileSatTrack_i = U8_i[iSec_i+26]

        iData_i = np.array( [],dtype=np.uint32 )
        fileT_i = np.array( [],dtype=np.float32 )
        for jj in np.arange( 0, (len(iSec_i)-1) ):
            N = (iSec_i[jj+1] - iSec_i[jj] - 33) / self.bytesPerSample
            cps = round( 40000000.0/N )
            dns = tAD_i[jj] - tPPS_i[jj]
            dne = tAD_i[jj+1] - tPPS_i[jj+1]

            # Number of 40 MHz clock cycles between first and last PPS
            freq = N * cps + dns - dne
            dt = cps / freq    # time between samples

            if (tAD_i[jj] >= tPPS_i[jj]):
                tStart = (tAD_i[jj] - tPPS_i[jj]) / freq
            else:
                tStart = (tAD_i[jj] + 65536.0 - tPPS_i[jj]) / freq
            fileT_i = np.concatenate((fileT_i, fileHour_i[jj+1]*3600.0 + \
                                               fileMinute_i[jj+1]*60.0 + \
                                               fileSecond_i[jj+1] + tStart + \
                                               np.arange(0,N)*dt - 37*dt))
            iData_i = np.concatenate((iData_i, np.arange(iSec_i[jj]+33,iSec_i[jj+1],\
                                                         self.bytesPerSample)),1)

        fileRaw_i = np.float32( U8_i[iData_i] )*256*256 + \
                    np.float32( U8_i[iData_i+1] )*256 + np.float32( U8_i[iData_i+2] )
        Vmax = -5.0   # Potential gradient convention
        jj = np.where( fileRaw_i > float(2**23) )[0]
        fileRaw_i[jj] = fileRaw_i[jj]-float(2**24)
        # fileMV_i = -1.0*fileRaw_i *1000.0 * Vmax/float(2**23)

        # Append or overwrite raw/time data arrays
        if bAppendData:
            self.raw_i = np.append( self.raw_i, fileRaw_i )
            self.t_i   = np.append( self.t_i,   fileT_i )
            self.fileList.append( filename )
        else:
            self.raw_i = fileRaw_i
            self.t_i   = fileT_i
            self.fileList = [ filename ]

        # Set sample frequency (assume no dropouts)
        if self.t_i.size >= 2:
            dT = self.t_i[-1] - self.t_i[0]
            if dT > 0.:
                self.freq = (self.t_i.size - 1) / dT

        # Append or overwrite GPS arrays
        if bAppendGPS:
            if bOmitLastGPS:
                jj = np.arange( len( fileLat_i ) - 1 )
            else:
                jj = np.arange( len( fileLat_i ) )

            self.gpsLat_i = np.append( self.gpsLat_i, fileLat_i[jj] )
            self.gpsLon_i = np.append( self.gpsLon_i, fileLon_i[jj] )
            self.gpsAlt_i = np.append( self.gpsAlt_i, fileAlt_i[jj] )

            self.gpsSatVis_i   = np.append( self.gpsSatVis_i,   fileSatVis_i[jj] )
            self.gpsSatTrack_i = np.append( self.gpsSatTrack_i, fileSatTrack_i[jj] )
        else:
            self.gpsLat_i = fileLat_i
            self.gpsLon_i = fileLon_i
            self.gpsAlt_i = fileAlt_i
            self.gpsSatVis_i   = fileSatVis_i
            self.gpsSatTrack_i = fileSatTrack_i

        return  True
