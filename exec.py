'''
My personal script for processing intf data to locate nbes

This is not part of the NBE package.

NBE(Narrow Bipolar Event) locating and clasification script

1 Loads in data from a box accounts
2 attempts to locate an nbe from the sferic signal(channel D from the array)
3 Processes the data to aproximate where the sources of the VHF signals are
4 From the avg change in elevation over time of the sources and the initial
  deviation from the base line the sferic signal: determin the breakdown
  polarity of the NBE
'''

import matplotlib.pyplot as plt
import os.path as pat

import os
import time






def main():
    pass




if __name__ == '__main__':
    start =  time.time()
    try:
        Main()
    except KeyboardInterrupt:
        print('Exit')
    end = time.time()

print('\n#### Computation time:',str(round((end - start)/60,4)) + ' min ####')
