#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2015-03-31 22:46:49
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2015-05-28 15:04:19


import numpy as np
import matplotlib.pyplot as plt





Tfis = open('/Users/isa/Desktop/tests/Ben_2015.2.22_anteriorLeftAntebrachium_1/avg_pulse.txt', 'r')
data1 = np.genfromtxt(Tfis)
Tfis2 = open('/Users/isa/Desktop/tests/Ben_2015.2.22_anteriorLeftAntebrachium_wound_6/avg_pulse.txt', 'r')
data2 = np.genfromtxt(Tfis2)


#integrate the max
data = np.vstack((data1[1,:],data2[1,:]))
max_data = np.max(data, 0)
min_data = np.min(data, 0)
int_max = np.trapz(max_data, data1[0,:])
int_min = np.trapz(min_data, data1[0,:])
# print data1[0,:] - data2[0,:]
area = int_max - int_min
print(area)
# np.max([data1[1,:],data1[1,:]])

#Plot
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.clear()
ax.set_ylim((0,1))
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Pulse Amplitude', fontsize=14)

ax.plot(data1[0,:], data1[1,:] , 'g-', label='antebrachium_1')
ax.plot(data2[0,:], data2[1,:], 'r-', label='wound_6')

ax.fill_between(data2[0,:], data1[1,:], data2[1,:])


ax.text(0.5, 0.5,"Area: {:.2f}".format(area) ,horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)

ax.legend(loc='best', frameon=False);


plt.show()



