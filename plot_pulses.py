#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2015-03-31 22:46:49
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2015-03-31 23:00:27


import numpy as np
import matplotlib.pyplot as plt

Tfis = open('/Users/isa/Desktop/tests/Ben_2015.2.22_anteriorLeftAntebrachium_wound_1/avg_pulse.txt', 'r')
data1 = np.genfromtxt(Tfis)
Tfis2 = open('/Users/isa/Desktop/tests/Ben_2015.2.22_anteriorLeftAntebrachium_wound_6/avg_pulse.txt', 'r')
data2 = np.genfromtxt(Tfis2)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.clear()
ax.set_ylim((0,1))
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Pulse Amplitude', fontsize=14)

ax.plot(data1[0,:], data1[1,:] , 'g-', label='wound_1')
ax.plot(data2[0,:], data2[1,:], 'r-', label='wound_6')

ax.legend(loc='best', frameon=False);


plt.show()