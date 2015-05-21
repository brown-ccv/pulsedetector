#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2015-03-18 10:17:06
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2015-04-05 11:09:33


import sys
import Tkinter
import ttk
import tkFileDialog

import FileDialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# import sklearn.utils.weight_vector

# import matplotlib
# matplotlib.use('macosx')
import matplotlib.pyplot as plt
import lib.pulse_align_ios
from lib.pulse_align_ios import getPulseWaveFromFileApp

class StdoutRedirector(object):
    def __init__(self,text_widget):
        self.text_space = text_widget

    def write(self,string):
        self.text_space.insert('end', string)
        self.text_space.see('end')


class Application(Tkinter.Frame):

    def __init__(self, master=None):
        Tkinter.Frame.__init__(self, master)
        self.grid()
        self.grid_columnconfigure(0, weight=1)


        # define options for opening or saving a file
        self.file_opt = options = {}
        options['defaultextension'] = '.txt'
        options['filetypes'] = [('all files', '.*'), ('movie files', '.mp4')]
        options['initialdir'] = 'C:\\'
        options['initialfile'] = 'myfile.mp4'
        options['parent'] = master
        options['title'] = 'Choose Video File'

        self.createWidgets()


    def createWidgets(self):
        # form = Tkinter.Tk()

        getFld = Tkinter.IntVar()

        # self.title('File Parser')

        numCols = 30

        stepOne = Tkinter.LabelFrame(self, text=" 1. File Details: ")
        stepOne.grid(row=0, columnspan=numCols, sticky='we', \
                     padx=5, pady=5, ipadx=5, ipady=5)

        for x in range(0,numCols):
            stepOne.grid_columnconfigure(x, weight=1)

        # helpLf = Tkinter.LabelFrame(self, text=" Quick Help ")
        # helpLf.grid(row=0, column=9, columnspan=2, rowspan=8, \
        #             sticky='NS', padx=5, pady=5)
        # helpLbl = Tkinter.Label(helpLf, text="Help will come - ask for it.")
        # helpLbl.grid(row=0)

        stepTwo = Tkinter.LabelFrame(self, text=" 2. Process: ")
        stepTwo.grid(row=2, columnspan=numCols, sticky='WE', \
                     padx=5, pady=5, ipadx=5, ipady=5)

        self.stepThree = Tkinter.LabelFrame(self, text=" 3. Display Plot: ")
        self.stepThree.grid(row=3, columnspan=numCols, sticky='WE', \
                       padx=5, pady=5, ipadx=5, ipady=5)

        #*************** Step 1 Widgets *****************************
        inFileLbl = Tkinter.Label(stepOne, text="Select Video File:")
        inFileLbl.grid(row=0, column=0, sticky='W', padx=5, pady=2)

        self.inFileTxt = Tkinter.Entry(stepOne)
        self.inFileTxt.grid(row=0, column=1, columnspan=30, sticky="WE", pady=3)
        # self.videofile = '/Users/isa/GoogleDrive/VACUScan/Research/Data/20150222_Epi_trial/Ben_2015.2.22_anteriorLeftAntebrachium_1.mp4'
        # self.inFileTxt.insert(0, self.videofile)


        inFileBtn = Tkinter.Button(stepOne, text="Browse ...", command=self.askopenfilename)
        inFileBtn.grid(row=0, column=31, sticky='E', padx=5, pady=2)

        outFileLbl = Tkinter.Label(stepOne, text="Output Directory:")
        outFileLbl.grid(row=1, column=0, sticky='W', padx=5, pady=2)

        self.outDirTxt = Tkinter.Entry(stepOne)
        self.outDirTxt.grid(row=1, column=1, columnspan=30, sticky="WE", pady=2)
        # self.outputDir = '/Users/isa/Desktop/tests'
        # self.outDirTxt.insert(0, self.outputDir)

        outFileBtn = Tkinter.Button(stepOne, text="Browse ...", command=self.askopendirectory)
        outFileBtn.grid(row=1, column=31, sticky='E', padx=5, pady=2)

        #*************** Step 2 Widgets *****************************

        processBtn = Tkinter.Button(stepTwo, text="Process", \
                                    command=self.findPulse)
        processBtn.grid(row=4, column=0, sticky='W', padx=5, pady=2)

        # getFldChk = Tkinter.Checkbutton(stepTwo, \
        #                        text="Save results to disk",\
        #                        onvalue=1, offvalue=0)
        # getFldChk.grid(row=4, column=5, columnspan=3, pady=2, sticky='W')

        #*************** Step 3 Widgets *****************************

        self.plot_tabs = ttk.Notebook(self.stepThree)
        self.data_tab = ttk.Frame(self.plot_tabs); # first page, which would get widgets gridded into it
        self.sVSr_tab = ttk.Frame(self.plot_tabs); # second page
        self.avg_tab = ttk.Frame(self.plot_tabs); # second page
        self.console_tab = ttk.Frame(self.plot_tabs); # first page, which would get widgets gridded into it


        self.plot_tabs.add(self.avg_tab, text='Pulse')
        self.plot_tabs.add(self.data_tab, text='Data')
        self.plot_tabs.add(self.sVSr_tab, text='Peaks')
        self.plot_tabs.add(self.console_tab, text='Console')


        self.plot_tabs.pack(fill=Tkinter.BOTH, expand=1)


        #Data Figure
        self.smooth_data_fig = plt.figure()
        # self.plot_ax = self.fig.add_subplot(111)
        self.smooth_data_canvas = FigureCanvasTkAgg(self.smooth_data_fig, master=self.data_tab)
        self.smooth_data_canvas.get_tk_widget().pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)
        # self.smooth_data_canvas.show()

        data_toolbar = NavigationToolbar2TkAgg(self.smooth_data_canvas, self.data_tab)
        data_toolbar.pack()
        data_toolbar.update()

        #peaks Figure
        self.peaks_fig = plt.figure()
        # self.plot_ax = self.fig.add_subplot(111)
        self.data_canvas = FigureCanvasTkAgg(self.peaks_fig, master=self.sVSr_tab)
        self.data_canvas.get_tk_widget().pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)
        # self.data_canvas.show()

        data_toolbar = NavigationToolbar2TkAgg(self.data_canvas, self.sVSr_tab)
        data_toolbar.pack()
        data_toolbar.update()

        #avg Figure
        self.good_pulse_fig = plt.figure()
        # self.plot_ax = self.fig.add_subplot(111)
        self.avg_canvas = FigureCanvasTkAgg(self.good_pulse_fig, master=self.avg_tab)
        self.avg_canvas.get_tk_widget().pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=1)
        # self.avg_canvas.show()

        data_toolbar = NavigationToolbar2TkAgg(self.avg_canvas, self.avg_tab)
        data_toolbar.pack()
        data_toolbar.update()

        #Console text
        self.console = Tkinter.Text(self.console_tab, height=35, width=90)
        self.console.pack()
        sys.stdout = StdoutRedirector(self.console)



    def askopenfilename(self):

      """Gets filename. The file needs to be opened by your own code.
      """

      # get filename
      self.videofile = tkFileDialog.askopenfilename(**self.file_opt)
      self.inFileTxt.delete(0, Tkinter.END)
      self.inFileTxt.insert(0, self.videofile)


    def askopendirectory(self):

      """Gets a directory path
      """

      # get dir
      self.outputDir = tkFileDialog.askdirectory()
      self.outDirTxt.delete(0, Tkinter.END)
      self.outDirTxt.insert(0, self.outputDir)

    def findPulse(self):
        # self.videofile = self.inFileTxt.get
        self.pulseApp = getPulseWaveFromFileApp(videofile   =  self.videofile,
                                           output_dir  =  self.outputDir)

       
        print 'Smoothing Data'
        self.pulseApp.smooth_data()
        self.pulseApp.plot_bandpass_data(smooth_data_fig=self.smooth_data_fig)
        print 'Averaging Pulses'
        self.pulseApp.find_pulses(good_pulse_fig=self.good_pulse_fig, peaks_fig=self.peaks_fig)
        self.pulseApp.close_fig_pdf()

        self.smooth_data_canvas.show()

        self.data_canvas.show()

        self.avg_canvas.show()


if __name__ == '__main__':

    root = Tkinter.Tk()
    root.columnconfigure(0, weight=1)
    app = Application(root)
    app.master.title('Pulse Analyzer')
    app.grid(sticky="nswe")
    # app.master.grid_columnconfigure(0, weight=1)
    # app.master.grid_rowconfigure(0, weight=1)
    # app.master.resizable(True, False)
    app.mainloop()


