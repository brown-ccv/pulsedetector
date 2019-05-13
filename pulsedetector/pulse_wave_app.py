#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Isa Restrepo
# @Date:   2015-03-18 10:17:06
# @Last Modified by:   Isa Restrepo
# @Last Modified time: 2015-04-05 11:09:33


import sys, os
import tkinter
import tkinter.ttk
import tkinter.filedialog

import tkinter.filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# import sklearn.utils.weight_vector

# import matplotlib
# matplotlib.use('macosx')
import matplotlib.pyplot as plt
from . import lib.pulse_align_ios
from .lib.pulse_align_ios import getPulseWaveFromFileApp

from .get_pulse import getPulseApp

from . import lib.video_process_util as vidPro



class StdoutRedirector(object):
    def __init__(self,text_widget):
        self.text_space = text_widget

    def write(self,string):
        self.text_space.insert('end', string)
        self.text_space.see('end')


class Application(tkinter.Frame):

    def __init__(self, master=None):
        tkinter.Frame.__init__(self, master)
        self.grid()
        self.grid_columnconfigure(0, weight=1)


        # define options for opening or saving a file
        self.file_opt = options = {}
        options['defaultextension'] = ''
        options['filetypes'] = [('all files', '.*'), ('movie files', '.mp4'), ('data files', '.npy'), ('text files', '.txt')]
        options['initialdir'] = ''
        options['initialfile'] = ''
        options['parent'] = master
        options['title'] = 'Choose File'

        self.createWidgets()


    def createWidgets(self):
        # form = Tkinter.Tk()

        getFld = tkinter.IntVar()

        # self.title('File Parser')

        numCols = 30

        stepOne = tkinter.LabelFrame(self, text=" 1. File Details: ")
        stepOne.grid(row=0, columnspan=numCols, sticky='we', \
                     padx=5, pady=5, ipadx=5, ipady=5)

        for x in range(0,numCols):
            stepOne.grid_columnconfigure(x, weight=1)

        # helpLf = Tkinter.LabelFrame(self, text=" Quick Help ")
        # helpLf.grid(row=0, column=9, columnspan=2, rowspan=8, \
        #             sticky='NS', padx=5, pady=5)
        # helpLbl = Tkinter.Label(helpLf, text="Help will come - ask for it.")
        # helpLbl.grid(row=0)

        stepTwo = tkinter.LabelFrame(self, text=" 2. Processing: ")
        stepTwo.grid(row=2, columnspan=numCols, sticky='WE', \
                     padx=5, pady=5, ipadx=5, ipady=5)

        self.stepThree = tkinter.LabelFrame(self, text=" 3. Visualization: ")
        self.stepThree.grid(row=3, columnspan=numCols, sticky='WE', \
                       padx=5, pady=5, ipadx=5, ipady=5)

        #*************** Step 1 Widgets *****************************
        inFileLbl = tkinter.Label(stepOne, text="Select Video File:")
        inFileLbl.grid(row=0, column=0, sticky='W', padx=5, pady=2)

        self.inFileTxt = tkinter.Entry(stepOne)
        self.inFileTxt.grid(row=0, column=1, columnspan=30, sticky="WE", pady=3)
        self.videofile = '/Users/isa/Desktop/ben-test/VID_20150915_123758391.mp4'
        self.inFileTxt.insert(0, self.videofile)

        inFileBtn = tkinter.Button(stepOne, text="Browse ...", command=self.askopenVideoFile)
        inFileBtn.grid(row=0, column=31, sticky='E', padx=5, pady=2)

        inDataFileLbl = tkinter.Label(stepOne, text="Select Data File:")
        inDataFileLbl.grid(row=1, column=0, sticky='W', padx=5, pady=2)

        self.inDataFileTxt = tkinter.Entry(stepOne)
        self.inDataFileTxt.grid(row=1, column=1, columnspan=30, sticky="WE", pady=3)
        # self.datafile = '/Users/isa/Desktop/ben-test-results/VID_20150915_123720605/rgb-50-1.npy'
        # self.inDataFileTxt.insert(0, self.datafile)

        inDataFileBtn = tkinter.Button(stepOne, text="Browse ...", command=self.askopenDataFile)
        inDataFileBtn.grid(row=1, column=31, sticky='E', padx=5, pady=2)

        outFileLbl = tkinter.Label(stepOne, text="Output Directory:")
        outFileLbl.grid(row=2, column=0, sticky='W', padx=5, pady=2)

        self.outDirTxt = tkinter.Entry(stepOne)
        self.outDirTxt.grid(row=2, column=1, columnspan=30, sticky="WE", pady=2)
        self.outputDir = '/Users/isa/Desktop/tests'
        self.outDirTxt.insert(0, self.outputDir)

        outFileBtn = tkinter.Button(stepOne, text="Browse ...", command=self.askopendirectory)
        outFileBtn.grid(row=2, column=31, sticky='E', padx=5, pady=2)

        #*************** Step 2 Widgets *****************************

        preprocessBtn = tkinter.Button(stepTwo, text="Generate Data File", \
                                    command=self.preProcess)
        preprocessBtn.grid(row=5, column=0, sticky='W', padx=5, pady=2)

        processBtn = tkinter.Button(stepTwo, text="Process", \
                                    command=self.findPulse)
        processBtn.grid(row=5, column=5, sticky='W', padx=5, pady=2)

        # getFldChk = Tkinter.Checkbutton(stepTwo, \
        #                        text="Save results to disk",\
        #                        onvalue=1, offvalue=0)
        # getFldChk.grid(row=4, column=5, columnspan=3, pady=2, sticky='W')

        #*************** Step 3 Widgets *****************************

        self.plot_tabs = tkinter.ttk.Notebook(self.stepThree)
        self.data_tab = tkinter.ttk.Frame(self.plot_tabs); # first page, which would get widgets gridded into it
        self.sVSr_tab = tkinter.ttk.Frame(self.plot_tabs); # second page
        self.avg_tab = tkinter.ttk.Frame(self.plot_tabs); # second page
        self.console_tab = tkinter.ttk.Frame(self.plot_tabs); # first page, which would get widgets gridded into it


        self.plot_tabs.add(self.avg_tab, text='Pulse')
        self.plot_tabs.add(self.data_tab, text='Data')
        self.plot_tabs.add(self.sVSr_tab, text='Peaks')
        self.plot_tabs.add(self.console_tab, text='Console')


        self.plot_tabs.pack(fill=tkinter.BOTH, expand=1)


        #Data Figure
        self.smooth_data_fig = plt.figure()
        # self.plot_ax = self.fig.add_subplot(111)
        self.smooth_data_canvas = FigureCanvasTkAgg(self.smooth_data_fig, master=self.data_tab)
        self.smooth_data_canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        # self.smooth_data_canvas.show()

        data_toolbar = NavigationToolbar2TkAgg(self.smooth_data_canvas, self.data_tab)
        data_toolbar.pack()
        data_toolbar.update()

        #peaks Figure
        self.peaks_fig = plt.figure()
        # self.plot_ax = self.fig.add_subplot(111)
        self.data_canvas = FigureCanvasTkAgg(self.peaks_fig, master=self.sVSr_tab)
        self.data_canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        # self.data_canvas.show()

        data_toolbar = NavigationToolbar2TkAgg(self.data_canvas, self.sVSr_tab)
        data_toolbar.pack()
        data_toolbar.update()

        #avg Figure
        self.good_pulse_fig = plt.figure()
        # self.plot_ax = self.fig.add_subplot(111)
        self.avg_canvas = FigureCanvasTkAgg(self.good_pulse_fig, master=self.avg_tab)
        self.avg_canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        # self.avg_canvas.show()

        data_toolbar = NavigationToolbar2TkAgg(self.avg_canvas, self.avg_tab)
        data_toolbar.pack()
        data_toolbar.update()

        #Console text
        self.console = tkinter.Text(self.console_tab, height=35, width=90)
        self.console.pack()
        sys.stdout = StdoutRedirector(self.console)



    def askopenVideoFile(self):

      """Gets filename. The file needs to be opened by your own code.
      """

      # get filename
      self.videofile = tkinter.filedialog.askopenfilename(**self.file_opt)
      self.inFileTxt.delete(0, tkinter.END)
      self.inFileTxt.insert(0, self.videofile)

    def askopenDataFile(self):

      """Gets filename. The file needs to be opened by your own code.
      """

      # get filename
      self.datafile = tkinter.filedialog.askopenfilename(**self.file_opt)
      self.inDataFileTxt.delete(0, tkinter.END)
      self.inDataFileTxt.insert(0, self.datafile)


    def askopendirectory(self):

      """Gets a directory path
      """

      # get dir
      self.outputDir = tkinter.filedialog.askdirectory()
      self.outDirTxt.delete(0, tkinter.END)
      self.outDirTxt.insert(0, self.outputDir)


    def preProcess(self):
        """Traverses a video file, reads color values and
        saves them to a fil
        """

        filename , fileext = os.path.splitext(self.videofile)
        newVideoFile = filename + '_halfed' + '.mov'
        print("Resizing ",  self.videofile, " to ", newVideoFile)

        # vidPro.resize(self.videofile, newVideoFile, 0.5)
        
        print("Extracting average info across video")

        App = getPulseApp(videofile   =  filename + '_halfed' + '.mov',
                          roi_percent =  0.5,
                          find_faces  =  False,
                          color_space =  'rgb',
                          output_dir  =  self.outputDir,
                          no_gui      =  True,
                          grid_size   =  1)

        App.run()

    def findPulse(self):
        # self.videofile = self.inFileTxt.get
        self.pulseApp = getPulseWaveFromFileApp(videofile   =  self.videofile,
                                                datafile = self.datafile,
                                                output_dir  =  self.outputDir)

        print("Done 5 ")
        print('Smoothing Data')
        self.pulseApp.smooth_data()
        self.pulseApp.plot_bandpass_data(smooth_data_fig=self.smooth_data_fig)
        print('Averaging Pulses')
        self.pulseApp.find_pulses(good_pulse_fig=self.good_pulse_fig, peaks_fig=self.peaks_fig)
        self.pulseApp.close_fig_pdf()

        self.smooth_data_canvas.show()

        self.data_canvas.show()

        self.avg_canvas.show()


if __name__ == '__main__':

    root = tkinter.Tk()
    root.columnconfigure(0, weight=1)
    app = Application(root)
    app.master.title('Pulse Analyzer')
    app.grid(sticky="nswe")
    # app.master.grid_columnconfigure(0, weight=1)
    # app.master.grid_rowconfigure(0, weight=1)
    # app.master.resizable(True, False)
    app.mainloop()



