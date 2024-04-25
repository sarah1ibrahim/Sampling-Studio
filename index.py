from scipy.fft import fft
import scipy.signal as sig
from scipy import interpolate
from scipy.interpolate import interp1d

from numpy import *
import numpy as np
import pandas as pd

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QGraphicsScene
import pyqtgraph as pg

import sys
import os

matplotlib.use("Qt5Agg")


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=20, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        uic.loadUi(r"main.ui", self)

        self.setWindowTitle("Sampling Studio")
        # self.setWindowIcon(QIcon('images\icon.jpg'))

        # initialize variables
        self.mag = 0
        self.freq = 1
        self.phase = 0

        self.sinusoidal = 0
        self.signalSum = 0
        self.signal = 0
        self.signalName = ""
        self.max_frequency = 1
        self.time = arange(0.0, 1.0, 0.001)
        print(len(self.time))

        self.values = None
        self.valueList = None
        self.fname1 = None
        self.maxFreq = None
        self.sampling_freq = 1.0
        self.sampled_values_points = []

        # self.max_frequency_noisy = None

        # dictionary with signal variables
        self.signaldict = dict()

        # initialize variables for matplotlib plotting
        self.canvas1 = MplCanvas(self.widget1, width=20, height=5, dpi=100)
        self.layout1 = QtWidgets.QVBoxLayout()
        self.layout1.addWidget(self.canvas1)

        self.canvas2 = MplCanvas(self.widget2, width=20, height=5, dpi=100)
        self.layout2 = QtWidgets.QVBoxLayout()
        self.layout2.addWidget(self.canvas2)

        self.canvas3 = MplCanvas(self.widget3)
        self.layout3 = QtWidgets.QVBoxLayout()
        self.layout3.addWidget(self.canvas3)
        self.plot_in_widget3 = False

        self.canvas4 = MplCanvas(self.widget4)
        self.layout4 = QtWidgets.QVBoxLayout()
        self.layout4.addWidget(self.canvas4)

        self.canvas5 = MplCanvas(self.widget5)
        self.layout5 = QtWidgets.QVBoxLayout()
        self.layout5.addWidget(self.canvas5)

        # connections
        self.load_option.triggered.connect(self.load_signal)
        self.add_button.clicked.connect(self.display_sum)
        self.remove_button.clicked.connect(self.remove_signal)
        self.confirm_button.clicked.connect(self.signalConfirm)
        self.reset_button.clicked.connect(
            self.reset_plot
        )  # Connect the reset button to the reset_plot method
        self.add_noise_button.clicked.connect(self.add_noise)

        # Get the model of the combo box &
        # Set the flags of the placeholder item to not be enabled or selectable
        self.freq_combobox.model().item(0).setEnabled(False)
        # Set the placeholder item to be viewed
        self.freq_combobox.setCurrentIndex(0)
        self.freq_combobox.currentIndexChanged.connect(self.update_frequency)

        # Add placeholder item
        self.signal_combobox.addItem("select signal components")
        # Get the model of the combo box &
        # Set the flags of the placeholder item to not be enabled or selectable
        self.signal_combobox.model().item(0).setEnabled(False)
        # Set the placeholder item to be viewed
        self.signal_combobox.setCurrentIndex(0)
        self.signal_combobox.currentIndexChanged.connect(self.display_selected_signal)

        self.graph = pg.PlotItem()

        pg.PlotItem.hideAxis(self.graph, "left")
        pg.PlotItem.hideAxis(self.graph, "bottom")

        # for sampling composed:
        self.horizontalSlider_2.setValue(10)
        self.horizontalSlider_2.valueChanged.connect(self.sample_signal)

        self.composed_signal_xdata = []
        self.loaded_signal_xdata = []
        self.noisy_signal_xdata = []
        self.noisy_signal_ydata = []
        self.fname_1 = False

        # Noise
        self.noisy = False
        self.snr_slider.hide()
        self.snr_lcd.hide()
        self.snr_label.hide()
        self.snr_slider.setMinimum(0)  # set minimum value
        self.snr_slider.setMaximum(100)  # set maximum value

        # Initialize the default parameters for the sine wave
        self.mag = 1.0  # Default magnitude
        self.freq = 1.0  # Default frequency
        self.phase = 0.0  # Default phase

        # Create the default sine wave
        self.sinusoidal = self.signal_parameters(self.mag, self.freq, self.phase)

        # Display the default sine wave on Canvas 1
        self.signal_plot(self.canvas1, self.widget1, self.layout1, self.sinusoidal)

        # Display default values in the textbox
        self.frequency.setText(str(self.freq))
        self.magnitude.setText(str(self.mag))
        self.phase_shift.setText(str(self.phase))
        self.add_label.setText("Default Signal")

        # Connect the textChanged signals to update_canvas1 slot
        self.frequency.textChanged.connect(self.update_canvas1)
        self.magnitude.textChanged.connect(self.update_canvas1)
        self.phase_shift.textChanged.connect(self.update_canvas1)

    # define signal using given parameters
    def signal_parameters(self, magnitude, frequency, phase):
        omega = 2 * pi * frequency
        theta = phase * pi / 180
        return magnitude * sin(omega * self.time + theta)

    # signal plotter function
    def signal_plot(self, canvas, widget, layout, signal):
        canvas.axes.cla()
        canvas.axes.plot(self.time, signal)
        canvas.draw()
        widget.setCentralItem(self.graph)
        widget.setLayout(layout)

    # return signal function/values through data and used to remove the signal
    def get_signal(self):
        self.signalName = self.signal_combobox.currentText()
        self.valueList = self.signaldict[self.signalName]
        self.signal = self.signal_parameters(
            self.valueList[0], self.valueList[1], self.valueList[2]
        )

    # sum of generated signals
    def display_sum(self):
        self.name = self.add_label.text()

        # check if the name already exist
        if self.name in self.signaldict:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("This label already exists. Please enter a different label.")
            msg.setWindowTitle("Input Error")
            msg.exec_()
            return
        if not self.name.strip():  # Check if the label is empty
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please enter a label.")
            msg.setWindowTitle("Warning")
            msg.exec_()
            return

        self.signaldict[self.name] = self.mag, self.freq, self.phase
        # Append the frequency to the label when adding it to the combobox
        self.signal_combobox.addItem(f"{self.name} ({self.freq} Hz)")
        self.signalSum += self.sinusoidal

        if len(self.composed_signal_xdata) < len(self.signalSum):
            self.composed_signal_xdata.extend(
                list(self.time[len(self.composed_signal_xdata) :])
            )
            # print(len(self.composed_signal_xdata))
            # (this to ensure that (which becomes self.yData in signalConfirm) always have the same length of self.composed_signal_xdata
        self.signal_plot(self.canvas2, self.widget2, self.layout2, self.signalSum)

    def update_canvas1(self):
        mag_text = self.magnitude.text()
        freq_text = self.frequency.text()
        phase_text = self.phase_shift.text()

        if mag_text:  # Check if the magnitude field is not empty
            self.mag = float(mag_text)

        if freq_text:  # Check if the frequency field is not empty
            self.freq = float(freq_text)

        if phase_text:  # Check if the phase shift field is not empty
            self.phase = float(phase_text)

        self.sinusoidal = self.signal_parameters(self.mag, self.freq, self.phase)

        # Update Canvas 1
        self.signal_plot(self.canvas1, self.widget1, self.layout1, self.sinusoidal)

    # Display the chosen signal component from the combobox
    def display_selected_signal(self):
        # Get the selected signal name from the combo box
        signal_name = self.signal_combobox.currentText().split(" (")[0]
        # Retrieve signal parameters from dictionary
        self.mag, self.freq, self.phase = self.signaldict[signal_name]

        # Display parameters in input boxes
        self.magnitude.setText(str(self.mag))
        self.frequency.setText(str(self.freq))
        self.phase_shift.setText(str(self.phase))
        self.add_label.setText(str(signal_name))

        # Calculate and plot the signal
        self.sinusoidal = self.signal_parameters(self.mag, self.freq, self.phase)
        self.signal_plot(self.canvas1, self.widget1, self.layout1, self.sinusoidal)

    def remove_signal(self):
        if self.signal_combobox.count() == 1:
            self.signalSum = [0] * (len(self.time))
            self.signaldict.clear()
            self.signal_combobox.clear()
        else:
            selected_index = self.signal_combobox.currentIndex()
            self.get_signal()
            self.signal_combobox.removeItem(selected_index)
            self.signalSum -= self.signal
            self.signaldict.pop(self.signalName, None)
        self.signal_plot(self.canvas2, self.widget2, self.layout2, self.signalSum)

    # confirm signal to main illustrator
    def signalConfirm(self):
        self.reset_all()
        self.xData_composed = self.time
        self.yData_composed = self.signalSum
        max_freq = []
        for i in self.signaldict:
            max_freq.append(self.signaldict[i][1])
        self.maxFreq = max(max_freq)
        self.tabWidget.setCurrentIndex(
            1
        )  # this to move you to the tab2 when press confirm
        self.canvas3.axes.autoscale()
        self.canvas3.axes.clear()
        self.canvas3.axes.plot(self.xData_composed, self.yData_composed)
        self.canvas3.draw()
        self.widget3.setLayout(
            self.layout3
        )  # i should add this line to fix problem of not adding signal to canvas3

        self.fname_1 = False
        self.plot_in_widget3 = True  # set attribute to True after drawing plot

    def load_signal(self):
        self.fname1, _ = QFileDialog.getOpenFileName(
            self, "Open file", "./", "CSV Files (*.csv);;Text Files (*.txt)"
        )
        # Check if a file was selected
        if not self.fname1:
            return
        self.reset_all()
        self.fname_1 = True
        path1 = self.fname1
        data1 = pd.read_csv(path1)
        if len(data1.columns) >= 3:  # Check if there are at least three columns
            self.time_col = data1.iloc[:, 0][:1000]
            self.values = data1.iloc[:, 1][:1000]
            self.max_frequency = data1.iloc[:, 2][
                :1000
            ].max()  # this to get max freq from 3rd col
        else:
            y_values = (
                data1.iloc[:, 0] / 1000
            )  # Convert from uV to mV # y-values are the only column
            self.values = y_values[:1000]  # take only the first 1000 values
            time_values = np.linspace(
                0, 20, len(y_values)
            )  # Generate x-values for a 20-second interval
            self.time_col = time_values[:1000]  # take only the first 1000 values
            self.max_frequency = 50

        # self.loaded_signal_xdata.extend(list(self.time_col[:1000]))
        # self.t_max = self.values[:1000].max()
        # self.t_min = self.values[:1000].min()
        self.loaded_signal_xdata.extend(list(self.time_col))
        self.t_max = self.values.max()
        self.t_min = self.values.min()
        self.tabWidget.setCurrentIndex(1)  # this to move you to the tab2 when loading a signal
        self.canvas3.axes.clear()
        self.canvas3.axes.plot(self.time_col, self.values)
        # # Plot only the first 1000 points
        # self.canvas3.axes.plot(self.time_col[:1000], self.values[:1000])
        self.canvas3.draw()
        self.widget3.setCentralItem(self.graph)
        self.widget3.setLayout(self.layout3)
        self.plot_in_widget3 = True  # set attribute to True after drawing plot

    def update_frequency(self):
        # Get the selected frequency
        selected_frequency = self.freq_combobox.currentText()

        # Check if a valid frequency has been selected
        if selected_frequency not in ["select frequency", ""]:
            # Enable the slider
            self.horizontalSlider_2.setEnabled(True)
            # Reset the slider
            self.horizontalSlider_2.setValue(self.horizontalSlider_2.minimum())
            # Update the plots
            self.sample_signal()
        else:
            # Disable the slider
            self.horizontalSlider_2.setEnabled(False)
            
    def reset_all(self):
        self.reset_noise()
        # Reset variables to default
        self.loaded_signal_xdata = []  # Clear loaded_signal_xdata
        self.sampled_values_points = []
        self.xData_composed = []
        self.yData_composed = []
        self.plot_in_widget3 = False
        self.fname_1 = False
        self.noisy = False

        # # Clear text boxes
        # self.magnitude.clear()
        # self.frequency.clear()
        # self.phase_shift.clear()
        # self.add_label.clear()

        # Clear comboboxes
        # self.signal_combobox.clear()
        self.freq_combobox.setCurrentIndex(0)

        # Clear plots
        # self.canvas1.axes.clear()
        # self.canvas2.axes.clear()
        self.canvas3.axes.clear()
        self.canvas4.axes.clear()
        self.canvas5.axes.clear()

        # Redraw canvases
        # self.canvas1.draw()
        # self.canvas2.draw()
        self.canvas3.draw()
        self.canvas4.draw()
        self.canvas5.draw()


    def sample_signal(self):
        if self.noisy:  # Check if a noisy signal is present
            xData = self.noisy_signal_xdata
            yData = self.noisy_signal_ydata
            if self.fname_1:
                max_frequency = self.max_frequency
            else:
                max_frequency = self.maxFreq
            print("2 enter here")
        elif self.fname_1:  # Check if a signal is loaded
            xData = self.loaded_signal_xdata
            yData = self.values
            max_frequency = self.max_frequency
            print("enter here")
        else:  # Otherwise, use composed signal
            xData = self.composed_signal_xdata
            yData = self.yData_composed
            max_frequency = self.maxFreq

        factor = self.horizontalSlider_2.value()  # set value of slider

        if self.freq_combobox.currentText() == "Actual frequency":
            self.sampling_freq = factor
        elif self.freq_combobox.currentText() == "Normal frequency":
            self.sampling_freq = 2 * factor * max_frequency

        # this just to avoid divide by 0 when slider be zero
        if self.sampling_freq == 0:
            self.sampling_freq = 0.01

        # check the length
        # if len(xData) > len(yData):
        #     xData = xData[:len(yData)]
        # elif len(yData) > len(xData):
        #     yData = yData[:len(xData)]

        # Calculate step size
        step_size = (xData[-1] - xData[0]) / len(xData)
        t_sample = 1 / self.sampling_freq
        # contains all point samples
        sampled_time_points = np.arange(xData[0], xData[-1], t_sample)
        # Use interpolation to calculate the sampled values
        interp_func = interpolate.interp1d(
            xData, yData, kind="slinear", fill_value="extrapolate"
        )
        self.sampled_values_points = interp_func(sampled_time_points)
        # self.sampled_values_points2 = interp_func(sampled_time_points2)
        self.lcdNumber.display(self.sampling_freq)
        self.lcdNumber_2.display(max_frequency)
        print("Sampled Time Points:", len(sampled_time_points))
        print("Sampled Values Points:", len(self.sampled_values_points))
        # self.sampled_values_points2 = yData[np.searchsorted(xData, sampled_time_points2)]

        # Plot the sampled signal
        self.canvas3.axes.clear()
        self.canvas4.axes.clear()
        self.canvas5.axes.clear()
        self.canvas3.axes.plot(xData, yData)
        self.canvas3.axes.scatter(
            sampled_time_points, self.sampled_values_points, marker="x", color="r"
        )

        self.canvas3.draw()
        self.widget3.setCentralItem(self.graph)
        self.widget3.setLayout(self.layout3)
        self.whittaker_shannon_interpolation()

    def whittaker_shannon_interpolation(self):
        # Define the sinc function
        # def sinc(x):
        #     if x == 0:
        #         return 1.0
        #     else:
        #         return np.sin(np.pi * x) / (np.pi * x)

        if self.noisy:  # Check if a noisy signal is present
            xData = self.noisy_signal_xdata
        elif self.fname_1:  # Check if a signal is loaded
            xData = self.loaded_signal_xdata
        else:  # Otherwise, use composed signal
            xData = self.composed_signal_xdata

        xData = np.array(xData)

        # Define the interpolated time points
        interpolated_time_points = np.arange(
            xData[0], xData[-1], 1 / self.sampling_freq
        )

        # Initialize an array for the reconstructed signal
        reconstructed_signal = np.zeros(len(xData))

        # Perform the Whittaker-Shannon interpolation
        for i, n in enumerate(interpolated_time_points):
            reconstructed_signal += self.sampled_values_points[i] * np.sinc(
                self.sampling_freq * (xData - interpolated_time_points[i])
            )

        # Plot the reconstructed signal
        self.canvas4.axes.plot(xData, reconstructed_signal)
        self.canvas4.draw()
        self.widget4.setCentralItem(self.graph)
        self.widget4.setLayout(self.layout4)
        self.plot_error()

    def plot_error(self):
        # if self.noisy:  # Check if a noisy signal is present
        #     xData = self.noisy_signal_xdata
        #     yData = self.noisy_signal_ydata
        print("enter error")
        if self.fname_1:  # Check if a signal is loaded
            xData = self.loaded_signal_xdata
            yData = self.values
        else:  # Otherwise, use composed signal
            xData = self.composed_signal_xdata
            yData = self.yData_composed

        # Interpolate the reconstructed signal to match the length of the loaded signal
        interp_func = interpolate.interp1d(
            np.linspace(0, 1, len(self.sampled_values_points)),
            self.sampled_values_points,
            kind="linear",
            fill_value="extrapolate",
        )
        interpolated_sampled_signal = interp_func(np.linspace(0, 1, len(yData)))

        # Calculate the error between the loaded signal and the interpolated sampled signal
        error = np.array(yData) - interpolated_sampled_signal
        # print(interpolated_sampled_signal)

        # Plot the error
        self.canvas5.axes.clear()
        # ymin, ymax = self.canvas3.axes.get_ylim()
        # self.canvas5.axes.set_ylim(ymin, ymax)
        self.canvas5.axes.plot(xData, error, label="Error")
        self.canvas5.axes.set_xlabel("Time")
        self.canvas5.axes.set_ylabel("Error")
        self.widget5.setCentralItem(self.graph)
        self.widget5.setLayout(self.layout5)
        self.canvas5.axes.legend()
        self.canvas5.draw()

    # Noise
    # Unhide noise features
    def add_noise(self):
        if self.plot_in_widget3:  # check if there's a plot in widget 3
            self.snr_slider.show()
            self.snr_lcd.show()
            self.snr_label.show()
            self.snr_slider.setValue(
                self.snr_slider.maximum()
            )  # set value to maximum after showing slider
            snr = self.snr_slider.value()  # get SNR from slider
            self.snr_lcd.display(snr)  # display SNR on LCD
            self.snr_slider.valueChanged.connect(self.apply_noise)
        else:
            # show information message
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Please load a plot first before adding noise.")
            msg.setWindowTitle("Information")
            msg.exec_()

    # Add noise to loaded plot
    def apply_noise(self):
        print("enter noise")
        self.noisy = True
        snr = self.snr_slider.value()  # get SNR from slider
        if snr == 0:
            snr = 0.01  # limit SNR to a small positive number when it reaches zero
        self.snr_lcd.display(snr)  # display SNR on LCD

        # Check if a signal is loaded
        if self.fname_1:
            xData = self.time_col
            yData = self.values
        # If not, use the composed signal
        else:
            xData = self.xData_composed
            yData = self.yData_composed

        # Add noise to the signal
        signal_power = np.mean(np.square(yData))
        noise_power = signal_power / (10 ** (snr / 10))
        noise = np.sqrt(noise_power) * np.random.normal(size=yData.shape)
        noisy_signal = yData + noise

        # Update noisy_signal_xdata and noisy_signal_ydata
        self.noisy_signal_xdata = list(xData)
        self.noisy_signal_ydata = list(noisy_signal)

        # Plot the noisy signal
        self.canvas3.axes.clear()
        self.canvas3.axes.plot(xData, noisy_signal)
        self.canvas3.draw()

        # Sample and reconstruct the noisy signal
        self.sample_signal()

    # return noise feature to its original state
    def reset_noise(self):
        self.noise = False
        self.snr_slider.hide()
        self.snr_lcd.hide()
        self.snr_label.hide()
        self.snr_slider.setValue(
            self.snr_slider.maximum()
        )  # reset slider to maximum value
        snr = self.snr_slider.value()  # get SNR from slider
        self.snr_lcd.display(snr)  # display SNR on LCD
        try:
            self.snr_slider.valueChanged.disconnect(
                self.apply_noise
            )  # disconnect slider from apply_noise function
        except TypeError:
            pass  # ignore error if apply_noise was not connected

    # Return the plot to its original noiseless state
    def reset_plot(self):
        if self.plot_in_widget3:  # check if there's a plot in widget 3
            # Check if the signal was loaded
            if self.fname_1:
                # Clear the axes
                self.canvas3.axes.clear()
                # Plot the original loaded signal
                self.canvas3.axes.plot(self.time_col, self.values)
                # Redraw the canvas
                self.canvas3.draw()
            else:
                # Clear the axes
                self.canvas3.axes.clear()
                # Plot the original composed signal
                self.canvas3.axes.plot(self.xData_composed, self.yData_composed)
                # Redraw the canvas
                self.canvas3.draw()
            self.reset_noise()
        else:
            # show information message
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("There is no plot to reset.")
            msg.setWindowTitle("Information")
            msg.exec_()


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
