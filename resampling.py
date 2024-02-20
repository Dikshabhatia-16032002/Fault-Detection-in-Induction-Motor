import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample
import numpy as np
import csv

def readDataFrame(path):
    dfn=pd.read_csv(path)
    return dfn

def addHeaders(path):
# Define the header
    header = ['rotational', 'uaxial','uradial','utangential','oaxial','oradial','otangential', 'microphone'] # Customize this according to your CSV file

    # Specify the path to your existing CSV file
    csv_file_path = path

    # Read existing data from the CSV file
    existing_data = []
    with open(csv_file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        existing_data = list(reader)

    # Prepend the header to the existing data
    existing_data.insert(0, header)

    # Write the modified data back to the CSV file
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(existing_data)


def fftCalculation(osignal,fs):
    fft_output = np.fft.fft(osignal)
    fft_magnitude = np.abs(fft_output)
    frequencies = np.fft.fftfreq(len(osignal), 1/fs)
    return fft_output,frequencies,fft_magnitude

def downsampling(original_signal,fs,res_fs=125000):
    # Downsample by a factor of 2 using scipy.signal.resample
    signal,frequency,signalMag=fftCalculation(original_signal,fs)
    downsample_frequency = res_fs
    downsampled_signal = resample(original_signal, downsample_frequency)
    dsignal,dfrequency,dsignalMag=fftCalculation(downsampled_signal,downsample_frequency)
    # Plotting
    plt.figure(figsize=(10, 6))
    # Original signal
    plt.subplot(2, 1, 1)
    #plt.plot(frequency,signal)
    plt.hist(original_signal,bins=20, color='blue')
    #plt.hist(signalMag,bins=30, color='blue')
    #plt.stem(frequency, signalMag)
    plt.title('Original Signal')
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    #plt.xlim(0,10000)
    # Downsampled signal
    plt.subplot(2, 1, 2)
    #plt.plot(dfrequency,dsignal)
    plt.hist(downsampled_signal,bins=20, color='blue')
    #plt.hist(dsignalMag,bins=30, color='blue')
    #plt.stem(dfrequency, dsignalMag)
    plt.title('Downsampled Signal')
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    #plt.xlim(0,10000)

    plt.tight_layout()
    plt.show()

def plotting(data1,data2):
    fig, axs = plt.subplots(4, 2, figsize=(10, 12))
    i=0
    for (x,y) in zip(data1.columns,data2.columns):
        row = i // 2
        col = i % 2
        #plt.figure(figsize=(10, 6))
        # Original signal
        #plt.subplot(2, 1, 1)
        #plt.plot(frequency,signal)
        print(x)
        axs[row,col].hist(data1[x], color='green')
        axs[row,col].hist(data2[y], color='red')
        #plt.stem(frequency, signalMag)
        #axs[row,col].set_title(f'{x}')
        axs[row,col].set_xlabel('signal')
        axs[row,col].set_ylabel('counts')
        #plt.xlim(0,250000)
        # Downsampled signal
        i=i+1
        plt.tight_layout()
    plt.show()
        
        
#addHeaders('E:/diksha/horizontal-misalignment/2.0mm/12.288.csv')
x=readDataFrame('E:/diksha/normal/12.288.csv')
y=readDataFrame('E:/diksha/horizontal-misalignment/2.0mm/12.288.csv')
o_signal=x.iloc[:,0] 
sr=250000
res_sr=25000
downsampling(o_signal,sr,res_sr)
#plotting(x,y)








