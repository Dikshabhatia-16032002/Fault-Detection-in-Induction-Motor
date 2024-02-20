import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import seaborn as sns
import os

#adding headers to the csv file
def addHeaders(csv_file_path):
    # Defining the header
    header = ['rotationalFrequency','underAxial','underRadial','underTangential','overAxial','overRadial','overTangential','microphone']  # Customize this according to your CSV file

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

    #print("Header added successfully.")

#reading the csv file in dataframe
def readFile(file_path):
    df=pd.read_csv(file_path)
    return df

def generateFFT(data1,data2,fname):
    # Sample signal and sampling rate
    sr = 50000  # Sampling rate (Hz)
    #figsize=(10, 12)
    fig, axs = plt.subplots(4, 2,figsize=(10, 16))
    plt.subplots_adjust(hspace=0.8,wspace=0.8)
    i=0
    for (datac1,datac2) in zip(data1.columns,data2.columns): 
        x1=data1[datac1]
        # Perform FFT
        X1 = np.fft.fft(x1)
        N1 = len(X1)  # Length of FFT
        # Frequency array
        freq1 = np.fft.fftfreq(N1, d=1/sr)
        x2=data2[datac2]
        # Perform FFT
        X2 = np.fft.fft(x2)
        N2 = len(X2)  # Length of FFT
        # Frequency array
        freq2 = np.fft.fftfreq(N2, d=1/sr)
        legends=['fault','normal']
        row = i // 2
        col = i % 2
        # plotting fourier series-displaying their frequency spectrum
        axs[row,col].stem(freq1, np.abs(X1), 'r', markerfmt=" ", basefmt="-b")
        axs[row,col].stem(freq2, np.abs(X2), 'b', markerfmt=" ", basefmt="-b")
        axs[row,col].set_title(f'{datac1}')
        axs[row,col].set_xlabel('Frequency (Hz)')
        axs[row,col].set_ylabel('FFT Magnitude')
        axs[row,col].set_xlim(0,25001)
        axs[row,col].legend(legends)
        i=i+1
    filepath = "D:/bnftech/dataset/horizontal-misalignment/result"
    fig.suptitle(f'{fname} 2.0mm horizontal misalignment vs normal frequency spectrum')
    plt.savefig(f'{filepath}/{fname} 2.0mm horizontal misalignment vs normal frequency spectrum .png')
    
    #plt.show()
    
def plotDataHist(data1,data2,fname):
    fig, axs = plt.subplots(4, 2,figsize=(10, 16))
    plt.subplots_adjust(hspace=0.8, wspace=0.8)
    i=0
    for (datac1,datac2) in zip(data1.columns,data2.columns): 
        x1=data1[datac1]
        x2=data2[datac2]
        legends=['fault','normal']
        row = i // 2
        col = i % 2
        #axs[row,col].hist(np.abs(x1),density=True,color='red')
        #axs[row,col].hist(np.abs(x2),density=True,color='green')
        sns.histplot(x1, kde=True, color='red', ax=axs[row,col],bins=50)
        sns.histplot(x2, kde=True, color='green', ax=axs[row,col],bins=50)
        #sns.kdeplot(data, color='green', ax=axs[row,col])
        axs[row,col].set_title(f' {datac1}')
        axs[row,col].legend(legends)
        i=i+1
    fig.suptitle(f'{fname} 2.0mm horizontal misalignment vs normal histogram')
    plt.savefig(f'{fname} 2.0mm horizontal misalignment vs normal histogram.png')

def plotDataHistFFT(data1,data2,fname):
    fig, axs = plt.subplots(4, 2,figsize=(10, 16))
    plt.subplots_adjust(hspace=0.8, wspace=0.8)
    i=0
    for (datac1,datac2) in zip(data1.columns,data2.columns): 
        x1=data1[datac1]
        # Perform FFT
        X1 = np.fft.fft(x1)
        N1 = len(X1)  # Length of FFT
        # Frequency array
        #freq1 = np.fft.fftfreq(N1, d=1/sr)
        x2=data2[datac2]
        # Perform FFT
        X2 = np.fft.fft(x2)
        N2 = len(X2)  # Length of FFT
        # Frequency array
        #freq2 = np.fft.fftfreq(N2, d=1/sr)

        legends=['fault','normal']
        row = i // 2
        col = i % 2
        #axs[row,col].hist(np.abs(x1),density=True,color='red')
        #axs[row,col].hist(np.abs(x2),density=True,color='green')
        sns.histplot(np.abs(X1), kde=True, color='red', ax=axs[row,col],bins=50)
        sns.histplot(np.abs(X2), kde=True, color='green', ax=axs[row,col],bins=50)
        #sns.kdeplot(data, color='green', ax=axs[row,col])
        axs[row,col].set_title(f' {datac1}')
        axs[row,col].legend(legends)
        i=i+1
    fig.suptitle(f'{fname} 2.0mm horizontal misalignment vs normal histogramFFT')
    plt.savefig(f'{fname} 2.0mm horizontal misalignment vs normal histogramFFT.png')

def statsNormal(n):
    dfn=pd.DataFrame()
    for i in n.columns:
        #j=0
        normal=[]
        meann=np.mean(n[i])
        normal.append(meann)
        mediann=np.median(n[i])
        normal.append(mediann)
        stdn=n[i].std()
        normal.append(stdn)
        maxn=np.max(n[i])
        normal.append(maxn)
        minn=np.min(n[i])
        normal.append(minn)
        #i=i+1
        dfn[i]=normal
    return dfn
    print(dfn)
#def statsF


def statisticalResults(d,ns):
    fault=[]
    #Variance=[]
    labels=['mean','median','std','min','max']
    for j in d.columns:
        meanf=np.mean(d[j])
        fault.append(meanf)
        medianf=np.median(d[j])
        fault.append(medianf)
        stdf=d[j].std()
        fault.append(stdf)
        maxf=np.max(d[j])
        fault.append(maxf)
        minf=np.min(d[j])
        fault.append(minf)
        df=pd.DataFrame({'      ':labels,'normal':ns[j],'fault':fault})
        df.set_index('      ',inplace=True)
        print('')
        print(df)
        fault=[]
    #print(df) 
#addHeaders('D:/bnftech/dataset/horizontal-misalignment/horizontal-misalignment/2.0mm/12.288.csv')
def traverse_files(folder_path1,folder_path2):
    for ([root1, dirs1, files1],[root2, dirs2, files2]) in zip(os.walk(folder_path1),os.walk(folder_path2)):
        for (file_name1,file_name2) in zip(files1,files2):
            # Construct the full path to the file
            file_path1 = os.path.join(root1, file_name1)
            file_path2 = os.path.join(root2, file_name2)
            fname, _ = os.path.splitext(file_name1)
            # Print or process the file_path as needed
            print(fname)
            #addHeaders(file_path1)
            #addHeaders(file_path2)
            d=readFile(file_path1)
            n=readFile(file_path2)
            #n=readFile('D:/bnftech/dataset/normal/normal/12.288.csv')
            #print(d.describe())
            #print(n.describe())
            #ns=statsNormal(n)
            #statisticalResults(d,ns)

            generateFFT(d,n,fname)
            #plotDataHist(d,n,fname)
            #plotDataHistFFT(d,n,fname)

# Example usage:
folder_path1 = "D:/bnftech/dataset/horizontal-misalignment/horizontal-misalignment/2.0mm"
folder_path2 = "D:/bnftech/dataset/normal/normal"
traverse_files(folder_path1,folder_path2)

#plt.show()






