from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import Resample
from scipy.stats import gaussian_kde
import seaborn as sns

NORMAL_FILEPATH = 'D:/bnftech/dataset/normal/13.1072.csv'
#FAULT_FILEPATHS = ["D:/bnftech/dataset/horizontal-misalignment/2.0mm/13.5168.csv","D:/bnftech/dataset/vertical-misalignment/1.90mm/12.0832.csv","D:/bnftech/dataset/imbalance/35g/12.0832.csv","D:/bnftech/dataset/underhang/ball_fault/35g/13.7216.csv","D:/bnftech/dataset/overhang/ball_fault/35g/12.288.csv"]
FAULT_FILEPATHS = ["D:/bnftech/dataset/overhang/ball_fault/35g/17.6128.csv"]
#titles=["horizontal Misalignment 2.0 mm","vertical Misalignment","imbalance 35g","underhang ball 35g","overhang ball 35g"]
titles=["overhang ball 35g 17.6128 file"]
DATA_COLUMNS = ["Tacho","Acc_U_Axial","Acc_U_Radial","Acc_U_Tangential","Acc_O_Axial","Acc_O_Radial","Acc_O_Tangential","MIC"]
N_BINS = 40

SAMPLING_RATE = 50000
RESAMPLING_RATE = 25000

def ReadData(filename=None,headers=None):
    df=pd.DataFrame()
    if not filename:
        filename = NORMAL_FILEPATH
    df = pd.read_csv(filename,header=None)
    if not headers:
        return df
    else:
        df.columns = headers
        return df

def fftCalculation(osignal,fs=50000):
    fft_output = np.fft.fft(osignal)
    fft_magnitude = np.abs(fft_output)
    frequencies = np.fft.fftfreq(len(osignal), 1/fs)
    return fft_output,frequencies,fft_magnitude

class MyPipeline(torch.nn.Module):
    def __init__(
        self,
        input_freq=16000,
        resample_freq=8000,
        n_fft=1024,
        n_mel=256,
        stretch_factor=0.8,
    ):
        super().__init__()
        self.resample = Resample(orig_freq=input_freq, new_freq=resample_freq)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Resample the input
        resampled = self.resample(waveform)
        return resampled
    
def ResamplingData(signal_data,sr=50000,se_res=25000):
    resampling_transform = Resample(sr,se_res,dtype=torch.float64)
    return resampling_transform(signal_data)

def csv_files(NORMAL_FILEPATH,FAULT_FILEPATHS,DATA_COLUMNS):
    df_normal_csv = ReadData(NORMAL_FILEPATH,DATA_COLUMNS)
    print(df_normal_csv.describe())
    df_fault_csv=[]
    for f in FAULT_FILEPATHS:
        csv = ReadData(f,DATA_COLUMNS)
        df_fault_csv.append(csv)
        print(f)
        print(csv.describe())
    return df_normal_csv,df_fault_csv

def downsampling(df_normal_csv,df_fault_csv,titles,DATA_COLUMNS,N_BINS):
    z=0
    for x in df_fault_csv:
        i=0
        for i in range(0,8):
            data = torch.from_numpy(np.array(df_normal_csv.loc[:,DATA_COLUMNS[i]]))
            fault_data = torch.from_numpy(np.array(x.loc[:,DATA_COLUMNS[i]]))
            #region Resampling the series
            figure,ax = plt.subplots(nrows=2,ncols=2,figsize=(10, 10))
            plt.subplots_adjust(right=0.9)

            # the histogram of the Normal data
            #n0, bins0, patches0 = ax[0,0].hist(data, N_BINS)
            sns.histplot(data, kde=True, color='green', alpha=0.7,ax=ax[0,0],bins=N_BINS)
            ax[0,0].set_title(f"• Normal • Original Signal : Sampling Rate: {SAMPLING_RATE}")
            # the histogram of the Downsampled data
            downsampled_data = ResamplingData(data,SAMPLING_RATE,RESAMPLING_RATE)
            #n1, bins1, patches1 = ax[1,0].hist(downsampled_data, N_BINS)
            sns.histplot(downsampled_data, kde=True, color='green', alpha=0.7,ax=ax[1,0],bins=N_BINS)
            ax[1,0].set_title(f"• Normal • Downsampled Signal : Sampling Rate: {RESAMPLING_RATE}")
        
            # the histogram of the fault data
            #n2, bins2, patches2 = ax[0,1].hist(fault_data, N_BINS)
            ax[0,1].set_title(f"• Fault • Original Signal : Sampling Rate: {SAMPLING_RATE}")
            sns.histplot(fault_data, kde=True, color='green', alpha=0.7,ax=ax[0,1],bins=N_BINS)
            # the histogram of the Downsampled data
            downsampled_fault_data = ResamplingData(fault_data,SAMPLING_RATE,RESAMPLING_RATE)
            #n3, bins3, patches3 = ax[1,1].hist(downsampled_fault_data, N_BINS)
            ax[1,1].set_title(f"• Fault • Downsampled Signal : Sampling Rate: {RESAMPLING_RATE}")
            sns.histplot(downsampled_fault_data, kde=True, color='green', alpha=0.7,ax=ax[1,1],bins=N_BINS)

            plt.suptitle(f"• {titles[z]} {DATA_COLUMNS[i]}")
            figure.tight_layout()
            savepath="D:/bnftech/Fault detection in induction motor/resampling"
            plt.savefig(f'{savepath}/• {titles[z]} for {DATA_COLUMNS[i]}.png')
            #plt.show()
            i=i+1
        z=z+1
    

def histogramPlt(df_normal_csv,df_fault_csv,titles,DATA_COLUMNS,N_BINS):
    z=0
    for x in df_fault_csv:
        figure, ax = plt.subplots(4, 2, figsize=(10, 16))
        j=0
        for i in range(0,8):
            row = j // 2
            col = j % 2
            # the histogram of the Normal data
            #n0, bins0, patches0 = ax[0,0].hist(tacho_data, N_BINS)
            sns.histplot(df_normal_csv.loc[:,DATA_COLUMNS[i]], kde=True, color='green', alpha=0.7,ax=ax[row,col],bins=N_BINS)
            ax[row,col].set_title(f" {DATA_COLUMNS[i]}")
            sns.histplot(x.loc[:,DATA_COLUMNS[i]], kde=True, color='red', alpha=0.7,ax=ax[row,col],bins=N_BINS)
            # Tweak spacing to prevent clipping of ylabel
            figure.tight_layout()
            j=j+1
        plt.legend()
        figure.suptitle(f"• normal vs {titles[z]} ")
        plt.savefig(f"D:\bnftech\Fault detection in induction motor\histogram\• normal vs {titles[z]}.png")
        plt.show()
        z=z+1
        
def fftGraph(df_normal_csv,df_fault_csv,titles,DATA_COLUMNS):
    df_csv=df_fault_csv
    df_csv.append(df_normal_csv)
    titles.append("normal")
    z=0
    for x in df_csv:#df_normal_csv- for normal file
        figure, ax = plt.subplots(nrows=4,ncols= 2, figsize=(10, 16))
        plt.subplots_adjust(hspace=0.8, wspace=0.8)
        j=0
        for i in range(0,8):
            row=j//2
            col=j%2
            
            fft,freq,fftMag=fftCalculation(x.loc[:,DATA_COLUMNS[i]])
            ax[row,col].stem(freq, fftMag)
            ax[row,col].set_title(f"{DATA_COLUMNS[i]}")
            ax[row,col].set_xlim(0,25000)
            ax[row,col].set_xlabel("frequency")
            ax[row,col].set_ylabel("FFT Magnitude")
            '''
            fftn,freqn,fftMagn=fftCalculation(df_normal_csv.loc[:,DATA_COLUMNS[i]])
            ax[row,col].stem(freqn, fftMagn)
            ax[row,col].set_title(f"{DATA_COLUMNS[i]}")
            ax[row,col].set_xlim(0,25000)
            ax[row,col].set_xlabel("frequency")
            ax[row,col].set_ylabel("FFT Magnitude")
            '''
            j=j+1 
        figure.tight_layout()
        figure.suptitle(f"• fft of {titles[z]}")
        savepath="D:/bnftech/Fault detection in induction motor/fft"
        plt.savefig(f'{savepath}/• fft of {titles[z]}.png')
        #plt.show()
        z=z+1

df_normal_csv,df_fault_csv=csv_files(NORMAL_FILEPATH,FAULT_FILEPATHS,DATA_COLUMNS)
#histogramPlt(df_normal_csv,df_fault_csv,titles,DATA_COLUMNS,N_BINS)
downsampling(df_normal_csv,df_fault_csv,titles,DATA_COLUMNS,N_BINS)
fftGraph(df_normal_csv,df_fault_csv,titles,DATA_COLUMNS)
