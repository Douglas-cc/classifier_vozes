import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import IPython.display

from librosa import load
from librosa.feature import (
    fourier_tempogram, 
    tempogram, 
    chroma_stft,
    chroma_cens,
    chroma_cqt,
    melspectrogram,
    mfcc,
    tonnetz,
    rms,
    zero_crossing_rate,
    spectral_rolloff
)

from collections import defaultdict
from scipy import stats
from pydub import AudioSegment
from IPython.display import Audio   

def play_audio(directory):
    audio, sr = load(directory, mono=True) 
    
    if np.ndim(audio) == 1:
        channels = [audio.tolist()]
    else:
        channels = audio.tolist()

    return IPython.display.HTML("""
        <script>
            if (!window.audioContext) {
                window.audioContext = new AudioContext();
                window.playAudio = function(audioChannels, sr) {
                    const buffer = audioContext.createBuffer(audioChannels.length, 
                    audioChannels[0].length, sr);
                    
                    for (let [channel, data] of audioChannels.entries()) {
                        buffer.copyToChannel(Float32Array.from(data), channel);
                    }
            
                    const source = audioContext.createBufferSource();
                    source.buffer = buffer;
                    source.connect(audioContext.destination);
                    source.start();
                }
            }
        </script>
        <button onclick="playAudio(%s, %s)">Play</button>
    """ % (json.dumps(channels), sr))
    
    
def create_dirs(folders, directory):
    for folder in folders:
        print(f'Creating folder: { folder } in { directory }')
        os.makedirs(f'{ directory }{ folder }', exist_ok=True)
        
                
def read_fold(folder):
    return os.listdir(folder)


def save_data(directory, name_out, variable):
    f = open(directory +  name_out, 'wb')
    pickle.dump(variable, f)
    f.close() 
    
    
def load_data(directory):
    f = open(directory, 'rb')  
    aux = pickle.load(f)
    f.close()
    return aux     


def remove_caracter(column, caracter):
    size = column.shape[0]
    return [column[i].replace(caracter, '') for i in range(size)]


def get_duration(data, sr):
    return len(data)/sr


def table_missing(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False) * 100
    missing = pd.concat([total, percent], axis=1, join='outer', keys=['total_faltantes', 'percentual'])
    missing.index.name = 'Variaveis Numericas'
    return missing


def convert_sonds(dir_input, dir_output, format_input, format_output):
    n = 0
    for music in os.listdir(dir_input):
        n += 1
        total = len(os.listdir(dir_input))
        
        print(f'converting file: {n} - { music } of total {total} ')
        output = AudioSegment.from_mp3(f'{dir_input}/{music}')
    
        name = music.replace(format_input, format_output)
        output.export(f'{dir_output}/{name}')
        
        
def statistics_values(data, types):
    if types == 'mean':
        return np.mean(data)
    
    if types == 'median':
        return np.median(data)    
    
    if types == 'min':
        return np.min(data)
    
    if types == 'max':
        return np.max(data)
    
    if types == 'std':
        return np.std(data)
    
    if types == 'mode':
        return stats.mode(data)[0][0]
          
def extract_features(dir_audio, statistic):
    files = read_fold(dir_audio)
    dic = defaultdict(list)
    
    for file in files:
        print(f'Arquivo: {file} - de {len(files)} - {statistic}')
        sample, sr = load(dir_audio + file, sr=None, mono=True)
        
        dic["arquivo"].append(file)
        
        # # Fourier Tempogram
        # four_tempog = fourier_tempogram(y=sample, sr=sr)    
             
        # for i in range(four_tempog.shape[0]):   
        #     name = f'fourier_tempogram-{statistic}-{str(i)}'
        #     value = statistics_values(four_tempog[i,:], statistic)
            
        #     dic[name].append(value)
            
        # Tempogram
        tempog = tempogram(y=sample, sr=sr)
        
        for i in range(tempog.shape[0]):  
            name = f'tempogram-{statistic}-{str(i)}'
            value = statistics_values(tempog[i,:], statistic) 
            
            dic[name].append(value)
            
        # chroma stft
        chrom_stft = chroma_stft(y = sample, sr = sr, n_fft=2048, hop_length=512)
            
        for i in range(chrom_stft.shape[0]):
            name = f'chroma_stft-{statistic}-{str(i)}'
            value = statistics_values(chrom_stft[i,:], statistic) 
            
            dic[name].append(value)
        
        # chroma cens
        chrom_cens = chroma_cens(y = sample, sr = sr, hop_length=512)
            
        for i in range( chrom_cens.shape[0] ):
            name = f'chroma_cens-{statistic}-{str(i)}'
            value = statistics_values(chrom_cens[i,:], statistic) 
            
            dic[name].append(value)
            
        # chroma cqt  
        chrom_cqt = chroma_cqt(y = sample, sr = sr, hop_length=512)
            
        for i in range(chrom_cqt.shape[0]):
            name =  f'chroma_cqt-{statistic}-{str(i)}'
            value = statistics_values(chrom_cqt[i,:], statistic) 
            
            dic[name].append(value)
                
        # mel spectgram 128 dados
        mel = melspectrogram(y = sample, sr=sr, n_mels=128)
        
        for i in range(mel.shape[0]):
            name = f'mel-{statistic}-{str(i)}' 
            value = statistics_values(mel[i,:], statistic) 
            
            dic[name].append(value)
                
        # mfcc, mel frequencia 30 contéudos                     
        value_mfcc = mfcc(y=sample, sr=sr, n_mfcc=50)
            
        for i in range(value_mfcc.shape[0]):
            name = f'mfcc-{statistic}-{str(i)}'
            value = statistics_values(value_mfcc[i,:], statistic)

            dic[name].append(value)
            
        # tonez 
        value_tonnetz = tonnetz(y=sample, sr=sr)
        
        for i in range(value_tonnetz.shape[0]):
            name = f'tonnetz-{statistic}-{str(i)}'
            value = statistics_values(value_tonnetz[i,:], statistic) 
           
            dic[name].append(value)
            
        # rms 
        value_rms = rms(y=sample)
            
        for i in range(value_rms.shape[0]):
            name = f'rms-{statistic}-{str(i)}'
            value = statistics_values(value_rms[i,:], statistic) 

            dic[name].append(value)  
                
        # zcr - 1 conteudo
        zcr = zero_crossing_rate(y = sample)
        
        for i in range(zcr.shape[0]):    
            name = f'zcr-{statistic}-{str(i)}'
            value = statistics_values(zcr[i,:], statistic) 
            
            dic[name].append(value)  
                
        #  roll - unico valor
        roll = spectral_rolloff(y=sample, sr=sr, n_fft=2048) 
        
        for i in range(roll.shape[0]):
            name = f'rolloff-{statistic}-{str(i)}'
            value = statistics_values(roll[i,:], statistic) 

            dic[name].append(value)          
    
    df = pd.DataFrame(dic) 
     
    # save_data('data/', f'features_sonds_{statistic}_mel50', df)
    df.to_csv(f'data/features_sonds_{statistic}_mel50.csv')
    
    return df