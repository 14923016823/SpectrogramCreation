import numpy as np 

def stft_band(signal, frame_size, overlap_size, window_function=np.hanning, f_sampeling=250_000, f_relevant = 15_000): 
    """ 
    Computes the Short-Time Fourier Transform (STFT) of a signal. 
    
    Parameters: 
    signal (np.ndarray): The input signal to be transformed.
    frame_size (int): The size of each frame for the STFT.
    overlap_size (int): The number of samples to overlap between frames.
    window_function: A function that generates a window of the specified size (default is np.hanning).
    f_sampeling (int): The sampling frequency of the input signal (default is 250,000 Hz).
    f_relevant (int): The relevant frequency range for the STFT (default is 15,000 Hz).
    Returns: np.ndarray: A 2D array containing the STFT of the input signal.
    """
    try: 
        step_size = frame_size - overlap_size 
        num_frames = (len(signal) - frame_size) // step_size + 1
        df = f_sampeling / frame_size
        k = int(np.floor(f_relevant / df))
        center = frame_size // 2
        lo = center - k
        hi = center + k + 1

        #stft_matrix = np.empty((num_frames, hi - lo), dtype=np.complex64)
        stft_matrix = np.empty((num_frames, frame_size), dtype=np.complex64)
        
        try:
            win = window_function(frame_size)
        except:
            print("The windowing function provided was incorrect. Defaulting to Hanning window.")
            win = np.hanning(frame_size)
        
        ### STFT frame-wise processing 
        for i in range(num_frames): 
            start_index = i * step_size
            end_index = start_index + frame_size
    
            signal_frame = signal[start_index:end_index]
            
            # Apply windowing function to the current frame 
            windowed_frame = signal_frame * win
            
            # DC removal
            #windowed_frame = windowed_frame - np.mean(windowed_frame)
            
            try: 
                stft_frame = np.fft.fftshift(np.fft.fft(windowed_frame))
                ### Parse the relevant spectral components from the STFT frame 
                #stft_frame = stft_frame[lo:hi] 
                stft_matrix[i, :] = stft_frame 
            except: 
                print("FFT computation failed for the current frame. Skipping this frame. '\n' The index of failure is: ", i) 
                continue 
          
        #f = (np.arange(lo, hi) - center) * df
        f = (np.arange(frame_size) - center) * df
        t = (np.arange(num_frames)*step_size + frame_size/2) / f_sampeling
        return stft_matrix, f, t
    
    except: 
        print("An error occurred during the STFT computation. Please check the input parameters and try again.") 
        return None