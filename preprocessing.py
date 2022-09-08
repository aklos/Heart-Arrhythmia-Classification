import sys
import numpy as np
import pandas as pd
import wfdb as wf
import mne
import heartpy as hp
from scipy import signal
from biosppy.signals import ecg


def manually_segment(edf_file_path):
    edf = mne.io.read_raw_edf(edf_file_path)
    # header = ','.join(edf.ch_names)
    data = [x[0] for x in edf.get_data().T]
    data = hp.remove_baseline_wander(data, 500, cutoff=0.01)

    # Resample to 125 Hz
    new_size = int(len(data) * 125 / 500)
    data = signal.resample(data, new_size)

    # Find heartbeats
    working_data, measures = hp.process(data, 125)
    samples = working_data['hr']
    peaks = working_data['peaklist']

    # Split heartbeats into 188 samples
    segmented_data = np.array([])
    for i, peak in enumerate(peaks):
        if i == 0:
            continue
        if i == len(peaks) - 1:
            continue

        # Lean towards right side
        # segment = np.zeros(188)
        start_index = int(peak - ((188 / 2) - 1))
        end_index = int(peak + ((188 / 2) + 1))

        # for i, x in enumerate(range(start_index, end_index)):
        #     segment[i] = samples[x]
        segment = np.array(samples[start_index:end_index])
        segmented_data = np.append(segmented_data, segment)

    print(segmented_data.size)


def wfdb_segment(edf_file_path):
    record = wf.edf2mit(edf_file_path, verbose=True)

    signal = record.p_signal
    data = signal.transpose()

    return data[0][:500 * 60 * 30], record.fs


def edftocsv(edf_file_path, csv_name):
    """
        Input:
            edf_file_path - Path to the EDF file to convert. (e.g. "./100.edf")
            csv_name - Name of the CSV file which will be created
        Output:
            CSV file containing the processes ECG data in 188 columns
    """
    data, sample_rate = wfdb_segment(edf_file_path)

    out = ecg.ecg(signal=data, sampling_rate=sample_rate,
                  show=False)
    rpeaks = np.zeros_like(data, dtype='float')
    rpeaks[out['rpeaks']] = 1.0

    beatstoremove = np.array([0])
    beats = np.split(data, out['rpeaks'])

    # (index,val) pairs of rpeak locations
    for idx, idxval in enumerate(out['rpeaks']):
        firstround = idx == 0
        lastround = idx == len(beats) - 1

        # Skip first and last beat.
        if (firstround or lastround):
            continue

        # Get the classification value that is on
        # or near the position of the rpeak index.
        fromidx = 0 if idxval < 10 else idxval - 10
        toidx = idxval + 10

        # Append some extra readings from next beat.
        beats[idx] = np.append(beats[idx], beats[idx+1][:40])

        # Normalize the readings to a 0-1 range for ML purposes.
        beats[idx] = (beats[idx] - beats[idx].min()) / beats[idx].ptp()

        # Resample to 125Hz
        newsize = int((beats[idx].size * 125 / sample_rate) + 0.5)
        beats[idx] = signal.resample(beats[idx], newsize)

        # Skipping records that are too long.
        if (beats[idx].size > 187):
            beatstoremove = np.append(beatstoremove, idx)
            continue

        # Pad with zeroes.
        zerocount = 187 - beats[idx].size
        beats[idx] = np.pad(beats[idx], (0, zerocount),
                            'constant', constant_values=(0.0, 0.0))

        # Append the classification to the beat data.
        beats[idx] = np.append(beats[idx], 0)

    beatstoremove = np.append(beatstoremove, len(
        beats)-1)  # Removing unnecessary beats
    # Remove first and last beats and the ones without classification.
    beats = np.delete(beats, beatstoremove)
    savedata = np.array(list(beats[:]), dtype=np.float)
    outfn = './files/' + csv_name + '.csv'
    with open(outfn, "wb") as fin:
        np.savetxt(fin, savedata, delimiter=",", fmt='%f')

    return pd.read_csv('./files/' + csv_name + '.csv', header=None)


if __name__ == '__main__':
    edf_file_path = sys.argv[1]
    edftocsv(edf_file_path, 'test')
