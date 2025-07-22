import pandas as pd
from scipy.signal import butter, lfilter, filtfilt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def _PCA(sensor_data, sensor_columns):
    scaler = StandardScaler()
    scaled_sensor_data = scaler.fit_transform(sensor_data)

    pca = PCA(n_components=3)
    pca_components = pca.fit_transform(scaled_sensor_data)
    pca_df = pd.DataFrame(pca_components, columns=['pca_1', 'pca_2', 'pca_3'], index=df.index)

    # 5. Combine with non-sensor metadata columns
    meta_columns = [col for col in df.columns if col not in sensor_columns]
    final_pca_df = pd.concat([pca_df, df[meta_columns]], axis=1)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    return final_pca_df

def low_pass_filter(data_table, col, sampling_frequency, cutoff_frequency, order=5, phase_shift=True,):
    # http://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    # Cutoff frequencies are expressed as the fraction of the Nyquist frequency, which is half the sampling frequency
    nyq = 0.5 * sampling_frequency
    cut = cutoff_frequency / nyq

    b, a = butter(order, cut, btype="low", output="ba", analog=False)
    if phase_shift:
        data_table[col + "_lowpass"] = filtfilt(b, a, data_table[col])
    else:
        data_table[col + "_lowpass"] = lfilter(b, a, data_table[col])
    return data_table

def initial_featureengineering(df,sensory_cols):
    # Interpolate missing values
    for col in sensory_cols:
        df[col] = df[col].interpolate()

    df.index = pd.to_datetime(df.index)
    # Average duration of a set
    for s in df["set"].unique():
        df_set = df[df["set"] == s]
        start = df_set.index[0]
        stop = df_set.index[-1]
        duration = (stop - start).total_seconds()
        df.loc[df["set"] == s, "duration"] = duration

    return df

if __name__ == "__main__":
    df = pd.read_pickle("../../data/interim/02_outliers_removed_bychauvenets.pkl")
    predictor_columns = list(df.columns[:6])
    clean_df = initial_featureengineering(df,predictor_columns)

    df_lowpass = clean_df.copy()
    fs = 1000 / 200
    cutoff = 1.2

    for col in predictor_columns:
        df_lowpass = low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
        df_lowpass[col] = df_lowpass[col + "_lowpass"]
        del df_lowpass[col + "_lowpass"]

    df_pca = df_lowpass.copy()
    sensor_data = df_pca[predictor_columns]  # <-- use filtered data here

    component_analysis = _PCA(sensor_data, predictor_columns)
    # Save the DataFrame with outliers removed
    component_analysis.to_csv("D:/PycharmProjects/Strength_Training_Tracker_with Fastapi/data/processed/final_feature_df.csv")
    component_analysis.to_pickle("../../data/interim/03_final_feature_df.pkl")
    print("Outlier removal complete. Saved to 03_final_feature_df.pkl.")