import argparse
import numpy as np
import os
import pandas as pd
import pywt




def data_to_wt(data, type, level):
    num_samples = data.shape[0]
    # print(num_samples)
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    coeffs_train = pywt.wavedec(data[:num_train], type, level=level)
    coeffs_val = pywt.wavedec(data[num_train: num_train + num_val], type, level=level)
    coeffs_test = pywt.wavedec(data[-num_test:], type, level=level)
    dealt = 1
    threshold_train = dealt * np.sqrt(2 * np.log(len(data[:num_train])))
    threshold_val = dealt * np.sqrt(2 * np.log(len(data[num_train: num_train + num_val])))
    threshold_test = dealt * np.sqrt(2 * np.log(len(data[-num_test:])))
    for i in range(level):
        coeffs_train[-(i + 1)] = pywt.threshold(coeffs_train[-(i + 1)], value=threshold_train, mode="soft", substitute=0)
        coeffs_val[-(i + 1)] = pywt.threshold(coeffs_val[-(i + 1)], value=threshold_val, mode="soft", substitute=0)
        coeffs_test[-(i + 1)] = pywt.threshold(coeffs_test[-(i + 1)], value=threshold_test, mode="soft", substitute=0)
    rec_train = list(pywt.waverec(coeffs=coeffs_train, wavelet=type))
    rec_val = list(pywt.waverec(coeffs=coeffs_val, wavelet=type))
    rec_test = list(pywt.waverec(coeffs=coeffs_test, wavelet=type))
    if len(rec_train)> num_train:
        rec_train.pop()
    if len(rec_val)> num_val:
        rec_val.pop()
    if len(rec_test)> num_test:
        rec_test.pop()
    # print(len(rec_train))
    # print(len(rec_val))
    # print(len(rec_test))
    # print(type(rec_train))
    rec = rec_train+rec_val+rec_test
    # print(len(rec))
    return rec


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 1))
        day_in_week[np.arange(num_samples), :, 0] = np.asarray(df.index.dayofweek/7).reshape(num_samples,1)
        data_list.append(day_in_week)
    data_towt = df.values
    wt_type_list = ['db1', 'db2', 'db3', 'db4']
    level = 5
    for type in wt_type_list:
        data_wt = []
        for i in range(data.shape[1]):
            data_wt.append(data_to_wt(data_towt[:, i], type, level))
        data_arr = np.asarray(data_wt)
        data_arr = data_arr.transpose((1, 0))
        data_arr = np.expand_dims(data_arr, axis=-1)
        data_list.append(data_arr)
    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    df = pd.read_hdf(args.traffic_df_filename)
    zero_mask = (df > 0).astype(np.float32)
    df = df.replace(0, np.nan)
    df = df.fillna(method='ffill')
    df = df.fillna(0.0)
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(1-args.history_length, 1, 1),)) # -11, -5, -2
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 1+args.horizon, 1)) # 4, 7, 13
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=True,   
    )

    x_mask, y_mask = generate_graph_seq2seq_io_data(
        zero_mask,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=True,  
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train] * y_mask[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val] * y_mask[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:] * y_mask[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)     
        np.savez_compressed(
            os.path.join(args.output_dir, "%s-history-%d-horizon-%d-day_in_week_wavelet_db_fam_level5_soft.npz" % (cat, args.history_length, args.horizon)),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/metr-la.h5",
        help="Raw traffic readings.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="The length of horison.",
    )
    parser.add_argument(
        "--history_length",
        type=int,
        default=3,
        help="The length of history.",
    )
    args = parser.parse_args()
    main(args)
