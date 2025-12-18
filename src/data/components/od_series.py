import torch
import numpy as np
import random
import tqdm
from torch.utils.data import Dataset
import pandas as pd
import random


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class busDataset(Dataset):
    def __init__(self, csv_path, input_window, output_window=1, train=True):
        self.csv_path = csv_path
        self.input_window = input_window
        self.output_window = output_window
        self.stride = self.input_window // 2
        self.train = train

        self.temperature_info = [20, 10]
        self.flow_info = [50, 50]

        self.keys = [
            "up",
            "temperature",
            "weather",
            "holiday",
            "hour",
            "weekday",
            "month",
        ]
        self.weather_num = 9 + 1

        data, nums = self.get_data(csv_path)
        self.data = data
        self.nums = np.cumsum(nums)
        self.weights = softmax(nums)

    def __len__(self):
        return self.nums[-1]

    def __getitem__(self, idx):
        if self.train:
            i = random.choices(range(len(self.data)), weights=self.weights)[0]
            sub = self.data[i]
            j = random.choice(range(len(sub) - self.input_window))
            data = []
            date = []
            lbls = []
            for k in range(self.input_window):
                data.append(sub[j + k]["data"])
                date.append(sub[j + k]["day"])
            for k in range(self.output_window):
                lbls.append(sub[j + k + self.input_window]["data"])
                date.append(sub[j + k]["day"])
            data = np.stack(data, axis=1)
            lbls = np.stack(lbls, axis=1)
            return {"data": data, "date": date, "lbls": lbls}
        else:
            for i, num in enumerate(self.nums):
                if num >= idx:
                    break

            sub = self.data[i]
            j = (idx - self.nums[i]) * self.stride
            data = []
            date = []
            lbls = []
            for k in range(self.input_window):
                data.append(sub[j + k]["data"])
                date.append(sub[j + k]["day"])
            for k in range(self.output_window):
                lbls.append(sub[j + k + self.input_window]["data"])
                date.append(sub[j + k]["day"])
            data = np.stack(data, axis=1)
            lbls = np.stack(lbls, axis=1)
            return {"data": data, "date": date, "lbls": lbls}

    @staticmethod
    def collect_fn(batch):
        data = []
        lbls = []
        dates = []
        for item in batch:
            data.append(item["data"])
            lbls.append(item["lbls"])
            dates.append(item["date"])
        return {
            "data": torch.tensor(data, dtype=torch.float32),
            "date": dates,
            "lbls": torch.tensor(lbls, dtype=torch.float32),
        }

    def get_data(self, csv_path):
        data = pd.read_csv(csv_path)
        mean, std = self.temperature_info
        data["temperature"] = data["temperature"].apply(lambda x: (x - mean) / std)
        data["holiday"] = data["holiday"].astype(int)
        # mean, std = self.flow_info
        # data["up"] = data["up"].apply(lambda x: (x - mean) / std)
        data["month"] = pd.to_datetime(data["date"]).apply(lambda x: x.month)
        data["date"] = pd.to_datetime(data["date"])

        data["weather"] = data["weather"].fillna(self.weather_num)
        train_sets = []
        for name, group in data.groupby(["lineId", "hour"]):
            if len(group) < 30:
                continue
            res = []
            for item in group.to_dict(orient="records"):
                item_array = []
                for key in self.keys:

                    if key == "weather":
                        value = np.zeros(self.weather_num-1)
                        if int(item[key])>1:
                            value[int(item[key]) - 2] = 1
                    else:
                        value = np.array(item[key]).reshape(-1)
                    item_array.append(value)
                item_array = np.concatenate(item_array)
                res.append({"day": item["date"], "data": item_array})
            res = sorted(res, key=lambda x: x["day"])
            train_sets.append(res)

        nums = []
        for value in train_sets:
            num = (
                len(value) - (self.input_window + self.output_window)
            ) // self.stride + 1
            nums.append(num)

        return train_sets, nums


if __name__ == "__main__":
    data = busDataset(r"D:\project\busODPredict\data\processed\train.csv", input_window=14, train=False)
    print(len(data))
    import matplotlib.pyplot as plt
    for i in tqdm.tqdm(data):
        data = i["data"][0,:]
        print(i["date"])
        print(i["data"][-2,:])
        plt.plot(data)
        plt.show()
        
