from dateutil.parser import parse
import pandas as pd
import numpy as np
import os
from collections import Counter
from collections import defaultdict
import pickle
import dateutil
from datetime import timedelta
from elasticsearch import Elasticsearch
from matplotlib import pyplot as plt


def connect_es(ip, port):
    #es = Elasticsearch(ip, http_auth=http_auth,scheme="https", port=port, timeout=500)
    es = Elasticsearch(ip, port=port, timeout=500)
    return es


# 创建有效时间内的时间序列
def create_time_series(time_index, value_dict):

    valid_days = list(Counter([str(item).split(" ")[0]
                               for item in time_index]).keys())

    valid_day_time = None

    for day in valid_days:
        tmp = pd.date_range("{} 10:00:00+00:00".format(day),
                            "{} 21:30:00+00:00".format(day), freq="30min")
        if valid_day_time is None:
            valid_day_time = tmp
        else:
            valid_day_time = valid_day_time.append(tmp)

    time_series = pd.DataFrame(data=value_dict, index=time_index)
    time_series = time_series[time_index.isin(valid_day_time)]  # 去掉无用时间

    return time_series


def adjust(row, times_series):

    times_index = [row.name - timedelta(days=x) for x in range(1, 11)]

    try:
        row = times_series.loc[times_index].mean()
    except:
        pass

    return row

# 周期调整值

def add_n_day_period(times_series, ndays):
    bias = times_series.copy(deep=True)

    bias = bias.apply(
        func=adjust, axis=1, args=(times_series,))

    # n 天的总平均
    days_mean = bias.resample('{}D'.format(ndays)).mean()

    # 观察 n 个数据点对于总平均值的偏离程度
    for i, days in enumerate(days_mean.index):
        days_index = str(days.date())
        bias[days_index] = (bias[days_index] -
                            days_mean.iloc[i]) / days_mean.iloc[i]

    # 添加周期调整值之后的 times_series
    new_times_series = (times_series / (1 + bias))
    return new_times_series


def adjust_for_week(row, times_series):
    #过去12个星期的索引
    times_index = [row.name - timedelta(days=7) for x in range(1, 13)]

    try:
        row = times_series.loc[times_index].mean()
    except:
        pass

    return row


def add_seven_static_period(time_series):
    new_times_series = time_series.resample("1D").mean()
    bias = new_times_series.copy(deep = True)

    bias = bias.apply(
        func=adjust_for_week, axis=1, args=(new_times_series,))

    bias = (new_times_series - bias) / bias

    # 添加周期调整值之后的 times_series
    for i, day in enumerate(bias.index):

        time_series[str(day.date())] = time_series[str(day.date())] / ( 1 + bias.iloc[i])

    return time_series


def dynamic_threshold(times_series, window):

    eps = 1e-7
    numerator = times_series - times_series.rolling(
        window, min_periods=1).mean()
    #numerator[numerator < eps] = 0

    denumerator = times_series.rolling(window, min_periods=100).std()
    #denumerator[denumerator < eps] = 0
    denumerator = denumerator + eps

    times_series_z_score = (numerator / denumerator).fillna(value=0, axis=0)

    return times_series_z_score


def get_z_score_for_one_group(time_series_for_one_group, window, ndays):

    period_times_series = add_n_day_period(time_series_for_one_group, ndays)
    #period_times_series = add_seven_static_period(time_series_for_one_group)
    times_series_z_score = dynamic_threshold(period_times_series, window)

    return times_series_z_score


def deal_reponse(response, window, ndays):
    result_dict = {}
    offset = timedelta(hours=8)

    for hullnum in response["aggregations"]["all_hullnums"]['buckets']:
        print(hullnum["key"])
        timestamps = []
        value_dict = defaultdict(list)
        for time_spans in hullnum["time_spans"]["buckets"]:
            for key, item in time_spans.items():
                if key.startswith("the"):
                    value_dict[key].append(time_spans[key]['value'])
                    continue
                if key == 'key_as_string':
                    timestamps.append(pd.Timestamp(
                        time_spans['key_as_string'], tz="GMT") + offset)
        time_index = pd.DatetimeIndex(timestamps)
        time_series_for_one_group = create_time_series(
            time_index, value_dict)

        result_dict[hullnum["key"]] = get_z_score_for_one_group(
            time_series_for_one_group, window, ndays)
    return result_dict


def main(query):
    ip = ["10.10.11.67"]
    port = 9200
    query = query
    # http_auth = ('admin', 'adminxdstar123456')
    window = 100
    ndays = 1

    es = connect_es(ip, port)
    response = es.search(
        index=query['index'], body=query['content'])

    result_dict = deal_reponse(response, window, ndays)

    return result_dict


def visual(result_dict):
    plt.figure(1, figsize=(20, 5))

    keys_list = list(map(str, result_dict.keys()))
    values = list(result_dict.values())

    x = list(range(values[0].shape[0]))

    plt.plot(x, values[0], color="#FF4040",
             label="hullnum = {}".format(keys_list[0]))
    plt.plot(x, values[1], color="#ADFF2F",
             label="hullnum = {}".format(keys_list[1]))
    plt.plot(x, values[2], color="#66CD00",
             label="hullnum = {}".format(keys_list[2]))
    plt.plot(x, values[3], color="#0000FF",
             label="hullnum = {}".format(keys_list[3]))
    # plt.plot(x, values[4], color = "#A9A9A9", label = keys_list[3])

    plt.legend(loc='upper center', ncol=4, fontsize=6)
    plt.title("SUM (period = 1 days, rolling window = 100 points)")
    # plt.ylim(2.5,3)

    plt.show()


if __name__ == '__main__':
    query = {

        "content": {
            "size": 0,
            "query": {
                "bool": {
                    "must": [{
                        "match": {
                            "operation": "BettingCardAddBalance"
                        }
                    }, {
                        "match": {
                            "status": "NO_EXCEPTION"
                        }
                    }, {
                        "range": {
                            "@timestamp": {
                                "gte": "2018-02-01T00:00:00",
                                "lt": "2018-02-15T00:00:00"
                            }
                        }
                    }]
                }
            },

            "aggs": {
                "all_hullnums": {
                    "terms": {
                        "field": "hallnum",
                        "size": 5
                    },
                    "aggs": {
                        "time_spans": {
                            "date_histogram": {
                                "field": "@timestamp",
                                "interval": "30m"

                            },
                            "aggs": {
                                "the_sum": {
                                    "sum": {
                                        "field": "add_balance"
                                    }
                                },
                            }
                        }
                    }
                }
            }
        },
        "index": "logstash-management-others-success-*"
    }

    # 所有的 hullnum 组成的 list
    # response["aggregations"]["all_hullnums"]['buckets']

    # 此时的 key 是 hullnum
    # response["aggregations"]["all_hullnums"]['buckets'][0]["key"]

    # 此时的 list 是每个 hullnum 所有30min 时间段的 aggregation 值
    # response["aggregations"]["all_hullnums"]['buckets'][0]["time_spans"]["buckets"]

    result_dict = main(query)

    with open("result_dict.pkl", "wb") as f:
        pickle.dump(result_dict, f)

    with open("result_dict.pkl", "rb") as f:
        result_dict = pickle.load(f)

    print(result_dict)
    visual(result_dict)
