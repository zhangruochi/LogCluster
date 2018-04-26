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


def connect_es(ip, port):
    #es = Elasticsearch(ip, http_auth=http_auth,scheme="https", port=port, timeout=500)
    es = Elasticsearch(ip, port=port, timeout=500)
    return es


def create_time_series(time_index, value_dict):
    time_series = pd.DataFrame(data=value_dict, index=time_index)
    return time_series


def adjust(row, times_series):

    times_index = [row.name - timedelta(days=x) for x in range(1, 11)]

    try:
        row = times_series.loc[times_index].mean()
    except:
        pass

    return row

# 周期调整值


def get_period_times_series(times_series):
    offset = timedelta(days=1)
    raw_times_series = times_series.copy(deep=True)

    times_series = times_series.apply(
        func=adjust, axis=1, args=(times_series,))

    # 每天48个数据的总平均
    day_mean = times_series.resample('1D').mean()

    # 观察每天48个数据点对于总平均值的偏离程度
    for i, day in enumerate(day_mean.index):
        date_index = str(day.date())
        times_series[date_index] = (
            times_series[date_index] - day_mean.iloc[i]) / day_mean.iloc[i]

    # 添加周期调整值之后的 times_series
    new_times_series = (raw_times_series / (1 + times_series)).dropna(axis=0)
    return new_times_series


def dynamic_threshold(times_series, n):
    eps = 1e-7
    numerator = times_series - times_series.rolling(
        n, min_periods=1).mean()
    numerator[numerator < eps] = 0

    denumerator = times_series.rolling(n, min_periods=1).std()
    denumerator[denumerator < eps] = 0

    times_series_z_score = numerator/denumerator

    return times_series_z_score


def get_z_score_for_one_group(time_series_for_one_group, n_points):

    period_times_series = get_period_times_series(time_series_for_one_group)
    times_series_z_score = dynamic_threshold(period_times_series, n_points)

    return times_series_z_score


def deal_reponse(response, n_points):
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
            time_series_for_one_group, n_points)
    return result_dict


def main(query):
    ip = ["10.10.11.67"]
    port = 9200
    query = query
    # http_auth = ('admin', 'adminxdstar123456')
    n_points = 10

    es = connect_es(ip, port)
    response = es.search(
        index=query['index'], body=query['content'])

    result_dict = deal_reponse(response, n_points)

    return result_dict


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
                                "lt": "2018-02-15T23:59:59"
                            }
                        }
                    }]
                }
            },

            "aggs": {
                "all_hullnums": {
                    "terms": {
                        "field": "hallnum",
                        "size": 1
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
    print(result_dict[563002])
