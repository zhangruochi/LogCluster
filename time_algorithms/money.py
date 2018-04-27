from dateutil.parser import parse
import pandas as pd
import numpy as np
from dateutil.parser import parse
#import matplotlib.pyplot as plt
from pyes import *
import os
from collections import Counter
from collections import defaultdict
import pickle
import dateutil
from datetime import timedelta

try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser


def get_parameters(section):
    cf = ConfigParser.ConfigParser()

    if os.path.exists("config.cof"):
        cf.read('config.cof')
    else:
        print("there is no config.cof!")
        exit()

    es_option_dict = dict()
    for key, value in cf.items(section):
        es_option_dict[key] = eval(value)

    return es_option_dict


def get_logs(from_time, to_time):

    conn = ES(es_option_dict["es_address"])

    must1 = RangeQuery(
        ESRange("@timestamp", from_value=from_time, to_value=to_time))

    musts = [must1]

    for item in es_option_dict["valid_dataset"]:
        musts.append(QueryStringQuery(item[1], item[0]))

    #for item in influencer_option_dict["influencers"]:
    #    musts.append(QueryStringQuery(item[1], item[0]))

    q = BoolQuery(must=musts)

    results = conn.search(
        query=q, indices=es_option_dict["index"], doc_types=es_option_dict["doc_type"])

    return results


def get_aggregation_function(name):
    if name == "sum":
        return sum
    elif name == "z_score":
        return get_z_score

    elif name == "min":
        return min
    elif name == "mean":
        return np.mean
    elif name == "max":
        return max


# 对每一个time span, 针对 detecter 进行多种 aggregation
def get_infor(logs):
    result_dict = {}
    for detecter in detecter_option_dict["detecters"]:
        all_values = []

        field = detecter[0]
        aggregation_function = get_aggregation_function(detecter[1])

        for item in logs:

            if type(item[field]) != type(1):
                single = sum(item[field])
            else:
                single = item[field]

            all_values.append(single)

        result = aggregation_function(all_values)
        result_dict[detecter[1]] = result

    return result_dict


def get_all():

    time_series = pd.date_range(
        es_option_dict["timestamp"][0], es_option_dict["timestamp"][1], freq=es_option_dict["time_span"])

    all_results = defaultdict(list)

    for i in range(time_series.size - 1):
        # 30min 的时间里所有的 log
        span_logs = []
        print(time_series[i])
        one_min_time_series = pd.date_range(
            time_series[i], time_series[i+1], freq='5min')

        for j in range(one_min_time_series.size - 1):
            start_time = one_min_time_series[j]
            end_time = one_min_time_series[j + 1]
            xx = []
            try:
                logs = get_logs(start_time, end_time)
                for log in logs:
                    xx.append(log)

                
            except:
                logs = []

            """    
            if len(logs) == 0:
                continue
            else:
                span_logs = span_logs + logs

            span_logs = []
            """
        """    
        if len(span_logs) == 0:
            for detecter in detecter_option_dict["detecters"]:
                all_results[detecter[1]].append(0)
            continue

        span_results = get_infor(span_logs)
        for key, value in span_results.items():
            all_results[key].append(value)

    times_series = create_times_series(all_results)
    """
    return times_series


def create_times_series(all_results):

    time_index = pd.date_range(
        es_option_dict["timestamp"][0], es_option_dict["timestamp"][1], freq=es_option_dict["time_span"])[1:]

    times_series = pd.DataFrame(data=all_results, index=time_index)

    return times_series


def ratio(values):
    return Counter(values)[0] / len(values)


def get_aggregation_function(name):
    if name == "sum":
        return sum
    elif name == "min":
        return min
    elif name == "mean":
        return np.mean
    elif name == "max":
        return max
    elif name == "count":
        return len
    elif name == "ratio":
        return ratio


def adjust(row, times_series):
    times_index = [row.name - timedelta(days=x) for x in range(1, 11)]

    try:
        times_series.loc[row.name] = times_series.loc[times_index].mean()
    except:
        pass


def get_period_times_series(times_series, period = "D"):
    offset = timedelta(days=1)
    raw_times_series = times_series.copy(deep=True)
    # print(times_series.resample('D').mean())
    times_series.apply(func=adjust, axis=1, args=(times_series,))

    # 一段时间的总平均(day or week)
    day_mean = times_series.resample(period).mean()

    # 观察每天48个数据点对于总平均值的偏离程度
    for i, day in enumerate(day_mean.index):
        date_index = str(day.date())
        print(times_series[date_index])

        times_series[date_index] = (
            times_series[date_index] - day_mean.iloc[i]) / day_mean.iloc[i]

    # 添加周期调整值之后的 times_series
    new_times_series = raw_times_series / (1 + times_series)

    return new_times_series


def dynamic_threshold(times_series,n):

    times_series = (times_series - times_series.shift(1).rolling(n).mean()
                    ) / times_series.shift(1).rolling(n).std()

    return times_series


def visual(all_results):

    dti = pd.date_range(
        es_option_dict["timestamp"][0], es_option_dict["timestamp"][1], freq=es_option_dict["time_span"])
    pydate_array = dti.to_pydatetime()
    #%Y-%m-%d %H:%M%S"
    date_only_array = list(np.vectorize(
        lambda s: s.strftime('%H:%M'))(pydate_array))[0:-1]

    x = np.arange(len(date_only_array))

    plt.figure(1, figsize=(8, 6))

    num = len(all_results.keys())
    row = num / 2
    if num % 2 != 0:
        row += 1
    column = 2
    index = 1

    for aggregation, values in all_results.items():

        plt.subplot(row, column, index)
        plt.plot(x, values)
        #plt.legend(loc='upper center', ncol=4, fontsize=6)
        plt.title(aggregation)
        index += 1

    plt.show()


if __name__ == '__main__':
    
    es_option_dict = get_parameters("ES")
    influencer_option_dict = get_parameters("INFLUENCERS")
    detecter_option_dict = get_parameters("DETETERS")
    period_option_dict = get_parameters("PERIODISM")

    #start = "2017-10-01T02:00:00"
    #end = "2017-10-31T03:00:00"
    #logs = get_logs(start,end)
    #print(logs.total)
    #xx = []
    #for log in logs[1:10000]:
    #    xx.append(log)

    #exit()

    


    times_series = get_all()
    with open("times_series.pkl","wb") as f:
        pickle.dump(times_series,f)
    """
    with open("times_series.pkl", "rb") as f:
        times_series = pickle.load(f)
    """
    #print(times_series)    
    times_series = get_period_times_series(times_series)
    dynamic_threshold(times_series, 10)
