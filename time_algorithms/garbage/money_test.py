from es import *
from dateutil.parser import parse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate

def create_pandas(logs,field):
    times = []
    datas = []


    try:
        for item in logs:
            times.append(parse(item["@timestamp"]))
            if type(item[field]) != type(1):
                datas.append(sum(item[field]))
            else:
                datas.append(item[field])
                           
    except:
        pass

    
    assert(len(times)==len(datas)) 

    
    times_amount_series = pd.Series(data = datas,index= times)
    print(times_amount_series)

    return times_amount_series


def get_infor(logs):

    save = 0
    withdraw = 0

    save_times = 0
    down_times = 0

    for item in logs:
        if type(item[field]) != type(1):
            tmp = sum(item[field])
        else:
            tmp = item[field]

        if tmp > 0:
            save += tmp
            save_times += 1
        else:
            withdraw += tmp   
            down_times += 1
    return save,withdraw,save_times,down_times             
                           

    

def get_all_day_logs(from_time,end_time,command,filed):


    thirty_min_time_series = pd.date_range(from_time,end_time, freq= '30min')
    results = []

    for i in range(thirty_min_time_series.size-1):
        half_hour_save = 0
        half_hour_down = 0
        half_hour_save_times = 0 
        half_hour_down_times = 0

        five_min_time_series = pd.date_range(thirty_min_time_series[i],thirty_min_time_series[i+1], freq= '5min')
        #print(five_min_time_series.size)

        for j in range(five_min_time_series.size-1):
            start_time = five_min_time_series[j]
            to_time = five_min_time_series[j+1]
            logs = get_logs(start_time,to_time,command)
            if logs.total == 0:
                continue
            else:
                save,withdraw,save_times,down_times = get_infor(logs)
                half_hour_save  += save
                half_hour_down += withdraw
                half_hour_save_times += save_times
                half_hour_down_times += down_times

        results.append((half_hour_save,half_hour_down,half_hour_save_times,half_hour_down_times))
    return results         






def get_z_score(times_amount_series):
    delete_zero_datas = times_amount_series[times_amount_series!=0]
    data_z_score = (delete_zero_datas - delete_zero_datas.mean())/(delete_zero_datas.std())
    #print(data_z_score)
    return data_z_score 


def visual(results,from_time,to_time):
    fig=plt.figure(figsize=(8,6))
    x = list(range(48)

    ax1=fig.add_subplot(221)
    ax1.plot(x,[item[0] for item in results],label = "Save") 
    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.xticks(x,pd.date_range(from_time,to_time,freq='30min'))
    plt.legend(loc='upper center',ncol = 4,fontsize = 6)

    ax2=fig.add_subplot(222)
    ax2.plot(x,[item[1] for item in results],label = "Withdraw") 
    ax2.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.xticks(pd.date_range(from_time,to_time,freq='30min'))
    plt.legend(loc='upper center',ncol = 4,fontsize = 6)

    ax3=fig.add_subplot(223)
    ax3.plot(x,[item[2] for item in results],label = "Save Times") 
    ax3.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.xticks(pd.date_range(from_time,to_time,freq='30min'))
    plt.legend(loc='upper center',ncol = 4,fontsize = 6)

    ax4=fig.add_subplot(224)
    ax4.plot(x,[item[3] for item in results],label = "Withdraw Times") 
    ax4.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.xticks(pd.date_range(from_time,to_time,freq='30min'))
    plt.legend(loc='upper center',ncol = 4,fontsize = 6)

   

    plt.show()

    


if __name__ == '__main__':

    command = 'BettingCardAddBalance'
    field = "AddBalance"
    from_time = '2017-09-01 00:00:00'
    to_time = '2017-09-02 00:00:00'

    
    results = get_all_day_logs(from_time,to_time,command,field)
    visual(results,from_time,to_time)







    
