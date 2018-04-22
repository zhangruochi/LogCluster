#!/usr/bin/env python3

# info
# -name   : zhangruochi
# -email  : zrc720@gmail.com

from pyes import *
import pandas
from dateutil.parser import parse
import pandas as pd


def get_logs(start_time, to_time):

    conn = ES("10.10.11.64:9202")

    #Search(MatchAllQuery(), size=20)

    must1 = RangeQuery(
        ESRange("@timestamp", from_value=start_time, to_value=to_time))
    must2 = QueryStringQuery("001018601", "posid")


    q = BoolQuery(must = [must1,must2])

    results = conn.search(query = q, indices="logstash-management-success-2017-10",doc_types = "management")

    return results


if __name__ == '__main__':
    start_time = "2017-10-20T14:37:04.348Z"
    to_time = "2017-10-30T14:37:04.348Z"

    logs = get_logs(start_time, to_time)
    print(logs.total)
