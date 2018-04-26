def get_all_logs(request_data):
    es = Elasticsearch(
        ['10.10.11.65'],
        http_auth=('admin', 'adminxdstar123456'),
        scheme="https",
        port=9210
    )

    querydata = es.search(
        index=request_data['index'], body=request_data['content'], scroll='5m')

    mdata = querydata.get("hits").get("hits")

    if not mdata:
        return -1  # 没有查询到数据

    data = [d.get("_source") for d in mdata]
    sid = querydata['_scroll_id']

    while True:
        rs = es.scroll(scroll_id=sid, scroll='10s')
        temp = rs.get("hits").get("hits")
        if not temp:
            break
        data += [d.get("_source") for d in temp]
        print(data[-1].get("@timestamp"))

    print("sum of {} logs".format(len(data)))

    return logs_dict


def test(query):
    es = Elasticsearch(
        ['10.10.11.65'],
        http_auth=('admin', 'adminxdstar123456'),
        scheme="https",
        port=9210
    )

    querydata = es.search(
        index=request_data['index'], body=query, scroll='5m')

    # print(querydata)

    return querydata
    