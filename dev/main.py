from pyobvector import *
from sqlalchemy import Column, Integer, JSON
from sqlalchemy import func
import pandas as pd
import yaml
import random


def get_client(filename):
    with open(filename, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
            uri = data["ip"] + ":" + str(data["port"])
            user = data["user"] + "@" + data["tenant"]
            password = data["password"]
            db = data["db"]
            return ObVecClient(uri, user, password, db)
        except yaml.YAMLError as exc:
            print(exc)
            return None


def get_random_vector(id: int):
    random.seed(id)
    return [random.random() for _ in range(3)]


def main():
    client = get_client("config.yaml")
    table = "test_collection"
    # create partitioned table
    range_part = ObRangePartition(
        False,
        range_part_infos=[
            RangeListPartInfo("p0", 50),
            RangeListPartInfo("p1", 100),
            RangeListPartInfo("p2", "maxvalue"),
        ],
        range_expr="id",
    )

    cols = [
        Column("id", Integer, primary_key=True, autoincrement=False),
        Column("embedding", VECTOR(3)),
        Column("meta", JSON),
    ]
    client.create_table(table_name=table, columns=cols, partitions=range_part)
    print("create table success")
    # create vector index
    client.create_index(
        table_name=table,
        is_vec_index=True,
        index_name="vidx",
        column_names=["embedding"],
        vidx_params="distance=l2, type=hnsw, lib=vsag",
    )
    print("create index success")
    # insert data
    data1 = [{"id": i, "embedding": get_random_vector(i)} for i in range(50)]
    data1.extend([{"id": i, "embedding": get_random_vector(i)} for i in range(50, 100)])
    data1.extend(
        [{"id": i, "embedding": get_random_vector(i)} for i in range(100, 150)]
    )
    client.insert(table, data=data1)
    print("insert data success")
    res = client.ann_search(
        table,
        vec_data=[0, 0, 0],
        vec_column_name="embedding",
        distance_func=func.l2_distance,
        topk=5,
        output_column_names=["id"],
    )
    print(pd.DataFrame(res))


if __name__ == "__main__":
    main()
