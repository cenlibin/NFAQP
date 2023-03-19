import pyverdict
import pymysql
from utils import *


host = 'localhost'
user = 'root'
password = '7837'
port = 3306

source_db = 'AQP'
target_db = 'verdictdb'
dataset = 'orders'
sample_rate = 0.01

if __name__ == "__main__":

    
    connect_string = f'jdbc:mysql://{host}:{port}?user={user}&password={password}&useSSL=False'
    mysql_conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        passwd=password,
        autocommit=True
    ).cursor()

    verdict_conn = pyverdict.VerdictContext(connect_string)

    T  = TimeTracker()
    sql = f"DROP TABLE IF EXISTS {target_db}.{dataset};"
    mysql_conn.execute(sql)
    T.report_interval_time_sec(sql)

    sql = f'CREATE SCRAMBLE {target_db}.{dataset} FROM {source_db}.{dataset} size {sample_rate}'
    verdict_conn.sql(sql)
    T.report_interval_time_sec(sql)