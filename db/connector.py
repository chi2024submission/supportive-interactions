from typing import List
import psycopg2
from config import config

class DB:
    def __init__(self) -> None:
        super().__init__()
        self.conn = self._get_connection()

    def _get_connection(self):
        conn = None
        try:
            # read connection parameters
            params = config()

            # connect to the PostgreSQL server
            print('Connecting to the PostgreSQL database...')
            conn = psycopg2.connect(**params)
            return conn
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

    def close_connection(self) -> None:
        if self.conn is not None:
            self.conn.close()



