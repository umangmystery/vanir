import psycopg2


try:
	conn = psycopg2.connect("host=host.docker.internal dbname=test_db user=root password=root")
	cursor = conn.cursor()

	print("Postgres server info:", conn.get_dsn_parameters())

	cursor.execute("Select * from public.tweets")

	record = cursor.fetchone()

	print("Data: ", record)
except (Exception, Error) as error:
	print("Error while connecting to PG", error)
finally:
	if (conn):
		cursor.close()
		conn.close()
		print("Connection closed")