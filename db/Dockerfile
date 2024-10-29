FROM postgres:17-bookworm
RUN apt-get update && apt-get upgrade -y && apt-get install vim -y

COPY ./sql_data /docker-entrypoint-initdb.d/
COPY ./exe_sql.sh /docker-entrypoint-initdb.d/
