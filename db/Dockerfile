FROM postgres:17-bookworm
RUN apt-get update && apt-get upgrade -y && apt-get install vim -y

COPY ./make_ca.sh /tmp/make_ca.sh
COPY ./sql_data /docker-entrypoint-initdb.d/
COPY ./exe_sql.sh /docker-entrypoint-initdb.d/

RUN chmod 755 /tmp/make_ca.sh
RUN sh /tmp/make_ca.sh
RUN chmod 755 /tmp/server.key
RUN chmod 755 /tmp/server.crt
