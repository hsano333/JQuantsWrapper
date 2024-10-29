eval "echo \"$(cat /docker-entrypoint-initdb.d/sql_data)\"" > /tmp/tmp_sql
psql -f /tmp/tmp_sql

cp /tmp/server.key /var/lib/postgresql/data/
cp /tmp/server.crt /var/lib/postgresql/data/
