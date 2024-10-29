openssl req -x509 -days 3650 -nodes -newkey rsa:4096 -keyout /tmp/server.key -out /tmp/server.crt -config - << __EOF__
[req]
distinguished_name = req_distinguished_name
prompt = no

[req_distinguished_name]
C = JP
ST = Aichi
L = Toyota-shi
O = 42
OU = 42tokyo
CN = hsano.42.fr

[v3_req]
keyUsage = TestInception
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = mydomain.example
DNS.2 = sub.mydomain.example
__EOF__

