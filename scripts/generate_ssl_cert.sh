#!/bin/bash

# Script to generate self-signed SSL certificates for development

CERT_DIR="certs"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"
CSR_FILE="$CERT_DIR/cert.csr"
CONFIG_FILE="$CERT_DIR/openssl.cnf"

# Create certs directory if it doesn't exist
mkdir -p $CERT_DIR

# Create OpenSSL configuration for Subject Alternative Names
cat > $CONFIG_FILE <<EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = State
L = City
O = RAG Server Development
CN = localhost

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = *.localhost
DNS.3 = 127.0.0.1
DNS.4 = ::1
IP.1 = 127.0.0.1
IP.2 = ::1
EOF

echo "Generating self-signed SSL certificate for development..."

# Generate private key
openssl genrsa -out $KEY_FILE 2048

# Generate certificate signing request
openssl req -new -key $KEY_FILE -out $CSR_FILE -config $CONFIG_FILE

# Generate self-signed certificate (valid for 365 days)
openssl x509 -req -days 365 -in $CSR_FILE -signkey $KEY_FILE -out $CERT_FILE \
    -extensions v3_req -extfile $CONFIG_FILE

# Clean up CSR and config file
rm -f $CSR_FILE $CONFIG_FILE

# Set appropriate permissions
chmod 600 $KEY_FILE
chmod 644 $CERT_FILE

echo "SSL certificate generated successfully!"
echo "Certificate: $CERT_FILE"
echo "Private Key: $KEY_FILE"
echo ""
echo "To use these certificates, add the following to your .env file:"
echo "SSL_ENABLED=true"
echo "SSL_CERT_FILE=$CERT_FILE"
echo "SSL_KEY_FILE=$KEY_FILE"
echo "HTTPS_PORT=8443"