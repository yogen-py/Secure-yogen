#!/bin/bash

echo "=== Federated Learning Network Setup ==="
echo "Checking and configuring network for FL..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo -e "\n1. Installing required tools..."
apt-get update
apt-get install -y net-tools netcat ufw psmisc

echo -e "\n2. Checking port 50051..."
if ss -tuln | grep -q ':50051'; then
    echo "Port 50051 is in use. Attempting to kill the process..."
    fuser -k 50051/tcp || echo "Could not kill process using port 50051"
    sleep 2
else
    echo "Port 50051 is available"
fi

echo -e "\n3. Configuring firewall..."
# Enable UFW if not active
if ! ufw status | grep -q "Status: active"; then
    echo "Enabling UFW..."
    ufw --force enable
fi

# Allow port 50051 TCP/UDP
echo "Opening port 50051 (TCP/UDP)..."
ufw allow 50051/tcp
ufw allow 50051/udp
ufw reload

echo -e "\n4. Current network status:"
echo "Network interfaces:"
ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}/\d+'
echo -e "\nOpen ports:"
ss -tuln | grep 50051
echo -e "\nFirewall status (50051):"
ufw status | grep 50051

echo -e "\n5. Testing peer connectivity..."
# Read peer IPs from host_config.yaml
if [ -f "host_config.yaml" ]; then
    echo "Testing connections to peers listed in host_config.yaml:"
    grep -E 'ip:\s*' host_config.yaml | while read -r line; do
        ip=$(echo "$line" | awk '{print $2}')
        echo -n "Testing $ip:50051 - "
        if nc -z -w 3 "$ip" 50051 2>/dev/null; then
            echo "Success"
        else
            echo "Failed"
        fi
    done
else
    echo "host_config.yaml not found"
fi

echo -e "\n=== Setup Complete ==="
echo "To monitor connections in real-time, run:"
echo 'watch -n 1 "ss -tuln | grep 50051"'

