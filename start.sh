#!/usr/bin/env bash
set -e

mkdir -p /root/.ssh /var/run/sshd
chmod 700 /root/.ssh

# Runpod injects PUBLIC_KEY for SSH access on custom templates
if [ -n "${PUBLIC_KEY:-}" ]; then
  echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

# Make sure host keys exist
ssh-keygen -A

# Optional: start Jupyter on 8888 if you actually want that port to do something
# jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &

exec /usr/sbin/sshd -D -e