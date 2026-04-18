#!/usr/bin/env bash
set -euo pipefail

# Installs Docker Engine on Ubuntu using Docker's official apt repository.
# Usage:
#   bash scripts/install_docker_ubuntu.sh
#
# Optional environment variables:
#   ADD_USER_TO_DOCKER_GROUP=1   Add the invoking non-root user to the docker group (default: 1)
#   ENABLE_DOCKER_SERVICE=1      Enable/start the docker service when systemd is available (default: 1)

ADD_USER_TO_DOCKER_GROUP="${ADD_USER_TO_DOCKER_GROUP:-1}"
ENABLE_DOCKER_SERVICE="${ENABLE_DOCKER_SERVICE:-1}"

log() {
  printf '\n==> %s\n' "$1"
}

die() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

run_as_root() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    die "This script needs root privileges. Re-run as root or install sudo."
  fi
}

if [[ ! -r /etc/os-release ]]; then
  die "Cannot detect OS because /etc/os-release is missing."
fi

# shellcheck disable=SC1091
source /etc/os-release

if [[ "${ID:-}" != "ubuntu" ]]; then
  die "This script only supports Ubuntu. Detected ID=${ID:-unknown} VERSION_ID=${VERSION_ID:-unknown}."
fi

ARCH="$(dpkg --print-architecture)"
CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
[[ -n "${CODENAME}" ]] || die "Could not determine Ubuntu codename."

log "Installing Docker apt prerequisites"
run_as_root apt-get update
run_as_root apt-get install -y ca-certificates curl

log "Configuring Docker's apt repository"
run_as_root install -m 0755 -d /etc/apt/keyrings
run_as_root curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
run_as_root chmod a+r /etc/apt/keyrings/docker.asc

run_as_root tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${CODENAME}
Components: stable
Architectures: ${ARCH}
Signed-By: /etc/apt/keyrings/docker.asc
EOF

log "Installing Docker Engine"
run_as_root apt-get update
run_as_root apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

if [[ "${ENABLE_DOCKER_SERVICE}" == "1" ]] && command -v systemctl >/dev/null 2>&1; then
  log "Enabling and starting Docker"
  run_as_root systemctl enable docker
  run_as_root systemctl restart docker
fi

TARGET_USER=""
if [[ "${EUID}" -eq 0 ]]; then
  TARGET_USER="${SUDO_USER:-}"
else
  TARGET_USER="${USER:-}"
fi

if [[ "${ADD_USER_TO_DOCKER_GROUP}" == "1" ]] && [[ -n "${TARGET_USER}" ]] && [[ "${TARGET_USER}" != "root" ]]; then
  log "Adding ${TARGET_USER} to the docker group"
  run_as_root usermod -aG docker "${TARGET_USER}"
fi

log "Docker installation complete"
docker --version || true

cat <<EOF

Next steps:
  1. If you were added to the docker group, refresh your session:
       newgrp docker
     or log out and back in.

  2. Verify Docker:
       docker run hello-world

  3. If you need GPU containers, also install NVIDIA Container Toolkit before using:
       docker run --gpus all ...

EOF
