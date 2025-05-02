# Jupyter Notebooks with Mullvad VPN for Asia Access

This setup allows running Jupyter notebooks through VPN points in Asia using Mullvad VPN credentials.

## Prerequisites

- Docker and Docker Compose installed on your system
- A valid Mullvad VPN account

## Setup

1. Copy the example environment file and configure it with your credentials:

```bash
cp .env.example .env
```

2. Edit the `.env` file with your Mullvad account number and preferred settings:

```
MULLVAD_ACCOUNT=your_mullvad_account_number
MULLVAD_LOCATION=sg  # or jp, hk, etc.
JUPYTER_PASSWORD=your_secure_password  # Optional
```

Available Asian locations:
- `sg`: Singapore
- `jp`: Japan
- `hk`: Hong Kong 
- `my`: Malaysia
- `tw`: Taiwan

## Usage

1. Build and start the container:

```bash
docker-compose up -d
```

2. Check the logs to get the Jupyter URL with token (if no password is set):

```bash
docker-compose logs jupyter-vpn
```

3. Access Jupyter in your browser using the URL from the logs, which will look something like:
   
```
http://localhost:8888/?token=abc123...
```

4. To stop the service:

```bash
docker-compose down
```

## Troubleshooting

### VPN Connection Issues

If the VPN connection fails to establish:

1. Check your Mullvad account number
2. Verify that the selected location is available
3. Check the container logs:

```bash
docker-compose logs jupyter-vpn
```

### Jupyter Access Issues

If you can't access Jupyter:

1. Make sure port 8888 is not being used by another service
2. Check the container is running:

```bash
docker-compose ps
```

## Notes

- All your notebooks will be stored in the `research_notebooks` directory
- VPN configuration is persisted across container restarts
- The container has NET_ADMIN capability which is required for VPN functionality 