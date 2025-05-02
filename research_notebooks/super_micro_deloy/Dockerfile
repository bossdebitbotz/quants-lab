FROM continuumio/miniconda3

# Install system dependencies including Mullvad VPN
RUN apt-get update && \
    apt-get install -y sudo wget curl gnupg ca-certificates apt-transport-https libusb-1.0 python3-dev gcc g++ make && \
    # Add Mullvad repository
    wget --https-only https://repository.mullvad.net/deb/mullvad-keyring.asc -O /etc/apt/trusted.gpg.d/mullvad-keyring.asc && \
    echo "deb https://repository.mullvad.net/deb/ stable main" | tee /etc/apt/sources.list.d/mullvad.list && \
    apt-get update && \
    apt-get install -y mullvad-vpn && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /quants-lab

# Copy the environment file
COPY environment.yml .

# Create the conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "quants-lab", "/bin/bash", "-c"]

# Install additional Jupyter dependencies
RUN conda run -n quants-lab pip install jupyter jupyterlab jupyter_contrib_nbextensions ipywidgets

# Copy research notebooks
COPY research_notebooks/ research_notebooks/

# Copy startup script
COPY research_notebooks/super_micro_deloy/start.sh /start.sh
RUN chmod +x /start.sh

# Expose Jupyter port
EXPOSE 8888

# Set entrypoint to the startup script
ENTRYPOINT ["/start.sh"] 