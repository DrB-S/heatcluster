FROM ubuntu:jammy
ARG HEATCLUSTER_VER="0.4.12"
LABEL base.image="ubuntu:jammy"
LABEL dockerfile.version="1"
LABEL software.version="${HEATCLUSTER_VER}"
LABEL version="${HEATCLUSTER_VER}"
LABEL website="https://github.com/DrB-S/heatcluster"
LABEL license="https://github.com/DrB-S/heatcluster/blob/master/LICENSE"
LABEL name="heatcluster/${HEATCLUSTER_VER}"
LABEL maintainer="Stephen Beckstrom-Sternberg"
LABEL maintainer.email="stephen.beckstrom-sternberg@azdhs.gov"
#LABEL org.opencontainers.image.source=https://github.com/DrB-S/heatcluster

# Install Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
ca-certificates \
  wget \
  procps \
  python3 \
  python3-pip && \
  apt-get autoclean && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache argparse pandas numpy pathlib seaborn matplotlib scipy --upgrade-strategy=only-if-needed

WORKDIR /heatcluster

RUN echo "Installing heatcluster" && echo

COPY . .

ENV PATH="/heatcluster:$PATH"
RUN echo && echo && ls -ltr /heatcluster && echo

RUN echo && echo "Show heatcluster help file and version number:  " && echo && \
heatcluster.py --help && \
heatcluster.py --version

RUN echo && echo "Run a test matrix thru the program:" && \
heatcluster.py -i test/snp-dists.txt && echo 
RUN ls -ltr .|tail && echo "DONE"
