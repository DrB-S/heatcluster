ARG HEATCLUSTER_VER="0.4.12"

FROM ubuntu:jammy as app
USER root

# re-instantiating for the app build layer if using ARG as a global variable above
ARG HEATCLUSTER_VER

LABEL base.image="ubuntu:jammy"
LABEL dockerfile.version="3"
LABEL software="HeatCluster"
LABEL software.version="${HEATCLUSTER_VER}"
LABEL description="This software produces a heatmap for a SNP matrix"
LABEL website="https://github.com/DrB-S/HeatCluster"
LABEL license="https://github.com/DrB-S/HeatCluster/blob/master/LICENSE"
LABEL name="heatcluster/${HEATCLUSTER_VER}"
LABEL maintainer="Stephen Beckstrom-Sternberg"
LABEL maintainer.email="stephen.beckstrom-sternberg@azdhs.gov"

# Install Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
apt-utils python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install pandas numpy pathlib seaborn matplotlib scipy 

# Set /data as working dir
RUN mkdir /data
WORKDIR /data

RUN echo "installing heatcluster" && echo

COPY . .

RUN echo && echo && ls -latr /data && echo

# 'ENV' instructions set environment variables that persist from the build into the resulting image
# Use for e.g. $PATH and locale settings for compatibility with Singularity
ENV PATH="/heatcluster-${HEATCLUSTER_VER}/bin:$PATH" \
 LC_ALL=C

FROM app as test

# print help and version info
# Mostly this ensures the tool of choice is in path and is executable
RUN echo && echo "Show heatcluster help file and program version number:  " && echo && \
python3 heatcluster.py --help && \
python3 heatcluster.py --version
 
 RUN echo && echo "Run a test matrix thru the program" && \
python3 heatcluster.py -i test/snp-dists.txt && echo
