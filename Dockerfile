FROM ubuntu:jammy as app
ARG HEATCLUSTER_VER="0.4.13"

LABEL base.image="ubuntu:jammy"
LABEL dockerfile.version="4"
LABEL software="heatcluster"
LABEL software.version="${HEATCLUSTER_VER}"
LABEL version="${HEATCLUSTER_VER}"
LABEL description="Visualize SNP matrix from snp-dists output"
LABEL website="https://github.com/DrB-S/heatcluster"
LABEL license="https://github.com/DrB-S/heatcluster/blob/master/LICENSE"
LABEL name="heatcluster/${HEATCLUSTER_VER}"
LABEL maintainer="Stephen Beckstrom-Sternberg"
LABEL maintainer.email="stephen.beckstrom-sternberg@azdhs.gov"

RUN echo "Installing python and pip" && echo
RUN apt-get update && apt-get upgrade -y && \
  apt-get install -y --no-install-recommends \
  ca-certificates \
  wget \
  procps \
  python3 \
  python3-pip && \
  apt-get autoclean && rm -rf /var/lib/apt/lists/*

RUN echo "Installing python packages" && echo
RUN pip3 install --no-cache argparse pandas numpy pathlib seaborn matplotlib scipy --upgrade-strategy=only-if-needed

RUN pwd && ls -la

RUN echo "Installing heatcluster from archive: " && echo
RUN wget -q https://github.com/DrB-S/heatcluster/archive/refs/tags/v${HEATCLUSTER_VER}.tar.gz && \
  tar -vxf v${HEATCLUSTER_VER}.tar.gz && \
  pwd && ls -latr && \
  rm v${HEATCLUSTER_VER}.tar.gz && \
  cd heatcluster-${HEATCLUSTER_VER} && ls -la 

#RUN mkdir heatcluster-${HEATCLUSTER_VER} && cd heatcluster-${HEATCLUSTER_VER}

#WORKDIR /heatcluster-${HEATCLUSTER_VER}
#COPY . .

RUN ls -la

ENV PATH=/heatcluster-${HEATCLUSTER_VER}:$PATH

RUN pwd; ls -la
WORKDIR /heatcluster-${HEATCLUSTER_VER}

FROM app as test

RUN echo && echo "Show heatcluster version number and help file:  " && echo 
RUN heatcluster.py --version && echo && \
heatcluster.py --help

RUN echo && echo "Test a small and medium matrix :" && echo && echo && \
heatcluster.py -i test/small_matrix.csv -t png -o small_test && \
heatcluster.py -i test/med_matrix.txt -t pdf -o med_test

RUN echo && ls -lh|tail && echo "DONE"
CMD [ "/bin/ls", "-l" ]
