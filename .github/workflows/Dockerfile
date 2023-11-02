From: python3.11.6-alpine

COPY .github /app/
COPY assets /app/
COPY test /app/
COPY HeatCluster.py LICENSE README.md /app/

WORKDIR /app

RUN HeatCluster.py

CMD 
