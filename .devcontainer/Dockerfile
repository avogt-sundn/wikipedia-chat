FROM mcr.microsoft.com/devcontainers/python:2-3.12-bookworm

ENV IGNORE_NOTICE=true


# BEGIN ------ install common utils: ping, dig, ncat, pip, yq
RUN apt-get update && apt-get install -y iputils-ping dnsutils file apache2-utils python3-pip httpie telnet \
    && pip install yq && pip install pipreqs


