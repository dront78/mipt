FROM ubuntu:jammy

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y --no-install-recommends \
  && apt install -y libopencv-dev libatlas3-base dotnet-runtime-6.0 aspnetcore-runtime-6.0 \
  && rm -rf /var/lib/apt/lists/*

# let's disable dotnet telemetry
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1

# let's set the dotnet bind url
ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080/tcp

# let's set the working directory
WORKDIR /app
ADD ./bin/Release/net6.0/publish /app

ENTRYPOINT [ "/app/mipt" ]