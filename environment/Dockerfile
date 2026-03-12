# hash:sha256:8be07a36d2b4c6f200ef5f55c7ba7ed89483a028fc17c44b57faab47a9b638c6
FROM registry.codeocean.com/codeocean/pytorch:2.1.0-cuda11.8.0-mambaforge23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install -U --no-cache-dir \
    pillow==11.0.0 \
    scikit-image==0.24.0 \
    scipy==1.14.1
