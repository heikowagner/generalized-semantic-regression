FROM tensorflow/tensorflow:2.12.0-gpu

# RUN apt-get upgrade -y
RUN apt update -y
RUN apt install -y git git-lfs
RUN apt install -y make wget git gcc g++ lhasa libgmp-dev libmpfr-dev libmpc-dev flex bison gettext texinfo ncurses-dev autoconf rsync
RUN apt-get install -y sox ffmpeg libcairo2 libcairo2-dev
RUN apt-get install -y texlive-full
# COPY ./requirements.txt requirements.txt
# RUN pip install -r requirements.txt
RUN pip install RiskBERT