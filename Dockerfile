FROM python:3.9-slim

RUN apt-get update && apt-get install git ffmpeg npm -y

ENV APP_HOME /app
WORKDIR /app
RUN mkdir -pv $APP_HOME
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install "git+https://github.com/openai/whisper.git"

ADD . $APP_HOME

ENV NODE_ENV production
ENV NPM_CONFIG_LOGLEVEL warn
RUN npm install
RUN npm run build

COPY src/js /app/static/js

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "-w", "4", "--timeout", "600", "--chdir", "src", "app:application"]
