FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install git ffmpeg npm -y

RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install "git+https://github.com/openai/whisper.git" 

COPY src/ /app/src/

COPY package-lock.json .
COPY package.json .
COPY postcss.config.js .
COPY tailwind.config.js .

RUN npm install
RUN npm run build

COPY src/js /app/static/js

EXPOSE 8080

CMD ["python", "/app/src/app.py"]
