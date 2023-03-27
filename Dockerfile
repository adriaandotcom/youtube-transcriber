FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install git ffmpeg -y

RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install "git+https://github.com/openai/whisper.git" 

COPY scripts/ /app/scripts/

EXPOSE 8080

CMD ["python", "./scripts/app.py"]
