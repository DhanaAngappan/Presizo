FROM python:3.6
# WORKDIR /app
# COPY . .
# RUN pip install -r requirements.txt
# CMD ["python", "app.py"]
# EXPOSE 5000

RUN mkdir -p /app_presizo
COPY . /app_presizo
RUN python3 -m pip install -r /app_presizo/requirements.txt
EXPOSE 5000
CMD ["python" , "/app_presizo/app.py"]