# FROM frolvlad/alpine-miniconda3:python3.7

FROM continuumio/miniconda3:master-alpine

COPY requirements.txt .

RUN pip install -r requirements.txt && \
    rm requirements.txt

EXPOSE 80

COPY ./source /app


#CMD ["pip", "freeze"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port","80"]