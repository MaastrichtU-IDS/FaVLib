FROM maven:3-jdk-8

LABEL maintainer "Ammar Ammar <ammar257ammar@gmail.com>"

ENV APP_DIR /app

WORKDIR $APP_DIR

COPY . .

ENTRYPOINT ["java","-jar","favlib-0.0.1-SNAPSHOT-jar-with-dependencies.jar"]
