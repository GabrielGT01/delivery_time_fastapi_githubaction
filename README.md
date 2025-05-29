# 📦 Delivery Time Prediction App

A machine learning-powered web application that predicts food delivery times based on various factors. Built with FastAPI, Docker, and deployed using GitHub Actions to AWS services like ECR, ECS.

---

## 📊 Dataset Overview

The application utilizes the `Food_Delivery_Times.csv` dataset, which includes features such as:

- **Order Details:** Order ID, order time, pickup time.
- **Delivery Partner Information:** Partner ID, rating.
- **Location Data:** Restaurant and delivery locations.
- **Weather Conditions:** Temperature, weather conditions.
- **Traffic Density:** Low, medium, high, jam.

These features are used to train a regression model that predicts the delivery time for food orders.

---

## 🧠 Machine Learning Workflow

1. **Data Preprocessing:**
   - Importing data from Mongo Db
   - Handling missing values.
   - Encoding categorical variables.
   - Feature scaling.

3. **Model Training:**
   - Utilized algorithms like XGBoost, Random Forest, and Linear Regression.
   - Evaluated models using metrics such as RMSE and R² score.

4. **Model Selection:**
   - Selected the best-performing model based on evaluation metrics.
   - Serialized the model using `joblib` for deployment.

5. **Experiment Tracking:**
   - Used MLflow to track experiments and model performance.

---

## 🖥️ Application Architecture

- **Backend Framework:** FastAPI for building RESTful APIs.
- **Frontend:** Jinja2 templates for rendering HTML pages.
- **Model Integration:** Loaded the serialized model to make predictions based on user input.
- **Logging:** Implemented logging to monitor application behavior and errors.

---

## 🐳 Dockerization

The application is containerized using Docker for consistency and ease of deployment.

**Dockerfile Overview:**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

```


## ⚙️ CI/CD Pipeline with GitHub Actions and AWS

This project features an automated deployment pipeline configured using GitHub Actions:

- **Trigger:** On push to the `main` branch.
- **Jobs:**
  - **Integration:** Linting and unit tests.
  - **Build and Push:** Docker image is built and pushed to Amazon ECR.
  - **Deployment:** The latest image is pulled and run on an AWS EC2 instance.

**Workflow File:** `.github/workflows/main.yml`

**Key Steps:**
1. Checkout code.
2. Install dependencies.
3. Configure AWS credentials.
4. Build Docker image.
5. Push image to ECR.
6. SSH into EC2 and deploy the new container.

---

## ☁️ Cloud Infrastructure

This project directly uses the following AWS services:

### 1. Mongo DB(Database)
- **Purpose:** Used to host the dataset from which the model was trained.


### 1. Amazon Elastic Container Registry (ECR)
- **Purpose:** Stores Docker images securely.
- **How Used:** GitHub Actions pushes built images to ECR after every code change.

### 2. Amazon EC2 (Elastic Compute Cloud)
- **Purpose:** Hosts and runs the Docker container with the FastAPI application.
- **How Used:** The deployment job connects to EC2 via SSH, pulls the latest image from ECR, and runs the container.



