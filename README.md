# RAG-Invasive-Species

## Overview

This project is a Retrieval-Augmented Generation (RAG) system designed to help users explore and understand invasive species in Pennsylvania. It combines a **Flask API backend** with a **React frontend** to deliver an interactive, intelligent tool that uses LLM-powered retrieval and summarization.

## Installation

### Backend (Flask)

1. Navigate to the backend directory:

```
cd backend
```

2. Create and activate a virtual environment (optional):

```
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the Flask server:

```
cd src
uvicorn main:app --reload
```

### Frontend (React)

1. Navigate to the frontend directory:

```
cd frontend
```

2. Install dependencies:

```
npm install
```

3. Start the development server:

```
npm run dev
```

## Usage

Once both the **Flask backend** and **React frontend** are running:

* The frontend will send user queries to the Flask API.
* The backend retrieves relevant documents from the knowledge base and combines them with an LLM to generate RAG responses.
* Results display in the UI with citations or extracted sources depending on your configuration.