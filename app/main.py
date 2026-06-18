import os
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select, func

from app.database import engine, get_session
from app.models import SQLModel, SentimentFeedback
from app.schemas import AnalyzeRequest, AnalyzeResponse, FeedbackRequest, FeedbackResponse
from app.ml import extract_title_from_url, analyze_sentiment

app = FastAPI(title="News Sentiment Analyzer")

os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

def generate_performance_graph(session: Session) -> tuple[str, float]:
    entries = session.exec(select(SentimentFeedback)).all()
    total = len(entries)
    
    if total == 0:
        return "", 0.0

    correct = sum(1 for e in entries if e.predicted_label == e.perceived_label)
    accuracy = (correct / total) * 100

    predicted_counts = [0, 0, 0]
    perceived_counts = [0, 0, 0]

    for e in entries:
        predicted_counts[e.predicted_label] += 1
        if e.perceived_label is not None:
            perceived_counts[e.perceived_label] += 1

    labels = ['Neutral', 'Positive', 'Negative']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, predicted_counts, width, label='Predicted')
    ax.bar(x + width/2, perceived_counts, width, label='Perceived')

    ax.set_ylabel('Counts')
    ax.set_title('Sentiment Prediction vs Perceived Sentiment')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close(fig)

    return graph_b64, accuracy

@app.get("/", response_class=HTMLResponse)
def index(request: Request, session: Session = Depends(get_session)):
    graph, accuracy = generate_performance_graph(session)
    return templates.TemplateResponse(
        request,
        "sample.html",
        {
            "graph": graph,
            "accuracy": f"{accuracy:.2f}"
        }
    )

@app.post("/analyze_sentiment/", response_model=AnalyzeResponse)
def analyze(data: AnalyzeRequest, session: Session = Depends(get_session)):
    headline = data.headline
    if not headline and data.url:
        headline = extract_title_from_url(data.url)

    if not headline:
        raise HTTPException(status_code=400, detail="Error retrieving headline")

    sentiment_val = analyze_sentiment(headline)
    
    feedback = SentimentFeedback(headline=headline, predicted_label=sentiment_val)
    session.add(feedback)
    session.commit()

    mapping = {0: 'NEUTRAL', 1: 'POSITIVE', 2: 'NEGATIVE'}
    return AnalyzeResponse(sentiment=mapping[sentiment_val], headline=headline)

@app.post("/store_feedback/", response_model=FeedbackResponse)
def store_feedback(data: FeedbackRequest, session: Session = Depends(get_session)):
    label_map = {'neutral': 0, 'positive': 1, 'negative': 2}
    perceived = label_map.get(data.userPerception.lower())

    if perceived is None:
        return FeedbackResponse(status="Invalid label")

    statement = select(SentimentFeedback).where(SentimentFeedback.headline == data.headline).order_by(SentimentFeedback.id.desc())
    entry = session.exec(statement).first()

    if entry:
        entry.perceived_label = perceived
        session.add(entry)
        session.commit()
        return FeedbackResponse(status="Feedback saved successfully")
    
    return FeedbackResponse(status="Headline not found")
