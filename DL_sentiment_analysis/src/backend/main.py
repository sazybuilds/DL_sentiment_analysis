#main API entry

from fastapi import FastAPI

#render the frontend styling through the API
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from DL_sentiment_analysis.src.backend.api.routes import router #get router
from DL_sentiment_analysis.src.frontend.app import FRONTEND_STATIC_DIR

app = FastAPI(
    title="Movie Review Sentiment Analyzer",
    description="A local BERT-based movie-review sentiment classifier.",
    version="1.0.0",
)

app.include_router(router, prefix="/api")
app.mount(
    "/assets",
    StaticFiles(directory=FRONTEND_STATIC_DIR),
    name="assets",
)


#at the first endpoint, we are returning the html filec, styling rendered
@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    return FileResponse(FRONTEND_STATIC_DIR / "index.html")
