from fastapi import FastAPI
from routers import alert, health

app = FastAPI()


app.include_router(health.router)
app.include_router(alert.router)


@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}
