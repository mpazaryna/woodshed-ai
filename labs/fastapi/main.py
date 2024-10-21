from fastapi import FastAPI
from routers import alert, health, response_generation

app = FastAPI()


app.include_router(health.router)
app.include_router(alert.router)
app.include_router(response_generation.router)


@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}
