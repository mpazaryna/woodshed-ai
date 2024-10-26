from fastapi import FastAPI
from routers import alert, finance_qa, health, response_generation

app = FastAPI()


app.include_router(health.router)
app.include_router(alert.router)
app.include_router(response_generation.router)
app.include_router(finance_qa.router)


@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}
