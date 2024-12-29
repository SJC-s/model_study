from typing import Annotated

from fastapi import FastAPI, Form

app = FastAPI()

# 자연어 처리 뼈대 코드
@app.post("/inference/")
async def inference(text: Annotated[str, Form()]):
    return {"result": text}