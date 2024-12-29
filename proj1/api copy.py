from fastapi import FastAPI, File, UploadFile

app = FastAPI()

# async 는 혹시 모를 비동기 작업을 위함

## @app.post("/files/")
## async def create_file(file: Annotated[bytes, File()]): # bytes : 
##     return {"file_size": len(file)}

# 파이썬은 싱글 스레드 동작, 비동기적 작동 불가
# 장점 : 동시 요청을 받을 수 있다(비동기)
# 뼈대 코드
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # # 형식이 맞지 않을 때 예외 처리
    # if file.content_type != "image/jpeg":
    #     return Exception()
    contents = await file.read()
    return {"filename": file.filename,
            "filesize": len(contents)}