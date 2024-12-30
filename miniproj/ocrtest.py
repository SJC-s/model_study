# STEP 1 : import module
import easyocr

# STEP 2 : create inference object
# 기본적으로 detector 들어있음(기본값 중국어 ch_sim)
reader = easyocr.Reader(['ko','en']) # this needs to run only once to load the model into memory

# STEP 3 : load data
data = 'charger.jpg'

# STEP 4 : inference
result = reader.readtext(data, detail=0)
print(result)

# STEP 5 : post processing(후처리)