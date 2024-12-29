# STEP 1 : import module
import easyocr

# STEP 2 : create inference object
reader = easyocr.Reader(['ch_sim','en']) # this needs to run only once to load the model into memory

# STEP 3 : load data
data = 'chinese.jpg'

# STEP 4 : inference
result = reader.readtext(data)
print(result)

# STEP 5 : post processing(후처리)