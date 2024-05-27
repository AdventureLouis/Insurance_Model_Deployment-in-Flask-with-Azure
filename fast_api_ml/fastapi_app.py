from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import uvicorn

app = FastAPI()
model = pickle.load(open('model.pkl', 'rb'))  # loading the model
templates = Jinja2Templates(directory="templates")  # Assuming your templates are in a 'templates' directory

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, bmi: float = Form(...), New_Smoker: int = Form(...), age: int = Form(...)):
    prediction = model.predict([[bmi, New_Smoker, age]])  # this returns a list e.g. [127.20488798], so pick first element [0]
    output = round(prediction[0], 2)

    return templates.TemplateResponse('index.html', {"request": request, "prediction_text": f'A policy holder with {bmi} bmi, {New_Smoker} smoker and {age} age will incur insurance cost of $ {output}K'})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



 