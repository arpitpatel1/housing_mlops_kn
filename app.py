from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import pickle
import uvicorn
app = FastAPI()

model = pickle.load(open('models/model.pkl','rb'))
preprocessor = pickle.load(open('models/preprocessor.pkl','rb'))

class PropertyInput(BaseModel):
    property_type: str
    sector: str
    bedRoom: int
    bathroom: int
    balcony: str
    agePossession: str
    built_up_area: int
    store_room: int
    pooja_room: int
    furnishing_type: str
    luxury_category: str
    floor_category: str


@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request):
    return open("templates/index.html").read()


def make_predictions(PropertyInput):

    query_data = {
        "property_type": PropertyInput.property_type,
        "sector": PropertyInput.sector,
        "bedRoom": PropertyInput.bedRoom,
        "bathroom": PropertyInput.bathroom,
        "balcony": PropertyInput.balcony,
        "agePossession": PropertyInput.agePossession,
        "built_up_area": PropertyInput.built_up_area,
        "store room": PropertyInput.store_room,
        "pooja room": PropertyInput.pooja_room,
        "furnishing_type": PropertyInput.furnishing_type,
        "luxury_category": PropertyInput.luxury_category,
        "floor_category": PropertyInput.floor_category
    }

    query_df = pd.DataFrame([query_data])
    print(query_data)
    transformed = preprocessor.transform(query_df)
    print(transformed.shape)
    predictions = model.predict(transformed)
    print(predictions)

    return predictions

@app.post("/predict/", response_class=HTMLResponse)
async def predict(
        property_type: str = Form(...),
        sector: str = Form(...),
        bedRoom: int = Form(...),
        bathroom: int = Form(...),
        balcony: str = Form(...),
        agePossession: str = Form(...),
        built_up_area: int = Form(...),
        store_room: int = Form(...),
        pooja_room: int = Form(...),
        furnishing_type: str = Form(...),
        luxury_category: str = Form(...),
        floor_category: str = Form(...)
):
    # Create a PropertyInput instance from form data
    property_input = PropertyInput(
        property_type=property_type,
        sector=sector,
        bedRoom=bedRoom,
        bathroom=bathroom,
        balcony=balcony,
        agePossession=agePossession,
        built_up_area=built_up_area,
        store_room=store_room,
        pooja_room=pooja_room,
        furnishing_type=furnishing_type,
        luxury_category=luxury_category,
        floor_category=floor_category
    )

    predictions = make_predictions(property_input)
    print(property_input.property_type)
    # Render the result in HTML
    result_html = f"<h2>Predicted Price: {predictions}</h2>"
    return result_html

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)