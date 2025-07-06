from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ml_module import get_recommendations  # Import ML function

app = FastAPI()

# Define input schema (all fields optional)
class UserQuery(BaseModel):
    keywords: str | None = None
    price: float | None = None
    stars: float | None = None
    reviews: int | None = None
    category_id: int | None = None
    isBestSeller: bool | None = None
    boughtInLastMonth: int | None = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Product Recommendation API"}

@app.post("/recommend")
def recommend_products(query: UserQuery):
    query_params = {k: v for k, v in query.dict().items() if v is not None}
    
    if not query_params:
        raise HTTPException(status_code=400, detail="At least one query parameter is required")
    
    # Get recommendations from ML model
    recommendations = get_recommendations(query_params)
    
    if not recommendations:
        raise HTTPException(status_code=404, detail="No products found matching your criteria")
    
    return {"recommendations": recommendations}