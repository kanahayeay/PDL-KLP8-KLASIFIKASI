from pydantic import BaseModel

class PredictRequest(BaseModel):
    text: str

class LabelResult(BaseModel):
    label: int
    probability: float

class PredictResponse(BaseModel):
    hate_speech: LabelResult
    abusive: LabelResult