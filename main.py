from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from database import save_evaluation
from evaluator import evaluate_response

app = FastAPI(title="LLM Eval Engine")


class EvaluationRequest(BaseModel):
    target_prompt: str = Field(..., min_length=1)
    target_response: str = Field(..., min_length=1)


@app.post("/evaluate")
def evaluate(payload: EvaluationRequest) -> dict:
    try:
        scorecard = evaluate_response(payload.target_prompt, payload.target_response)
        save_evaluation(
            target_prompt=payload.target_prompt,
            target_response=payload.target_response,
            faithfulness_score=scorecard["faithfulness"],
            relevancy_score=scorecard["relevancy"],
            toxicity_score=scorecard["toxicity"],
            reasoning=scorecard["reasoning"],
        )
        return scorecard
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {exc}",
        ) from exc
