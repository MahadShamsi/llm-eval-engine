import pandas as pd
import requests
import streamlit as st

from database import engine

API_URL = "http://127.0.0.1:8000/evaluate"

st.set_page_config(page_title="Evaluation Engine", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("LLM Evaluation Dashboard")

if "last_scorecard" not in st.session_state:
    st.session_state.last_scorecard = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None


def load_evaluations() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM evaluation_logs ORDER BY created_at", engine)


def post_evaluation(target_prompt: str, target_response: str) -> dict:
    api_response = requests.post(
        API_URL,
        json={"target_prompt": target_prompt, "target_response": target_response},
        timeout=120,
    )
    api_response.raise_for_status()
    return api_response.json()


evaluations = load_evaluations()
metrics_tab, suite_tab = st.tabs(["📈 Metrics Dashboard", "🧪 Evaluation Suite"])

with metrics_tab:
    st.subheader("Average Faithfulness Score")
    if evaluations.empty:
        st.info("No evaluations yet. Run a test in the Evaluation Suite to populate the dashboard.")
    else:
        average_faithfulness = evaluations["faithfulness_score"].mean()
        st.metric("Overall average faithfulness", f"{average_faithfulness:.2f} / 5")
        chart_data = (
            evaluations.assign(
                created_at=pd.to_datetime(evaluations["created_at"], utc=True)
            )
            .set_index("created_at")[["faithfulness_score"]]
            .rename(columns={"faithfulness_score": "Average Faithfulness"})
        )
        daily_average = chart_data.resample("D").mean().dropna()
        st.line_chart(daily_average if not daily_average.empty else chart_data)

    st.subheader("Failed Tests (faithfulness < 3)")
    if evaluations.empty:
        st.caption("No evaluation history to filter.")
    else:
        failed_tests = evaluations[evaluations["faithfulness_score"] < 3].copy()
        failed_tests = failed_tests.sort_values("created_at", ascending=False)
        if failed_tests.empty:
            st.success("No failed tests. Every evaluation scored 3 or higher on faithfulness.")
        else:
            st.dataframe(failed_tests, use_container_width=True)

with suite_tab:
    manual_col, batch_col = st.columns(2)

    with manual_col:
        st.subheader("Manual Testing")
        prompt = st.text_area("Prompt", placeholder="Enter the target prompt")
        response = st.text_area("Response", placeholder="Enter the target response")

        if st.button("Evaluate", type="primary"):
            if not prompt.strip() or not response.strip():
                st.warning("Please provide both a Prompt and a Response.")
            else:
                try:
                    st.session_state.last_scorecard = post_evaluation(prompt, response)
                    st.rerun()
                except requests.RequestException as exc:
                    st.session_state.last_scorecard = None
                    st.error(f"Request to the evaluation API failed: {exc}")
                    if getattr(exc, "response", None) is not None:
                        try:
                            st.json(exc.response.json())
                        except ValueError:
                            st.code(exc.response.text)

        if st.session_state.last_scorecard is not None:
            st.success("Evaluation complete.")
            st.json(st.session_state.last_scorecard)

    with batch_col:
        st.subheader("Batch Evaluation")
        uploaded_file = st.file_uploader(
            "Upload a CSV with prompt and response columns",
            type=["csv"],
        )

        batch_df = None
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
            except (pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(f"Could not read the CSV file: {exc}")
                batch_df = None
            else:
                missing_columns = [
                    column
                    for column in ("prompt", "response")
                    if column not in batch_df.columns
                ]
                if missing_columns:
                    st.error(
                        "CSV must include 'prompt' and 'response' columns. "
                        f"Missing: {', '.join(missing_columns)}"
                    )
                    batch_df = None
                else:
                    st.dataframe(batch_df, use_container_width=True)

        if st.button("Run Batch Evaluation", disabled=batch_df is None or batch_df.empty):
            rows = list(batch_df.iterrows())
            total = len(rows)
            progress = st.progress(0, text="Starting batch evaluation...")
            results = []

            for index, (_, row) in enumerate(rows, start=1):
                target_prompt = str(row["prompt"]) if pd.notna(row["prompt"]) else ""
                target_response = str(row["response"]) if pd.notna(row["response"]) else ""
                record = {
                    "row": index,
                    "prompt": target_prompt,
                    "response": target_response,
                }
                try:
                    scorecard = post_evaluation(target_prompt, target_response)
                    record.update(scorecard)
                    record["status"] = "ok"
                except requests.RequestException as exc:
                    record["status"] = "error"
                    record["error"] = str(exc)

                results.append(record)
                progress.progress(
                    index / total,
                    text=f"Evaluating row {index} of {total}",
                )

            st.session_state.batch_results = pd.DataFrame(results)
            progress.progress(1.0, text="Batch evaluation complete.")
            st.success(f"Finished evaluating {total} row(s).")

        if st.session_state.batch_results is not None:
            st.subheader("Batch Results")
            st.dataframe(st.session_state.batch_results, use_container_width=True)
