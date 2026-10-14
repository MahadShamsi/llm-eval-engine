from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = "sqlite:///evaluations.db"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class EvaluationLog(Base):
    __tablename__ = "evaluation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    target_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    target_response: Mapped[str] = mapped_column(Text, nullable=False)
    faithfulness_score: Mapped[int] = mapped_column(Integer, nullable=False)
    relevancy_score: Mapped[int] = mapped_column(Integer, nullable=False)
    toxicity_score: Mapped[int] = mapped_column(Integer, nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


Base.metadata.create_all(bind=engine)


def save_evaluation(
    target_prompt: str,
    target_response: str,
    faithfulness_score: int,
    relevancy_score: int,
    toxicity_score: int,
    reasoning: str,
) -> EvaluationLog:
    """Persist a scored evaluation and return the stored row."""
    record = EvaluationLog(
        target_prompt=target_prompt,
        target_response=target_response,
        faithfulness_score=faithfulness_score,
        relevancy_score=relevancy_score,
        toxicity_score=toxicity_score,
        reasoning=reasoning,
    )
    with SessionLocal() as session:
        session.add(record)
        session.commit()
        session.refresh(record)
        session.expunge(record)
        return record
