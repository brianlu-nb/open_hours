import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_by_limit_len(text: str, limit: int):
    if text is None:
        logger.warning("text is None in get_by_limit_len")
        return ""
    text_split = text.split(" ")
    return " ".join(text_split[: limit])


def get_by_limit_len_title(text: str):
    return get_by_limit_len(text, 256)


def get_by_limit_len_content(text: str):
    return get_by_limit_len(text, 1536)

def remove_writing_quality_rational(result_json: dict):
    rationales = ["Clarity_and_Coherence_Evaluation_Rationale",
                  "Relevance_and_Focus_Evaluation_Rationale",
                  "Accuracy_and_Credibility_Evaluation_Rationale",
                  "Originality_and_Insightfulness_Evaluation_Rationale",
                  "Engagement_and_Interest_Evaluation_Rationale",
                  "Grammar_and_Style_Evaluation_Rationale",
                  "Structural_Integrity_Evaluation_Rationale",
                  "Argumentation_and_Evidence_Evaluation_Rationale",
                  "Overall_Evaluation_Rationale"]
    for rationale in rationales:
        if rationale in result_json:
            del result_json[rationale]
    return result_json

def remove_content_genre_reason(result_json: dict):
    reasons = ["reason"]
    for reason in reasons:
        if reason in result_json:
            del result_json[reason]
    return result_json