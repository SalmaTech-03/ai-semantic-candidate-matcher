import fitz
import docx
import io
import magic
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SecureParser:
    @staticmethod
    def to_text(content: bytes) -> str:
        mime = magic.from_buffer(content, mime=True)
        try:
            if mime == "application/pdf":
                with fitz.open(stream=content, filetype="pdf") as doc:
                    return " ".join([page.get_text() for page in doc])
            elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(io.BytesIO(content))
                return "\n".join([p.text for p in doc.paragraphs])
            return ""
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            return ""