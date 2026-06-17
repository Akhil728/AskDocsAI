from dotenv import load_dotenv
load_dotenv()

import logging
import uvicorn

from startup import main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting RAG Application...")

        # main()  # commented out to prevent double FAISS init

        logger.info("🌐 Launching FastAPI server...")

        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )

    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")

    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")