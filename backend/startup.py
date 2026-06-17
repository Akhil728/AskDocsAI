"""
Enhanced startup script for Document Q&A RAG system
Run this before starting the FastAPI server
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up environment
os.environ.setdefault("PYTHONPATH", str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_system.log")
    ]
)

logger = logging.getLogger(__name__)


def check_environment():
    """Check if all required environment components are available."""

    logger.info("🔍 Checking environment...")

    issues = []

    # ---------------------------------------------------
    # Python version
    # ---------------------------------------------------

    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required, found {sys.version}")

    # ---------------------------------------------------
    # Required packages
    # ---------------------------------------------------

    required_packages = [
        "torch",
        "transformers",
        "sentence_transformers",
        "fastapi",
        "uvicorn",
        "faiss"
    ]

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            issues.append(f"Missing package: {package}")

    # ---------------------------------------------------
    # FAISS CHECK (CPU SAFE)
    # ---------------------------------------------------

    try:
        import faiss
        logger.info("✅ FAISS installed")

    except ImportError:
        issues.append("Missing FAISS package. Install with: pip install faiss-cpu")

    # ---------------------------------------------------
    # CUDA CHECK
    # ---------------------------------------------------

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ GPU Available: {gpu_name}")

            # Only check GPU FAISS when CUDA exists
            try:
                import faiss

                if hasattr(faiss, "StandardGpuResources"):
                    logger.info("✅ FAISS GPU support available")
                else:
                    logger.warning("⚠️ FAISS GPU not available, using CPU FAISS")

            except Exception as e:
                logger.warning(f"⚠️ FAISS GPU check failed: {e}")

        else:
            logger.warning("⚠️ CUDA not available, using CPU")

    except Exception as e:
        issues.append(f"PyTorch issue: {e}")

    # ---------------------------------------------------
    # Create directories
    # ---------------------------------------------------

    required_dirs = [
        "data/documents",
        "data/embeddings",
        "data/processed"
    ]

    for dir_path in required_dirs:
        full_path = project_root / dir_path

        if not full_path.exists():
            logger.info(f"📁 Creating directory: {full_path}")
            full_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------
    # Final result
    # ---------------------------------------------------

    if issues:
        logger.error("❌ Environment issues found:")

        for issue in issues:
            logger.error(f"  - {issue}")

        return False

    logger.info("✅ Environment check passed")
    return True


def initialize_database():
    """Initialize database."""

    logger.info("🗄️ Initializing database...")

    try:
        from app.database import init_db

        init_db()

        logger.info("✅ Database initialized")
        return True

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False


def initialize_services():
    """Initialize RAG services."""

    logger.info("🚀 Initializing RAG services...")

    try:
        from app.utils.service_manager import service_manager

        init_result = service_manager.initialize_services()

        if init_result["success"]:

            logger.info("✅ Services initialized successfully")

            for service in init_result["services_initialized"]:
                logger.info(
                    f"  - {service['service']}: "
                    f"{service['status']} "
                    f"({service['time']:.2f}s)"
                )

            gpu_info = init_result.get("gpu_info", {})

            if gpu_info.get("available"):

                for device in gpu_info.get("devices", []):
                    logger.info(
                        f"  - GPU {device['id']}: "
                        f"{device['name']} "
                        f"({device['memory_gb']}GB)"
                    )

            return True

        else:
            logger.error("❌ Service initialization failed")

            for error in init_result.get("errors", []):
                logger.error(f"  - {error}")

            return False

    except Exception as e:
        logger.error(f"❌ Service initialization error: {e}")
        return False


def test_system():
    """Run basic tests."""

    logger.info("🧪 Running system tests...")

    try:
        from app.services.rag_service import RAGService
        from app.services.llm import LLMService

        # Test RAG
        rag_service = RAGService()
        rag_stats = rag_service.get_stats()

        logger.info(f"✅ RAG Service test passed")

        # Test LLM
        llm_service = LLMService()

        llm_status = llm_service.get_service_status()

        logger.info(
            f"✅ LLM Service test passed: "
            f"Primary LLM = {llm_status.get('primary_llm')}"
        )

        return True

    except Exception as e:
        logger.error(f"❌ System test failed: {e}")
        return False


def main():
    """Main startup routine."""

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("🚀 Starting Document Q&A RAG System")
    logger.info("=" * 60)

    # Environment
    if not check_environment():
        logger.error("❌ Environment check failed")
        sys.exit(1)

    # Database
    if not initialize_database():
        logger.error("❌ Database initialization failed")
        sys.exit(1)

    # Services
    if not initialize_services():
        logger.warning("⚠️ Service initialization had issues")

    # Tests
    if not test_system():
        logger.warning("⚠️ Some system tests failed")

    total_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info(f"✅ Startup completed in {total_time:.2f}s")
    logger.info("🌐 Ready to start FastAPI server")
    logger.info("=" * 60)

    logger.info("")
    logger.info("Run server with:")
    logger.info("uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    logger.info("")


if __name__ == "__main__":
    main()