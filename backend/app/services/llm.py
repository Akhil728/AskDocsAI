import os
import gc
import time
import torch
import logging
import requests

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# =========================
# TRANSFORMERS
# =========================

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        pipeline
    )

    TRANSFORMERS_AVAILABLE = True

except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not installed")

# =========================
# LLAMA CPP
# =========================

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True

except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not installed")

# =========================
# OPENAI
# =========================

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True

except ImportError:
    OPENAI_AVAILABLE = False


# ============================================================
# BASE CLASS
# ============================================================

class BaseLLM(ABC):

    @abstractmethod
    def generate_response(self, prompt: str,
                          max_tokens: int = 256,
                          temperature: float = 0.7) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


# ============================================================
# LOCAL LLM
# ============================================================

class LocalLLM(BaseLLM):

    def __init__(self, model_path: Optional[str] = None):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.tokenizer = None
        self.pipeline = None

        self.model_name = None
        self._available = False

        self.model_path = model_path

        self.transformer_models = [
            "distilgpt2",
            "microsoft/DialoGPT-medium"
        ]

        self._initialize_model()

    # ========================================================

    def _initialize_model(self):

        logger.info(f"Loading local model on {self.device}")

        # ============================================
        # GGUF MODELS
        # ============================================

        if self.model_path and self.model_path.endswith(".gguf"):

            if not LLAMA_CPP_AVAILABLE:
                logger.warning("llama-cpp-python not installed")
                return

            if not os.path.exists(self.model_path):
                logger.warning(f"GGUF model not found: {self.model_path}")
                return

            try:
                logger.info(f"Loading GGUF model: {self.model_path}")

                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=2048,
                    n_threads=6,
                    verbose=False
                )

                self.model_name = os.path.basename(self.model_path)
                self._available = True

                logger.info(f"Loaded GGUF model: {self.model_name}")

                return

            except Exception as e:
                logger.error(f"Failed GGUF load: {e}")

        # ============================================
        # TRANSFORMERS MODELS
        # ============================================

        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers unavailable")
            return

        for model_name in self.transformer_models:

            try:
                logger.info(f"Trying transformers model: {model_name}")

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name
                )

                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1
                )

                self.model_name = model_name
                self._available = True

                logger.info(f"Loaded transformers model: {model_name}")

                return

            except Exception as e:
                logger.warning(f"Failed loading {model_name}: {e}")

        logger.error("No local model could be loaded")

    # ========================================================

    def generate_response(self,
                          prompt: str,
                          max_tokens: int = 256,
                          temperature: float = 0.7) -> str:

        if not self._available:
            return "Local model unavailable."

        try:

            # ====================================
            # LLAMA CPP
            # ====================================

            if LLAMA_CPP_AVAILABLE and isinstance(self.model, Llama):

                output = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["Question:", "User:"]
                )

                text = output["choices"][0]["text"]

                return text.strip()

            # ====================================
            # TRANSFORMERS
            # ====================================

            response = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True
            )

            text = response[0]["generated_text"]

            text = text.replace(prompt, "").strip()

            return text

        except Exception as e:
            logger.error(f"Generation error: {e}")

            return "Error generating response."

    # ========================================================

    def is_available(self) -> bool:
        return self._available

    # ========================================================

    def cleanup(self):

        try:

            if self.pipeline:
                del self.pipeline

            if self.model:
                del self.model

            if self.tokenizer:
                del self.tokenizer

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


# ============================================================
# OPENAI LLM
# ============================================================

class OpenAILLM(BaseLLM):

    def __init__(self):

        self.api_key = os.getenv("OPENAI_API_KEY")

        self.client = None
        self._available = False

        if not self.api_key:
            return

        if not OPENAI_AVAILABLE:
            return

        try:

            self.client = OpenAI(api_key=self.api_key)

            self._available = True

            logger.info("OpenAI initialized")

        except Exception as e:
            logger.error(f"OpenAI init failed: {e}")

    # ========================================================

    def generate_response(self,
                          prompt: str,
                          max_tokens: int = 256,
                          temperature: float = 0.7) -> str:

        try:

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI error: {e}")

            return "OpenAI request failed."

    # ========================================================

    def is_available(self) -> bool:
        return self._available


# ============================================================
# GROQ
# ============================================================

class GroqLLM(BaseLLM):

    def __init__(self):

        self.api_key = os.getenv("GROQ_API_KEY")

        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

        self._available = bool(self.api_key)

    # ========================================================

    def generate_response(self,
                          prompt: str,
                          max_tokens: int = 256,
                          temperature: float = 0.7) -> str:

        try:

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=60
            )

            data = response.json()

            if "choices" not in data:
                logger.error(f"Groq unexpected response: {data}")
                return "Groq request failed."
            return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"Groq error: {e}")

            return "Groq request failed."

    # ========================================================

    def is_available(self) -> bool:
        return self._available


# ============================================================
# MAIN SERVICE
# ============================================================

class LLMService:

    def __init__(self):

        self.local_llm = None
        self.openai_llm = None
        self.groq_llm = None

        self.primary_llm = None

        self._initialize()

    # ========================================================

    def _initialize(self):

        logger.info("Initializing LLM service")

        # ========================
        # LOCAL
        # ========================

        try:
            self.local_llm = LocalLLM()
            prefer_local = os.getenv("PREFER_LOCAL", "true").lower() == "true"
            if self.local_llm.is_available() and prefer_local:
                self.primary_llm = self.local_llm
                logger.info("Using LOCAL LLM")

        except Exception as e:
            logger.error(f"Local LLM failed: {e}")

        # ========================
        # GROQ
        # ========================

        if not self.primary_llm:

            try:

                self.groq_llm = GroqLLM()

                if self.groq_llm.is_available():
                    self.primary_llm = self.groq_llm
                    logger.info("Using GROQ LLM")

            except Exception as e:
                logger.error(f"Groq failed: {e}")

        # ========================
        # OPENAI
        # ========================

        if not self.primary_llm:

            try:

                self.openai_llm = OpenAILLM()

                if self.openai_llm.is_available():
                    self.primary_llm = self.openai_llm
                    logger.info("Using OPENAI LLM")

            except Exception as e:
                logger.error(f"OpenAI failed: {e}")

        # ========================

        if not self.primary_llm:
            logger.error("NO LLM AVAILABLE")

    # ========================================================

    def generate_answer(self,
                        context_chunks: List[Dict[str, Any]],
                        question: str) -> Dict[str, Any]:

        start = time.time()

        try:

            context = "\n\n".join([
                chunk.get("text", "")
                for chunk in context_chunks[:5]
            ])

            prompt = f"""
Context:
{context}

Question:
{question}

Answer based only on the context above.
"""

            if not self.primary_llm:

                return {
                    "success": False,
                    "answer": "No LLM available."
                }

            answer = self.primary_llm.generate_response(prompt)

            return {
                "success": True,
                "answer": answer,
                "response_time": time.time() - start
            }

        except Exception as e:

            logger.error(f"Generate answer failed: {e}")

            return {
                "success": False,
                "answer": "Failed generating answer."
            }

    # ========================================================

    def get_service_status(self):

        return {
            "local_available": self.local_llm.is_available() if self.local_llm else False,
            "groq_available": self.groq_llm.is_available() if self.groq_llm else False,
            "openai_available": self.openai_llm.is_available() if self.openai_llm else False,
            "primary_llm": type(self.primary_llm).__name__ if self.primary_llm else None
        }

    # ========================================================

    def cleanup(self):

        if self.local_llm:
            self.local_llm.cleanup()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()