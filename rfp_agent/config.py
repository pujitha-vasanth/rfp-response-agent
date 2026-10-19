"""
All settings in one place.

Values come from the .env file in the project folder (copy .env.example to .env).
If a value is missing, the default shown here is used.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Which AI model provider to use:
#   "demo"   runs offline with simple rules instead of a real model (no key needed)
#   "azure"  uses Azure OpenAI (Microsoft Foundry)
#   "openai" uses OpenAI directly
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "demo").strip().lower()

# Azure OpenAI settings
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")      # e.g. https://my-resource.openai.azure.com
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")  # your deployment name

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

# The company we are writing the RFP response for (our fictional bidder)
COMPANY_NAME = os.getenv("COMPANY_NAME", "Brightpath Digital")

# Reviewer loop settings
PASS_SCORE = int(os.getenv("PASS_SCORE", "4"))        # reviewer score (1 to 5) needed to pass
MAX_DRAFTS = int(os.getenv("MAX_DRAFTS", "3"))        # first draft + up to 2 rewrites

# How many requirements are processed at the same time (parallel model calls)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))

# File locations
CAPABILITY_LIBRARY = PROJECT_ROOT / "data" / "capability_library.json"
SAMPLE_RFP_DIR = PROJECT_ROOT / "data" / "rfps"
GOLDEN_DIR = PROJECT_ROOT / "data" / "golden"
OUTPUT_DIR = PROJECT_ROOT / "output"
