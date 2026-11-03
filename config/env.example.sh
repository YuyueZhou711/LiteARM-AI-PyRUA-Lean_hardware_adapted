# Copy this file to config/env.sh and fill in the endpoint.  config/env.sh is
# git-ignored; never commit real credentials.

# OpenAI-compatible chat endpoint used by the PyRUA-Lean policy server.
export OPENAI_BASE_URL="https://your-endpoint.example/v1"
export OPENAI_API_KEY="replace-with-your-key"
export PYRUALEAN_MODEL="your-model-name"

# Conda environment names created by scripts/setup.sh.
export PYRUALEAN_POLICY_ENV="pyrualean"
export ROBODOJO_EVAL_ENV="RoboDojo"
