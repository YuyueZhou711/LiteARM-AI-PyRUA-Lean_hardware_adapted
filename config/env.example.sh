# Copy this file to config/env.sh and fill in the endpoint.  config/env.sh is
# git-ignored; never commit real credentials.

# OpenAI-compatible chat endpoint used by the PyRUA-Lean policy server.
export OPENAI_BASE_URL="https://api.example.com/v1"
export OPENAI_API_KEY="replace-with-your-key"
# The reference run used gpt-6.1-sol; set the model your endpoint serves.
export PYRUALEAN_MODEL="gpt-6.1-sol"

# Conda environment names created by scripts/setup.sh.
export PYRUALEAN_POLICY_ENV="pyrualean"
export ROBODOJO_EVAL_ENV="RoboDojo"
