# CCB demo

Simple demo of a multislot contextual bandit with a model in the loop. Start the demo, choose a strategy "e.g. I click on products only if they are cheap" and act accordingly. Click "next round" to learn and proceed to the next round, and watch the model slowly learn your preferences.

## Install

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Start demo

`sh start-server.sh`, followed by navigation to `http://localhost:8000/demo`. Docs available at `http://localhost:8000/docs`.

To reset the learner, stop and start the server.
