#!/bin/bash

ollama serve &
echo 'ollama server started'
python3 entrypoint.py