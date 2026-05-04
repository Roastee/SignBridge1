import os
import sys

# Force pure Python Protobuf implementation BEFORE Streamlit or MediaPipe loads
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from streamlit.web import cli

if __name__ == '__main__':
    # Launch Streamlit pointing to app.py
    sys.argv = ["streamlit", "run", "app.py"]
    sys.exit(cli.main())
