# Slouch Detector
This project is a real-time posture analysis tool that uses your webcam to detect slouching and log biomechanical data.

## Prerequisites
*   **Python Version**: You must use a Python version **lower than 3.12** (e.g., Python 3.10 or 3.11).

## Installation
1.  **Set up your environment** (ensure Python < 3.12 is active).
2.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage
To launch the application, run the main script:

```bash
python main.py
```

## Building the Executable (.exe)

To create a standalone `.exe` file with the GUI:

1.  **Install PyInstaller**:
    ```bash
    pip install pyinstaller
    ```

2.  **Build the executable**:
    ```bash
    pyinstaller --noconsole --onefile main.py
    ```
    *   `--noconsole`: Hides the terminal window.
    *   `--onefile`: Bundles everything into a single `.exe` file.

3.  **Run the app**:
    Find the generated `main.exe` in the `dist/` folder.
