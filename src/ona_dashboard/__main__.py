"""
Main entry point for ONA Dashboard package.
"""

import sys
import subprocess


def main():
    """Main entry point for running the ONA Dashboard."""
    try:
        # Run the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app.py", 
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\nONA Dashboard stopped.")
    except Exception as e:
        print(f"Error running ONA Dashboard: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())