import plotly.io as pio

def save_visualization(fig, path):
    try:
        pio.write_image(fig, file=path)
        print(f"Visualization saved to {path}")
    except Exception as e:
        print(f"Failed to save visualization to {path}: {e}")

def save_interactive_visualization(fig, path):
    try:
        fig.write_html(path)
        print(f"Interactive visualization saved to {path}")
    except Exception as e:
        print(f"Failed to save interactive visualization to {path}: {e}")
