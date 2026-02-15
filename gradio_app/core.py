import gradio as gr

class GradioTab:
    """Base class for a UI Tab."""
    def __init__(self, tab_title):
        self.tab_title = tab_title

    def render(self):
        with gr.Tab(self.tab_title):
            self.build_ui()

    def build_ui(self):
        """Override this to add components."""
        gr.Markdown(f"### {self.tab_title} Tab")
        gr.Interface(fn=lambda x: x, inputs="text", outputs="text")

class GradioWindow:
    """Main Orchestrator for the Gradio App."""
    def __init__(self, title="Gradio Application", tabs=None):
        self.title = title
        self.tabs = tabs or []

    def build_window(self):
        """Prepares the Blocks object without launching yet."""
        with gr.Blocks(title=self.title) as demo:
            gr.Markdown(f"# {self.title}")
            with gr.Tabs():
                for tab in self.tabs:
                    tab.render()
        return demo

    def launch(self, **kwargs):
        demo = self.build_window()
        demo.launch(**kwargs)