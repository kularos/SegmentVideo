from gradio_app import GradioWindow, GradioTab
import gradio as gr


# Example of a specialized tab
class AnalyticsTab(GradioTab):
    def build_ui(self):
        gr.Markdown("## 📈 Analytics Dashboard")
        input_box = gr.Textbox(label="Enter Data")
        output_box = gr.Textbox(label="Result")
        btn = gr.Button("Analyze")

        # Link logic to components
        btn.click(fn=self.analyze_data, inputs=input_box, outputs=output_box)

    def analyze_data(self, data):
        return f"Processed: {data.upper()}"


if __name__ == "__main__":
    # Define our 4 tabs
    my_tabs = [
        AnalyticsTab("Dashboard"),
        GradioTab("Model Config"),
        GradioTab("Logs"),
        GradioTab("Settings")
    ]

    # Create and run
    app = GradioWindow(title="Modular AI Suite", tabs=my_tabs)
    app.launch(server_port=7860)